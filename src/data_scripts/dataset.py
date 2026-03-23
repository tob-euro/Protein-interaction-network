import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from sklearn.model_selection import train_test_split

class ProteinInteractionDataset(Dataset):
    """Dataset for protein-protein interactions"""
    
    def __init__(self, data, protein_to_idx):
        self.data = data
        self.protein_to_idx = protein_to_idx
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        protein1_idx = self.protein_to_idx[row['ensp_1']]
        protein2_idx = self.protein_to_idx[row['ensp_2']]
        interaction = torch.tensor(row['interact'], dtype=torch.float32)
        
        return protein1_idx, protein2_idx, interaction
    
class WeightedProteinInteractionDataset(ProteinInteractionDataset):
    def __init__(self, data, protein_to_idx):
        super().__init__(data, protein_to_idx)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        protein1_idx = self.protein_to_idx[row['ensp_1']]
        protein2_idx = self.protein_to_idx[row['ensp_2']]
        interaction = torch.tensor(row['interact'], dtype=torch.float32)
        pi = torch.tensor(row['pi'], dtype=torch.float32)

        return protein1_idx, protein2_idx, interaction, pi

def augment_with_synthetic_negatives(csv_file, output_file, random_state=42):
    """
    Augments a protein interaction dataset with synthetic negative gene pairs.
    Samples unmeasured gene combinations (using genes already in the dataset),
    expands them to all isoform-level pairs, and saves the augmented CSV.

    Only needs to be run once before training.
    """
    df = pd.read_csv(csv_file)

    total_interactions = len(df)
    pos_interactions = df['interact'].sum()
    negative_interactions = total_interactions - pos_interactions
    neg_pos_ratio = negative_interactions / pos_interactions

    print(f"Loaded {total_interactions} protein pairs")
    print(f"Positive interactions: {pos_interactions} ({pos_interactions/total_interactions*100:.2f}%)")
    print(f"Existing neg:pos ratio: {neg_pos_ratio:.2f}")

    # Build per-gene isoform lookup
    gene_to_isoforms = defaultdict(dict)  # gene -> {ensp: enst}
    for _, row in df.iterrows():
        gene_to_isoforms[row['gene_1']][row['ensp_1']] = row['enst_1']
        gene_to_isoforms[row['gene_2']][row['ensp_2']] = row['enst_2']

    all_genes = sorted(gene_to_isoforms.keys())
    existing_gene_pairs_undirected = set(zip(df['gene_1'], df['gene_2']))
    existing_gene_pairs_undirected |= {(g2, g1) for g1, g2 in existing_gene_pairs_undirected}

    # Estimate how many gene pairs to sample
    avg_isoforms_per_gene = np.mean([len(v) for v in gene_to_isoforms.values()])
    avg_isoform_pairs_per_gene_pair = avg_isoforms_per_gene ** 2
    print(f"Average isoforms per gene: {avg_isoforms_per_gene:.2f}")
    print(f"Estimated isoform pairs per synthetic gene pair: {avg_isoform_pairs_per_gene_pair:.1f}")

    target_total_negatives = int(2 * negative_interactions)
    additional_negatives_needed = max(0, target_total_negatives - negative_interactions)
    gene_pairs_to_sample = max(1, int(np.ceil(additional_negatives_needed / avg_isoform_pairs_per_gene_pair)))
    print(f"Additional isoform-level negatives needed: {additional_negatives_needed}")
    print(f"Sampling ~{gene_pairs_to_sample} synthetic gene pairs to reach target")

    # Sample unmeasured gene pairs via rejection sampling
    rng = np.random.default_rng(random_state)
    sampled_gene_pairs = []
    attempts = 0
    max_attempts = gene_pairs_to_sample * 20

    while len(sampled_gene_pairs) < gene_pairs_to_sample and attempts < max_attempts:
        g1, g2 = rng.choice(all_genes, size=2, replace=False)
        if (g1, g2) not in existing_gene_pairs_undirected:
            sampled_gene_pairs.append((g1, g2))
            existing_gene_pairs_undirected.add((g1, g2))
        attempts += 1

    print(f"Sampled {len(sampled_gene_pairs)} synthetic gene pairs ({attempts} attempts)")

    # Expand to isoform-level pairs
    synthetic_rows = []
    for gene_1, gene_2 in sampled_gene_pairs:
        for ensp_1, enst_1 in gene_to_isoforms[gene_1].items():
            for ensp_2, enst_2 in gene_to_isoforms[gene_2].items():
                synthetic_rows.append({
                    'gene_1': gene_1, 'gene_2': gene_2,
                    'ensp_1': ensp_1, 'ensp_2': ensp_2,
                    'enst_1': enst_1, 'enst_2': enst_2,
                    'pi': 0.0, 'interact': 0
                })

    synthetic_df = pd.DataFrame(synthetic_rows)
    print(f"Synthetic isoform pairs generated: {len(synthetic_df)}")

    # Trim to target if overshot
    if len(synthetic_df) > additional_negatives_needed:
        synthetic_df = synthetic_df.sample(n=additional_negatives_needed, random_state=random_state)
        print(f"Trimmed to {len(synthetic_df)} synthetic pairs to match neg:pos ratio")

    df_augmented = pd.concat([df, synthetic_df], ignore_index=True)
    total_neg = (df_augmented['interact'] == 0).sum()
    total_pos = df_augmented['interact'].sum()
    print(f"Augmented dataset: {len(df_augmented)} pairs | neg:pos = {total_neg/total_pos:.2f}")

    df_augmented.to_csv(output_file, index=False)
    print(f"Saved augmented dataset to {output_file}")



def load_and_prepare_data(csv_file, test_size=0.2, val_size=0.1, random_state=42):
    """
    Load an (optionally pre-augmented) protein interaction CSV and prepare for training.
    Split at gene-pair level to prevent data leakage.
    Stratified on whether a gene pair has any positive isoform interaction.

    Returns:
    - train_data, val_data, test_data: dataframes
    - protein_to_idx: mapping from protein ID to index
    - num_proteins: number of unique proteins
    - neg_pos_ratio: ratio of negatives to positives
    """
    df = pd.read_csv(csv_file)

    total_interactions = len(df)
    pos_interactions = df['interact'].sum()
    negative_interactions = total_interactions - pos_interactions
    neg_pos_ratio = negative_interactions / pos_interactions

    print(f"Loaded {total_interactions} protein pairs")
    print(f"Positive interactions: {pos_interactions} ({pos_interactions/total_interactions*100:.2f}%)")

    all_proteins = sorted(set(df['ensp_1']).union(set(df['ensp_2'])))
    protein_to_idx = {protein: idx for idx, protein in enumerate(all_proteins)}
    num_proteins = len(all_proteins)
    print(f"Total unique proteins: {num_proteins}")

    gene_pairs_df = df.groupby(['gene_1', 'gene_2'])['interact'].max().reset_index()
    train_val_pairs, test_pairs = train_test_split(
        gene_pairs_df, test_size=test_size,
        stratify=gene_pairs_df['interact'], random_state=random_state
    )
    train_pairs, val_pairs = train_test_split(
        train_val_pairs, test_size=val_size / (1 - test_size),
        stratify=train_val_pairs['interact'], random_state=random_state
    )

    train_data = df.merge(train_pairs[['gene_1', 'gene_2']], on=['gene_1', 'gene_2'])
    val_data   = df.merge(val_pairs[['gene_1', 'gene_2']],   on=['gene_1', 'gene_2'])
    test_data  = df.merge(test_pairs[['gene_1', 'gene_2']],  on=['gene_1', 'gene_2'])

    for name, split in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        s_pos = split['interact'].sum()
        s_neg = (split['interact'] == 0).sum()
        print(f"{name}: {len(split)} pairs | pos={s_pos} | neg={s_neg} | neg:pos={s_neg/s_pos:.2f}")

    return train_data, val_data, test_data, protein_to_idx, num_proteins, neg_pos_ratio

# def load_and_prepare_data(csv_file, test_size=0.2, val_size=0.1, random_state=42):

#     """
#     Load protein interaction data and prepare for training.
#     Split at gene-pair level to prevent data leakage.
#     Stratified on whether a gene pair has any positive isoform interaction.

#     Returns:
#     - train_data, val_data, test_data: dataframes
#     - protein_to_idx: mapping fra protein ID til index
#     - num_proteins: antal unique proteiner
#     """
#     df = pd.read_csv(csv_file)

#     total_interactions = len(df)
#     pos_interactions = df['interact'].sum()
#     negative_interactions = total_interactions - pos_interactions
#     neg_pos_ratio = negative_interactions / pos_interactions

#     print(f"Loaded {total_interactions} protein pairs")
#     print(f"Positive interactions: {pos_interactions} ({pos_interactions/total_interactions*100:.2f}%)")

#     # Build vocabulary from full dataset (transductive: all proteins have embeddings)
#     all_proteins = sorted(set(df['ensp_1']).union(set(df['ensp_2'])))
#     protein_to_idx = {protein: idx for idx, protein in enumerate(all_proteins)}
#     num_proteins = len(all_proteins)
#     print(f"Total unique proteins: {num_proteins}")

#     # Split at gene-pair level, stratified on whether the pair has any positive
#     gene_pairs = df.groupby(['gene_1', 'gene_2'])['interact'].max().reset_index()
#     train_val_pairs, test_pairs = train_test_split(gene_pairs, test_size=test_size,
#                                                     stratify=gene_pairs['interact'], random_state=random_state)
#     train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=val_size/(1-test_size),
#                                                stratify=train_val_pairs['interact'], random_state=random_state)

#     # Map gene-pair split back to individual isoform-pair rows
#     train_data = df.merge(train_pairs[['gene_1', 'gene_2']], on=['gene_1', 'gene_2'])
#     val_data   = df.merge(val_pairs[['gene_1', 'gene_2']],   on=['gene_1', 'gene_2'])
#     test_data  = df.merge(test_pairs[['gene_1', 'gene_2']],  on=['gene_1', 'gene_2'])

#     print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

#     return train_data, val_data, test_data, protein_to_idx, num_proteins, neg_pos_ratio

if __name__ == "__main__":
    augment_with_synthetic_negatives("data/results_PHYSICAL_Prob_Model_16_02_26.csv", "data/results_PHYSICAL_Prob_Model_16_02_26_modified.csv")