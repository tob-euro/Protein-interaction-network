import pandas as pd
import torch
from torch.utils.data import Dataset
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

def load_and_prepare_data(csv_file, test_size=0.2, val_size=0.1, random_state=42):
    """
    Load protein interaction data and prepare for training.
    Split at gene-pair level to prevent data leakage.
    Stratified on whether a gene pair has any positive isoform interaction.

    Returns:
    - train_data, val_data, test_data: dataframes
    - protein_to_idx: mapping fra protein ID til index
    - num_proteins: antal unique proteiner
    """
    df = pd.read_csv(csv_file)

    print(f"Loaded {len(df)} protein pairs")
    print(f"Positive interactions: {df['interact'].sum()} ({df['interact'].mean()*100:.2f}%)")

    # Build vocabulary from full dataset (transductive: all proteins have embeddings)
    all_proteins = sorted(set(df['ensp_1']).union(set(df['ensp_2'])))
    protein_to_idx = {protein: idx for idx, protein in enumerate(all_proteins)}
    num_proteins = len(all_proteins)
    print(f"Total unique proteins: {num_proteins}")

    # Split at gene-pair level, stratified on whether the pair has any positive
    gene_pairs = df.groupby(['gene_1', 'gene_2'])['interact'].max().reset_index()
    train_val_pairs, test_pairs = train_test_split(gene_pairs, test_size=test_size,
                                                    stratify=gene_pairs['interact'], random_state=random_state)
    train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=val_size/(1-test_size),
                                               stratify=train_val_pairs['interact'], random_state=random_state)

    # Map gene-pair split back to individual isoform-pair rows
    train_data = df.merge(train_pairs[['gene_1', 'gene_2']], on=['gene_1', 'gene_2'])
    val_data   = df.merge(val_pairs[['gene_1', 'gene_2']],   on=['gene_1', 'gene_2'])
    test_data  = df.merge(test_pairs[['gene_1', 'gene_2']],  on=['gene_1', 'gene_2'])

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    return train_data, val_data, test_data, protein_to_idx, num_proteins