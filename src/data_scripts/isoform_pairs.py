import numpy as np
import pandas as pd
import itertools
from collections import Counter
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
        train_dataset, train_data, val_data, test_data, protein_to_idx, num_proteins, neg_pos_ratio
    """
    df = pd.read_csv(csv_file)

    total_interactions    = len(df)
    pos_interactions      = df['interact'].sum()
    negative_interactions = total_interactions - pos_interactions
    neg_pos_ratio         = negative_interactions / pos_interactions

    print(f"Loaded {total_interactions} protein pairs")
    print(f"Positive interactions: {pos_interactions} ({pos_interactions/total_interactions*100:.2f}%)")

    # Build vocabulary from full dataset (transductive: all proteins have embeddings)
    all_proteins   = sorted(set(df['ensp_1']).union(set(df['ensp_2'])))
    protein_to_idx = {protein: idx for idx, protein in enumerate(all_proteins)}
    num_proteins   = len(all_proteins)
    print(f"Total unique proteins: {num_proteins}")

    # Split at gene-pair level, stratified on whether the pair has any positive
    gene_pairs = df.groupby(['gene_1', 'gene_2'])['interact'].max().reset_index()
    train_val_pairs, test_pairs = train_test_split(
        gene_pairs, test_size=test_size,
        stratify=gene_pairs['interact'], random_state=random_state)
    train_pairs, val_pairs = train_test_split(
        train_val_pairs, test_size=val_size / (1 - test_size),
        stratify=train_val_pairs['interact'], random_state=random_state)

    train_data = df.merge(train_pairs[['gene_1', 'gene_2']], on=['gene_1', 'gene_2'])
    val_data   = df.merge(val_pairs[['gene_1', 'gene_2']],   on=['gene_1', 'gene_2'])
    test_data  = df.merge(test_pairs[['gene_1', 'gene_2']],  on=['gene_1', 'gene_2'])

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    train_dataset = ProteinInteractionDataset(train_data, protein_to_idx)
    return train_dataset, train_data, val_data, test_data, protein_to_idx, num_proteins, neg_pos_ratio


# =============================================================================
# Inductive isoform-level split
# =============================================================================

def load_and_prepare_data_inductive(csv_file, test_size=0.2, val_size=0.1, random_state=42,
                                     gene_level=False):
    """
    Load protein interaction data with inductive splitting.

    gene_level=False (default): each isoform is randomly assigned to exactly one
        partition (train / val / test). A pair (i, j) lands in the 'hardest' split
        of the two isoforms.
    gene_level=True: genes are partitioned; all isoforms of a gene land in the same
        split, eliminating within-gene data leakage. A pair lands in the hardest
        split of the two genes.

    Returns:
        train_dataset, train_data, val_data, test_data,
        protein_to_idx, num_proteins, neg_pos_ratio,
        train_proteins, val_proteins, test_proteins
    """
    df = pd.read_csv(csv_file)

    total_interactions = len(df)
    pos_interactions   = df['interact'].sum()
    print(f"Loaded {total_interactions} protein pairs")
    print(f"Positive interactions: {int(pos_interactions)} "
          f"({pos_interactions / total_interactions * 100:.2f}%)")

    all_proteins   = sorted(set(df['ensp_1']).union(set(df['ensp_2'])))
    protein_to_idx = {p: i for i, p in enumerate(all_proteins)}
    num_proteins   = len(all_proteins)
    print(f"Total unique isoforms: {num_proteins:,}")

    rng = np.random.default_rng(random_state)

    if gene_level:
        ensp_to_gene = {}
        for ensp, gene in zip(df['ensp_1'], df['gene_1']):
            ensp_to_gene[ensp] = gene
        for ensp, gene in zip(df['ensp_2'], df['gene_2']):
            ensp_to_gene[ensp] = gene

        all_genes = np.array(sorted(set(ensp_to_gene.values())))
        num_genes = len(all_genes)
        perm      = rng.permutation(num_genes)
        n_test    = int(num_genes * test_size)
        n_val     = int(num_genes * val_size)

        test_genes  = frozenset(all_genes[perm[:n_test]])
        val_genes   = frozenset(all_genes[perm[n_test:n_test + n_val]])

        print(f"Gene partition: {num_genes - n_test - n_val:,} train  {n_val:,} val  {n_test:,} test")

        rank1 = np.where(df['gene_1'].isin(test_genes), 2,
                np.where(df['gene_1'].isin(val_genes),  1, 0))
        rank2 = np.where(df['gene_2'].isin(test_genes), 2,
                np.where(df['gene_2'].isin(val_genes),  1, 0))

        test_proteins  = frozenset(p for p in all_proteins if ensp_to_gene.get(p) in test_genes)
        val_proteins   = frozenset(p for p in all_proteins if ensp_to_gene.get(p) in val_genes)
        train_proteins = frozenset(all_proteins) - test_proteins - val_proteins
    else:
        protein_arr = np.array(all_proteins)
        perm        = rng.permutation(num_proteins)
        n_test      = int(num_proteins * test_size)
        n_val       = int(num_proteins * val_size)

        test_proteins  = frozenset(protein_arr[perm[:n_test]])
        val_proteins   = frozenset(protein_arr[perm[n_test:n_test + n_val]])
        train_proteins = frozenset(protein_arr[perm[n_test + n_val:]])

        rank1 = np.where(df['ensp_1'].isin(test_proteins), 2,
                np.where(df['ensp_1'].isin(val_proteins),  1, 0))
        rank2 = np.where(df['ensp_2'].isin(test_proteins), 2,
                np.where(df['ensp_2'].isin(val_proteins),  1, 0))

    print(f"Isoform partition: {len(train_proteins):,} train  "
          f"{len(val_proteins):,} val  {len(test_proteins):,} test")

    pair_rank  = np.maximum(rank1, rank2)
    train_data = df[pair_rank == 0].reset_index(drop=True)
    val_data   = df[pair_rank == 1].reset_index(drop=True)
    test_data  = df[pair_rank == 2].reset_index(drop=True)

    print(f"Interaction split: train {len(train_data):,}  val {len(val_data):,}  test {len(test_data):,}")
    for name, split in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        if len(split):
            n_pos = int(split['interact'].sum())
            print(f"  {name}: {n_pos:,} pos  {len(split) - n_pos:,} neg  "
                  f"({100 * n_pos / len(split):.2f}% positive)")

    neg_pos_ratio = ((len(train_data) - int(train_data['interact'].sum()))
                     / max(int(train_data['interact'].sum()), 1))

    train_dataset = ProteinInteractionDataset(train_data, protein_to_idx)

    return (train_dataset, train_data, val_data, test_data,
            protein_to_idx, num_proteins, neg_pos_ratio,
            train_proteins, val_proteins, test_proteins)


# =============================================================================
# Split diagnostics
# =============================================================================

def diagnose_split(train_dataset, val_data, test_data):
    """Print a full diagnostic of the train/val/test split."""
    print("\n" + "="*70)
    print("SPLIT DIAGNOSTICS")
    print("="*70)

    train_data = train_dataset.data

    splits = {'Train': train_data, 'Val': val_data, 'Test': test_data}

    print(f"\n[1] SIZE & CLASS BALANCE")
    print(f"  {'Split':<8} {'Total':>10} {'Positives':>12} {'Negatives':>12} {'Pos %':>8}")
    print(f"  {'-'*54}")
    for name, df in splits.items():
        n     = len(df)
        n_pos = int(df['interact'].sum())
        pct   = 100 * n_pos / n if n > 0 else 0
        print(f"  {name:<8} {n:>10,} {n_pos:>12,} {n - n_pos:>12,} {pct:>7.3f}%")

    if all('gene_1' in df.columns for df in splits.values()):
        print(f"\n[2] GENE-PAIR LEVEL STRATIFICATION")
        print(f"  {'Split':<8} {'Gene pairs':>12} {'w/ any positive':>18} {'Pos GP %':>10}")
        print(f"  {'-'*52}")
        for name, df in splits.items():
            gp       = df.groupby(['gene_1', 'gene_2'])['interact'].max().reset_index()
            n_gp     = len(gp)
            n_gp_pos = int(gp['interact'].sum())
            print(f"  {name:<8} {n_gp:>12,} {n_gp_pos:>18,} {100*n_gp_pos/max(n_gp,1):>9.2f}%")
        print("  → Pos GP % should be roughly equal across splits.")

        print(f"\n[3] GENE-PAIR LEAKAGE CHECK")
        def gp_set(df):
            return set(frozenset((r.gene_1, r.gene_2)) for r in df.itertuples(index=False))
        train_gp = gp_set(train_data)
        tv = len(train_gp & gp_set(val_data))
        tt = len(train_gp & gp_set(test_data))
        vt = len(gp_set(val_data) & gp_set(test_data))
        print(f"  Train ∩ Val: {tv:,}  Train ∩ Test: {tt:,}  Val ∩ Test: {vt:,}  (all should be 0)")
        print(f"  {'✓ No leakage.' if tv + tt + vt == 0 else '✗ WARNING: overlap found.'}")

    print(f"\n[4] PROTEIN-LEVEL OVERLAP (expected in transductive LDM)")
    train_p = set(train_data['ensp_1']) | set(train_data['ensp_2'])
    val_p   = set(val_data['ensp_1'])   | set(val_data['ensp_2'])
    test_p  = set(test_data['ensp_1'])  | set(test_data['ensp_2'])
    print(f"  Train: {len(train_p):,}   Val: {len(val_p):,} ({len(train_p & val_p):,} in train)"
          f"   Test: {len(test_p):,} ({len(train_p & test_p):,} in train)")

    val_only  = val_p  - train_p
    test_only = test_p - train_p
    print(f"\n[5] COLD-START PROTEINS (only in val or test, never in train)")
    print(f"  Val-only: {len(val_only):,}   Test-only: {len(test_only):,}")
    if val_only or test_only:
        print("  ⚠ Unanchored embeddings — predictions on these are less reliable.")
    else:
        print("  ✓ All val/test proteins appear in training.")

    print(f"\n[6] DEGREE DISTRIBUTION (interacting proteins only)")
    for name, df in splits.items():
        pos = df[df['interact'] == 1]
        if len(pos) == 0:
            print(f"  {name}: no positives"); continue
        deg = Counter(itertools.chain(pos['ensp_1'], pos['ensp_2']))
        d   = list(deg.values())
        print(f"  {name:<6}: {len(pos):,} pos edges | {len(deg):,} proteins | "
              f"max {max(d)} | mean {np.mean(d):.1f}")

    print("\n" + "="*70 + "\n")


def diagnose_split_inductive(train_data, val_data, test_data,
                              train_proteins, val_proteins, test_proteins):
    """Print diagnostics for an isoform-level inductive split."""
    print("\n" + "=" * 70)
    print("INDUCTIVE SPLIT DIAGNOSTICS")
    print("=" * 70)

    # [1] Partition sizes and overlap
    tv = len(train_proteins & val_proteins)
    tt = len(train_proteins & test_proteins)
    vt = len(val_proteins   & test_proteins)
    print(f"\n[1] ISOFORM PARTITION")
    print(f"  Train: {len(train_proteins):,}   Val: {len(val_proteins):,}   Test: {len(test_proteins):,}")
    print(f"  Train∩Val: {tv}  Train∩Test: {tt}  Val∩Test: {vt}  (all must be 0)")
    print(f"  {'✓ No isoform leakage.' if tv + tt + vt == 0 else '✗ WARNING: overlap found.'}")

    # [2] Class balance
    splits = {'Train': train_data, 'Val': val_data, 'Test': test_data}
    print(f"\n[2] CLASS BALANCE")
    print(f"  {'Split':<8} {'Total':>10} {'Positives':>12} {'Negatives':>12} {'Pos %':>8}")
    print(f"  {'-' * 54}")
    for name, df in splits.items():
        n     = len(df)
        n_pos = int(df['interact'].sum())
        pct   = 100 * n_pos / n if n > 0 else 0.0
        print(f"  {name:<8} {n:>10,} {n_pos:>12,} {n - n_pos:>12,} {pct:>7.3f}%")

    # [3] Pair composition
    print(f"\n[3] PAIR COMPOSITION")
    print(f"  {'Split':<8} {'both-train':>12} {'cross (1 unseen)':>18} {'both-unseen':>14}")
    print(f"  {'-' * 56}")
    for name, df in splits.items():
        if len(df) == 0:
            continue
        r1 = np.where(df['ensp_1'].isin(train_proteins), 0, 1)
        r2 = np.where(df['ensp_2'].isin(train_proteins), 0, 1)
        both_train  = int(((r1 == 0) & (r2 == 0)).sum())
        cross       = int(((r1 + r2) == 1).sum())
        both_unseen = int(((r1 == 1) & (r2 == 1)).sum())
        print(f"  {name:<8} {both_train:>12,} {cross:>18,} {both_unseen:>14,}")
    print("  → Val/Test 'both-train' count should be 0.")

    # [4] Degree distribution on positive edges
    print(f"\n[4] DEGREE DISTRIBUTION (positive edges only)")
    for name, df in splits.items():
        pos = df[df['interact'] == 1]
        if len(pos) == 0:
            print(f"  {name}: no positives")
            continue
        deg = Counter(itertools.chain(pos['ensp_1'], pos['ensp_2']))
        d   = list(deg.values())
        print(f"  {name:<6}: {len(pos):,} pos edges | {len(deg):,} isoforms | "
              f"max {max(d)} | mean {np.mean(d):.1f}")

    print("\n" + "=" * 70 + "\n")
