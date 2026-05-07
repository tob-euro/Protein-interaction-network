import itertools
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class ProteinInteractionDataset(Dataset):
    """Iso–iso interaction triples (protein1_idx, protein2_idx, label)."""

    def __init__(self, data, protein_to_idx):
        self.data = data
        self.protein_to_idx = protein_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return (
            self.protein_to_idx[row['ensp_1']],
            self.protein_to_idx[row['ensp_2']],
            torch.tensor(row['interact'], dtype=torch.float32),
        )


def load_and_prepare_data(csv_file, test_size=0.2, val_size=0.1, random_state=42):
    """Transductive split: gene-pair level, stratified on positive presence.

    Args:
        csv_file: path to interaction CSV with gene_1, gene_2, ensp_1, ensp_2, interact.
        test_size: fraction of gene pairs held out for testing.
        val_size: fraction of gene pairs held out for validation.
        random_state: seed for the train_test_split.

    Returns:
        train_dataset, train_data, val_data, test_data,
        protein_to_idx, num_proteins, neg_pos_ratio
    """
    df = pd.read_csv(csv_file)

    n_total = len(df)
    n_pos   = int(df['interact'].sum())
    neg_pos_ratio = (n_total - n_pos) / max(n_pos, 1)
    print(f"Loaded {n_total} protein pairs ({n_pos} positives, "
          f"{n_pos / n_total * 100:.2f}%)")

    # Transductive: every protein gets an embedding, so we build the full vocabulary up-front.
    all_proteins   = sorted(set(df['ensp_1']) | set(df['ensp_2']))
    protein_to_idx = {p: i for i, p in enumerate(all_proteins)}
    num_proteins   = len(all_proteins)
    print(f"Total unique proteins: {num_proteins}")

    # Split at gene-pair level, stratified on whether the pair has any positive.
    gene_pairs = df.groupby(['gene_1', 'gene_2'])['interact'].max().reset_index()
    train_val_pairs, test_pairs = train_test_split(
        gene_pairs, test_size=test_size,
        stratify=gene_pairs['interact'], random_state=random_state,
    )
    train_pairs, val_pairs = train_test_split(
        train_val_pairs, test_size=val_size / (1 - test_size),
        stratify=train_val_pairs['interact'], random_state=random_state,
    )

    train_data = df.merge(train_pairs[['gene_1', 'gene_2']], on=['gene_1', 'gene_2'])
    val_data   = df.merge(val_pairs[['gene_1', 'gene_2']],   on=['gene_1', 'gene_2'])
    test_data  = df.merge(test_pairs[['gene_1', 'gene_2']],  on=['gene_1', 'gene_2'])
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    train_dataset = ProteinInteractionDataset(train_data, protein_to_idx)
    return (train_dataset, train_data, val_data, test_data,
            protein_to_idx, num_proteins, neg_pos_ratio)


def load_and_prepare_data_inductive(csv_file, test_size=0.2, val_size=0.1, random_state=42):
    """Inductive split: genes are partitioned, so all isoforms of a gene end up
    in the same split. A pair lands in the hardest split of its two endpoints,
    eliminating within-gene leakage.

    Args:
        csv_file: path to interaction CSV.
        test_size: fraction of genes held out for testing.
        val_size: fraction of genes held out for validation.
        random_state: seed for the gene permutation.

    Returns:
        train_dataset, train_data, val_data, test_data,
        protein_to_idx, num_proteins, neg_pos_ratio,
        train_proteins, val_proteins, test_proteins
    """
    df = pd.read_csv(csv_file)
    n_total = len(df)
    n_pos   = int(df['interact'].sum())
    print(f"Loaded {n_total} protein pairs ({n_pos} positives, "
          f"{n_pos / n_total * 100:.2f}%)")

    all_proteins   = sorted(set(df['ensp_1']) | set(df['ensp_2']))
    protein_to_idx = {p: i for i, p in enumerate(all_proteins)}
    num_proteins   = len(all_proteins)
    print(f"Total unique isoforms: {num_proteins:,}")

    # Map each isoform to its canonical gene (column 1 first, column 2 may overwrite).
    ensp_to_gene = dict(zip(df['ensp_1'], df['gene_1']))
    ensp_to_gene.update(zip(df['ensp_2'], df['gene_2']))

    # Partition genes randomly into train/val/test.
    rng       = np.random.default_rng(random_state)
    all_genes = np.array(sorted(set(ensp_to_gene.values())))
    n_genes   = len(all_genes)
    perm      = rng.permutation(n_genes)
    n_test    = int(n_genes * test_size)
    n_val     = int(n_genes * val_size)

    test_genes = frozenset(all_genes[perm[:n_test]])
    val_genes  = frozenset(all_genes[perm[n_test:n_test + n_val]])
    print(f"Gene partition: {n_genes - n_test - n_val:,} train  "
          f"{n_val:,} val  {n_test:,} test")

    # Rank each endpoint (0=train, 1=val, 2=test); a pair takes the maximum rank.
    rank1 = np.where(df['gene_1'].isin(test_genes), 2,
            np.where(df['gene_1'].isin(val_genes),  1, 0))
    rank2 = np.where(df['gene_2'].isin(test_genes), 2,
            np.where(df['gene_2'].isin(val_genes),  1, 0))

    test_proteins  = frozenset(p for p in all_proteins if ensp_to_gene.get(p) in test_genes)
    val_proteins   = frozenset(p for p in all_proteins if ensp_to_gene.get(p) in val_genes)
    train_proteins = frozenset(all_proteins) - test_proteins - val_proteins
    print(f"Isoform partition: {len(train_proteins):,} train  "
          f"{len(val_proteins):,} val  {len(test_proteins):,} test")

    pair_rank  = np.maximum(rank1, rank2)
    train_data = df[pair_rank == 0].reset_index(drop=True)
    val_data   = df[pair_rank == 1].reset_index(drop=True)
    test_data  = df[pair_rank == 2].reset_index(drop=True)

    print(f"Interaction split: train {len(train_data):,}  val {len(val_data):,}  "
          f"test {len(test_data):,}")
    for name, split in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        if len(split):
            n = int(split['interact'].sum())
            print(f"  {name}: {n:,} pos  {len(split) - n:,} neg "
                  f"({100 * n / len(split):.2f}% positive)")

    train_pos = int(train_data['interact'].sum())
    neg_pos_ratio = (len(train_data) - train_pos) / max(train_pos, 1)
    train_dataset = ProteinInteractionDataset(train_data, protein_to_idx)

    return (train_dataset, train_data, val_data, test_data,
            protein_to_idx, num_proteins, neg_pos_ratio,
            train_proteins, val_proteins, test_proteins)


def diagnose_split(train_dataset, val_data, test_data):
    """Diagnostics for a transductive (gene-pair level) split.

    Args:
        train_dataset: ProteinInteractionDataset over the training pairs.
        val_data: validation DataFrame.
        test_data: test DataFrame.
    """
    print("\n--- Split diagnostics ---")
    train_data = train_dataset.data
    splits = {'Train': train_data, 'Val': val_data, 'Test': test_data}

    print("\n[1] Size & class balance")
    print(f"  {'Split':<8} {'Total':>10} {'Pos':>12} {'Neg':>12} {'Pos %':>8}")
    for name, df in splits.items():
        n     = len(df)
        n_pos = int(df['interact'].sum())
        pct   = 100 * n_pos / n if n > 0 else 0
        print(f"  {name:<8} {n:>10,} {n_pos:>12,} {n - n_pos:>12,} {pct:>7.3f}%")

    if all('gene_1' in df.columns for df in splits.values()):
        print("\n[2] Gene-pair stratification")
        print(f"  {'Split':<8} {'Pairs':>12} {'w/ positive':>16} {'Pos %':>10}")
        for name, df in splits.items():
            gp = df.groupby(['gene_1', 'gene_2'])['interact'].max().reset_index()
            n_gp     = len(gp)
            n_gp_pos = int(gp['interact'].sum())
            print(f"  {name:<8} {n_gp:>12,} {n_gp_pos:>16,} "
                  f"{100 * n_gp_pos / max(n_gp, 1):>9.2f}%")

        print("\n[3] Gene-pair leakage (must be 0)")
        def gp_set(df):
            return {frozenset((r.gene_1, r.gene_2)) for r in df.itertuples(index=False)}
        train_gp = gp_set(train_data)
        tv = len(train_gp & gp_set(val_data))
        tt = len(train_gp & gp_set(test_data))
        vt = len(gp_set(val_data) & gp_set(test_data))
        print(f"  Train∩Val: {tv:,}  Train∩Test: {tt:,}  Val∩Test: {vt:,}")
        print(f"  {'OK' if tv + tt + vt == 0 else 'WARNING: overlap found'}")

    print("\n[4] Protein-level overlap (transductive — overlap expected)")
    train_p = set(train_data['ensp_1']) | set(train_data['ensp_2'])
    val_p   = set(val_data['ensp_1'])   | set(val_data['ensp_2'])
    test_p  = set(test_data['ensp_1'])  | set(test_data['ensp_2'])
    print(f"  Train: {len(train_p):,}  Val: {len(val_p):,} "
          f"({len(train_p & val_p):,} in train)  "
          f"Test: {len(test_p):,} ({len(train_p & test_p):,} in train)")

    val_only  = val_p  - train_p
    test_only = test_p - train_p
    print("\n[5] Cold-start proteins (only in val/test)")
    print(f"  Val-only: {len(val_only):,}   Test-only: {len(test_only):,}")
    if val_only or test_only:
        print("  Unanchored embeddings — predictions on these are less reliable.")

    print("\n[6] Degree distribution (positive edges)")
    for name, df in splits.items():
        pos = df[df['interact'] == 1]
        if pos.empty:
            print(f"  {name}: no positives")
            continue
        deg = Counter(itertools.chain(pos['ensp_1'], pos['ensp_2']))
        d   = list(deg.values())
        print(f"  {name:<6}: {len(pos):,} pos | {len(deg):,} proteins | "
              f"max {max(d)} | mean {np.mean(d):.1f}")
    print()


def diagnose_split_inductive(train_data, val_data, test_data,
                             train_proteins, val_proteins, test_proteins):
    """Diagnostics for an inductive (gene-level) split.

    Args:
        train_data, val_data, test_data: split DataFrames.
        train_proteins, val_proteins, test_proteins: isoform sets per split
            (must be disjoint).
    """
    print("\n--- Inductive split diagnostics ---")

    tv = len(train_proteins & val_proteins)
    tt = len(train_proteins & test_proteins)
    vt = len(val_proteins   & test_proteins)
    print("\n[1] Isoform partition")
    print(f"  Train: {len(train_proteins):,}  Val: {len(val_proteins):,}  "
          f"Test: {len(test_proteins):,}")
    print(f"  Train∩Val: {tv}  Train∩Test: {tt}  Val∩Test: {vt}  (must be 0)")
    print(f"  {'OK' if tv + tt + vt == 0 else 'WARNING: overlap found'}")

    splits = {'Train': train_data, 'Val': val_data, 'Test': test_data}
    print("\n[2] Class balance")
    print(f"  {'Split':<8} {'Total':>10} {'Pos':>12} {'Neg':>12} {'Pos %':>8}")
    for name, df in splits.items():
        n     = len(df)
        n_pos = int(df['interact'].sum())
        pct   = 100 * n_pos / n if n > 0 else 0.0
        print(f"  {name:<8} {n:>10,} {n_pos:>12,} {n - n_pos:>12,} {pct:>7.3f}%")

    print("\n[3] Pair composition")
    print(f"  {'Split':<8} {'both-train':>12} {'cross':>12} {'both-unseen':>14}")
    for name, df in splits.items():
        if df.empty:
            continue
        un1 = ~df['ensp_1'].isin(train_proteins)
        un2 = ~df['ensp_2'].isin(train_proteins)
        both_train  = int((~un1 & ~un2).sum())
        cross       = int((un1 ^ un2).sum())
        both_unseen = int((un1 & un2).sum())
        print(f"  {name:<8} {both_train:>12,} {cross:>12,} {both_unseen:>14,}")

    print("\n[4] Degree distribution (positive edges)")
    for name, df in splits.items():
        pos = df[df['interact'] == 1]
        if pos.empty:
            print(f"  {name}: no positives")
            continue
        deg = Counter(itertools.chain(pos['ensp_1'], pos['ensp_2']))
        d   = list(deg.values())
        print(f"  {name:<6}: {len(pos):,} pos | {len(deg):,} isoforms | "
              f"max {max(d)} | mean {np.mean(d):.1f}")
    print()
