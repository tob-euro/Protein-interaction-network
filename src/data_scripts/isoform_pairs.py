import numpy as np
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


class ClosedWorldTrainDataset(Dataset):
    """
    Training dataset that mixes tested negatives with freshly sampled unseen negatives.

    Each epoch, alpha fraction of the negatives are replaced with pairs sampled
    from the full isoform space that are absent from the dataset entirely.
    Total number of negatives stays the same as the original train split.

    Leakage prevention: unseen negatives are rejected if their gene pair appears
    in val or test (checked at gene level, so all isoform combos of those genes
    are blocked, not just the observed ones).
    """

    def __init__(self, train_df, protein_to_idx, all_isoforms, forbidden_gene_pairs, alpha=0.5, seed=42):
        self.protein_to_idx = protein_to_idx
        self.alpha          = alpha
        self.rng            = np.random.default_rng(seed)
        self.forbidden      = forbidden_gene_pairs  # set of frozenset({gene_a, gene_b})

        pos_df = train_df[train_df['interact'] == 1]
        neg_df = train_df[train_df['interact'] == 0]

        self.pos_e1 = pos_df['ensp_1'].values
        self.pos_e2 = pos_df['ensp_2'].values
        self.neg_e1 = neg_df['ensp_1'].values
        self.neg_e2 = neg_df['ensp_2'].values

        self.isoforms = np.array([x[0] for x in all_isoforms])
        self.genes    = np.array([x[1] for x in all_isoforms])
        self.n_iso    = len(self.isoforms)

        # All observed pairs — skipped when sampling unseen negatives
        self.observed = set(frozenset((r.ensp_1, r.ensp_2)) for r in train_df.itertuples(index=False))

        self._resample()

    def _resample(self):
        n_neg    = len(self.neg_e1)
        n_unseen = int(n_neg * self.alpha)
        n_tested = n_neg - n_unseen

        # Subsample tested negatives
        idx  = self.rng.choice(n_neg, size=n_tested, replace=False)
        s_e1 = list(self.neg_e1[idx])
        s_e2 = list(self.neg_e2[idx])

        # Sample unseen negatives with rejection
        collected = 0
        while collected < n_unseen:
            a = self.rng.integers(0, self.n_iso)
            b = self.rng.integers(0, self.n_iso)
            if a == b or self.genes[a] == self.genes[b]:
                continue
            if frozenset((self.genes[a], self.genes[b])) in self.forbidden:
                continue
            if frozenset((self.isoforms[a], self.isoforms[b])) in self.observed:
                continue
            s_e1.append(self.isoforms[a])
            s_e2.append(self.isoforms[b])
            collected += 1

        all_e1     = np.concatenate([self.pos_e1, s_e1])
        all_e2     = np.concatenate([self.pos_e2, s_e2])
        all_labels = np.concatenate([np.ones(len(self.pos_e1)), np.zeros(n_neg)]).astype(np.float32)

        perm        = self.rng.permutation(len(all_labels))
        self.e1     = all_e1[perm]
        self.e2     = all_e2[perm]
        self.labels = all_labels[perm]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.protein_to_idx[self.e1[idx]],
                self.protein_to_idx[self.e2[idx]],
                torch.tensor(self.labels[idx]))


def load_and_prepare_data(csv_file, test_size=0.2, val_size=0.1, random_state=42, alpha=0.0):
    """
    Load protein interaction data and prepare for training.
    Split at gene-pair level to prevent data leakage.
    Stratified on whether a gene pair has any positive isoform interaction.

    Args:
        alpha: fraction of train negatives replaced by unseen closed-world negatives.
               0.0 = original behaviour (no unseen negatives).

    Returns:
        train_dataset, val_data, test_data, protein_to_idx, num_proteins, neg_pos_ratio
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
    train_val_pairs, test_pairs = train_test_split(gene_pairs, test_size=test_size,
                                                    stratify=gene_pairs['interact'], random_state=random_state)
    train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=val_size/(1-test_size),
                                               stratify=train_val_pairs['interact'], random_state=random_state)

    train_data = df.merge(train_pairs[['gene_1', 'gene_2']], on=['gene_1', 'gene_2'])
    val_data   = df.merge(val_pairs[['gene_1', 'gene_2']],   on=['gene_1', 'gene_2'])
    test_data  = df.merge(test_pairs[['gene_1', 'gene_2']],  on=['gene_1', 'gene_2'])

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    if alpha > 0:
        # Build isoform universe from the CSV itself
        all_isoforms = (list(zip(df['ensp_1'], df['gene_1'])) +
                        list(zip(df['ensp_2'], df['gene_2'])))
        all_isoforms = list({k: v for k, v in all_isoforms}.items())  # deduplicate
        forbidden    = set(frozenset((r.gene_1, r.gene_2))
                           for r in pd.concat([val_data, test_data]).itertuples(index=False))
        train_dataset = ClosedWorldTrainDataset(train_data, protein_to_idx, all_isoforms,
                                                forbidden, alpha=alpha, seed=random_state)
    else:
        train_dataset = ProteinInteractionDataset(train_data, protein_to_idx)

    return train_dataset, train_data, val_data, test_data, protein_to_idx, num_proteins, neg_pos_ratio

# =============================================================================
# Split diagnostics
# =============================================================================

def diagnose_split(train_dataset, val_data, test_data):
    """
    Print a full diagnostic of the train/val/test split.
    Call immediately after load_and_prepare_data().
    """
    import itertools
    from collections import Counter

    print("\n" + "="*70)
    print("SPLIT DIAGNOSTICS")
    print("="*70)

    if hasattr(train_dataset, 'data'):
        train_data = train_dataset.data
    else:
        train_data = pd.DataFrame({
            'ensp_1':   train_dataset.pos_e1.tolist() + train_dataset.neg_e1.tolist(),
            'ensp_2':   train_dataset.pos_e2.tolist() + train_dataset.neg_e2.tolist(),
            'interact': [1]*len(train_dataset.pos_e1) + [0]*len(train_dataset.neg_e1),
        })

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