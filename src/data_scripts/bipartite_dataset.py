import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


# =============================================================================
# Graph construction
# =============================================================================

def build_gene_isoform_graph(df):
    """
    Extract gene→isoform membership from a raw interaction DataFrame.

    The DataFrame must contain columns:
        gene_1, ensp_1   →  gene_1 has isoform ensp_1
        gene_2, ensp_2   →  gene_2 has isoform ensp_2

    Returns:
    - gene_to_idx:      mapping from gene ID to index
    - gene_to_isoforms: mapping from gene ID to set of isoform IDs
    """
    gene_to_isoforms = {}

    for _, row in df.iterrows():
        gene_to_isoforms.setdefault(row['gene_1'], set()).add(row['ensp_1'])
        gene_to_isoforms.setdefault(row['gene_2'], set()).add(row['ensp_2'])

    all_genes   = sorted(gene_to_isoforms.keys())
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}

    total_isoforms = sum(len(v) for v in gene_to_isoforms.values())
    print(f'Bipartite graph: {len(all_genes)} genes → {total_isoforms} membership edges')
    print(f'  Avg isoforms per gene: {total_isoforms / max(len(all_genes), 1):.2f}')

    return gene_to_idx, gene_to_isoforms


def sample_gene_isoform_pairs(gene_to_isoforms, gene_to_idx, protein_to_idx, genes_in_split,
                               neg_ratio=5, random_state=42):
    """
    Build a list of (gene_idx, protein_idx, label) triples for one split.

    Only genes whose canonical name appears in genes_in_split are included,
    so train/val/test splits stay consistent with the isoform-pair split.
    For each positive edge, neg_ratio negatives are sampled uniformly at random,
    rejecting any isoforms that actually belong to the gene.

    Returns:
    - list of (gene_idx, protein_idx, label)
    """
    rng          = np.random.default_rng(random_state)
    all_iso_arr  = np.array(list(protein_to_idx.keys()))

    triples = []

    for gene, isoforms in gene_to_isoforms.items():
        if gene not in genes_in_split:
            continue

        g_idx = gene_to_idx[gene]

        # Positive edges
        for iso in isoforms:
            # if iso not in protein_to_idx:
            #     continue                          # safety: should not happen
            triples.append((g_idx, protein_to_idx[iso], 1))

        # Negative edges
        n_neg        = len(isoforms) * neg_ratio
        sampled      = 0
        attempts     = 0
        max_attempts = n_neg * 20                 # avoid infinite loop

        while sampled < n_neg and attempts < max_attempts:
            candidates = all_iso_arr[rng.integers(0, len(all_iso_arr), size=n_neg - sampled + 10)]
            for iso in candidates:
                if iso not in isoforms and iso in protein_to_idx:
                    triples.append((g_idx, protein_to_idx[iso], 0))
                    sampled += 1
                    if sampled >= n_neg:
                        break
            attempts += n_neg

    positives = sum(1 for _, _, l in triples if l == 1)
    negatives = len(triples) - positives
    print(f'  Split ({len(genes_in_split)} genes): '
          f'{positives:,} positives  {negatives:,} negatives  '
          f'(ratio 1:{negatives // max(positives, 1)})')

    return triples


def prepare_bipartite_splits(df, protein_to_idx, train_data, val_data, test_data,
                              neg_ratio=5, random_state=42):
    """
    Convenience wrapper: builds the bipartite graph and produces
    train/val/test triple lists aligned with the isoform-pair splits.

    Returns:
    - gene_to_idx, train_triples, val_triples, test_triples
    """
    gene_to_idx, gene_to_isoforms = build_gene_isoform_graph(df)

    # Determine which genes belong to each split (union of gene_1 and gene_2)
    def gene_set(split_df):
        return set(split_df['gene_1']) | set(split_df['gene_2'])

    print('\nSampling bipartite edges per split:')
    train_triples = sample_gene_isoform_pairs(
        gene_to_isoforms, gene_to_idx, protein_to_idx,
        gene_set(train_data), neg_ratio, random_state)

    val_triples = sample_gene_isoform_pairs(
        gene_to_isoforms, gene_to_idx, protein_to_idx,
        gene_set(val_data), neg_ratio, random_state + 1)

    test_triples = sample_gene_isoform_pairs(
        gene_to_isoforms, gene_to_idx, protein_to_idx,
        gene_set(test_data), neg_ratio, random_state + 2)

    return gene_to_idx, train_triples, val_triples, test_triples


# =============================================================================
# Dataset
# =============================================================================

class GeneMembershipDataset(Dataset):
    """Dataset for gene–isoform membership triples."""

    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        gene_idx, protein_idx, label = self.triples[idx]
        return (
            torch.tensor(gene_idx,    dtype=torch.long),
            torch.tensor(protein_idx, dtype=torch.long),
            torch.tensor(label,       dtype=torch.float32),
        )