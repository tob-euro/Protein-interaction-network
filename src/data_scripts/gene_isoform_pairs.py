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

    Only genes whose canonical name appears in genes_in_split are included.
    For each positive edge, neg_ratio negatives are sampled from a pre-computed
    pool of non-member isoforms (no rejection loop).
    """
    rng          = np.random.default_rng(random_state)
    all_iso_keys = np.array(list(protein_to_idx.keys()))
    all_iso_vals = np.array(list(protein_to_idx.values()))
    triples      = []

    for gene, isoforms in gene_to_isoforms.items():
        if gene not in genes_in_split:
            continue

        g_idx             = gene_to_idx[gene]
        included_isoforms = [iso for iso in isoforms if iso in protein_to_idx]
        if not included_isoforms:
            continue

        for iso in included_isoforms:
            triples.append((g_idx, protein_to_idx[iso], 1))

        # Pre-compute non-member pool for this gene (no rejection loop)
        member_set       = set(isoforms)
        non_member_mask  = np.array([k not in member_set for k in all_iso_keys])
        non_member_pool  = all_iso_vals[non_member_mask]

        if len(non_member_pool) == 0:
            continue

        n_neg    = len(included_isoforms) * neg_ratio
        replace  = len(non_member_pool) < n_neg
        sampled  = rng.choice(non_member_pool, size=n_neg, replace=replace)
        for idx in sampled:
            triples.append((g_idx, int(idx), 0))

    positives = sum(1 for _, _, l in triples if l == 1)
    negatives = len(triples) - positives
    print(f'  Split ({len(genes_in_split)} genes): '
          f'{positives:,} positives  {negatives:,} negatives  '
          f'(ratio 1:{negatives // max(positives, 1)})')

    return triples


def prepare_gene_isoform_splits(df, protein_to_idx, train_data, val_data, test_data,
                                 neg_ratio=5, random_state=42,
                                 inductive=False, held_out_isoforms=None):
    """
    Build gene–isoform training (and optionally val) data.

    Transductive mode (inductive=False):
        Splits gene–isoform edges into train/val/test aligned with the
        isoform-pair split by gene set.

    Inductive mode (inductive=True):
        ALL gene–isoform edges go to training — no val/test split.
        Gene membership is always-known annotation, not experimental data.
        Gene–isoform edges for held-out isoforms are included; the bipartite
        loss anchors held-out isoform positions toward their parent gene.

    Returns:
        gene_to_idx, train_triples, val_triples, test_triples, neg_pos_ratio
        (In inductive mode, val_triples and test_triples are empty lists.)
    """
    gene_to_idx, gene_to_isoforms = build_gene_isoform_graph(df)

    def gene_set(split_df):
        return set(split_df['gene_1']) | set(split_df['gene_2'])

    if inductive:
        if held_out_isoforms:
            print(f'\n  Inductive: gene–isoform edges for {len(held_out_isoforms):,} '
                  f'held-out isoforms included as bipartite anchors')

        all_genes = gene_set(train_data) | gene_set(val_data) | gene_set(test_data)

        print(f'\nSampling gene-isoform edges (ALL → train):')
        train_triples = sample_gene_isoform_pairs(
            gene_to_isoforms, gene_to_idx, protein_to_idx,
            all_genes, neg_ratio, random_state)

        val_triples  = []
        test_triples = []

    else:
        print('\nSampling gene-isoform edges per split:')
        train_triples = sample_gene_isoform_pairs(
            gene_to_isoforms, gene_to_idx, protein_to_idx,
            gene_set(train_data), neg_ratio, random_state)

        val_triples = sample_gene_isoform_pairs(
            gene_to_isoforms, gene_to_idx, protein_to_idx,
            gene_set(val_data), neg_ratio, random_state + 1)

        test_triples = sample_gene_isoform_pairs(
            gene_to_isoforms, gene_to_idx, protein_to_idx,
            gene_set(test_data), neg_ratio, random_state + 2)

    n_pos         = sum(1 for _, _, l in train_triples if l == 1)
    n_neg         = len(train_triples) - n_pos
    neg_pos_ratio = n_neg / max(n_pos, 1)

    return gene_to_idx, train_triples, val_triples, test_triples, neg_pos_ratio


# =============================================================================
# Dataset
# =============================================================================

class GeneIsoformDataset(Dataset):
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
