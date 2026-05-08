import numpy as np
import torch
from torch.utils.data import Dataset


def build_gene_isoform_graph(df):
    """Extract gene→isoform membership from an interaction dataframe.

    Args:
        df: DataFrame with columns gene_1, ensp_1, gene_2, ensp_2.

    Returns:
        gene_to_idx: mapping from gene ID to dense index.
        gene_to_isoforms: mapping from gene ID to a set of its isoforms.
    """
    gene_to_isoforms = {}
    for g_col, i_col in (('gene_1', 'ensp_1'), ('gene_2', 'ensp_2')):
        for g, iso in zip(df[g_col].values, df[i_col].values):
            gene_to_isoforms.setdefault(g, set()).add(iso)

    all_genes   = sorted(gene_to_isoforms)
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}

    n_edges = sum(len(v) for v in gene_to_isoforms.values())
    print(f"Bipartite graph: {len(all_genes)} genes → {n_edges} membership edges")
    print(f"  Avg isoforms per gene: {n_edges / max(len(all_genes), 1):.2f}")
    return gene_to_idx, gene_to_isoforms


def sample_gene_isoform_pairs(gene_to_isoforms, gene_to_idx, protein_to_idx,
                              genes_in_split, neg_ratio=5, random_state=42):
    """Build (gene_idx, protein_idx, label) triples for one split.

    Args:
        gene_to_isoforms: gene → set of member isoforms.
        gene_to_idx: gene → dense index.
        protein_to_idx: isoform → dense index.
        genes_in_split: only sample for genes appearing here.
        neg_ratio: negatives per positive membership edge.
        random_state: seed for the RNG.

    Returns:
        list of (gene_idx, protein_idx, label) triples.
    """
    rng      = np.random.default_rng(random_state)
    iso_keys = np.array(list(protein_to_idx.keys()))
    iso_vals = np.array(list(protein_to_idx.values()))
    triples  = []

    for gene, isoforms in gene_to_isoforms.items():
        if gene not in genes_in_split:
            continue
        g_idx    = gene_to_idx[gene]
        included = [iso for iso in isoforms if iso in protein_to_idx]
        if not included:
            continue

        for iso in included:
            triples.append((g_idx, protein_to_idx[iso], 1))

        # Pre-compute the non-member pool once per gene (no rejection sampling loop).
        non_member_pool = iso_vals[np.array([k not in isoforms for k in iso_keys])]
        if len(non_member_pool) == 0:
            continue
        n_neg   = len(included) * neg_ratio
        sampled = rng.choice(non_member_pool, size=n_neg,
                             replace=len(non_member_pool) < n_neg)
        triples.extend((g_idx, int(idx), 0) for idx in sampled)

    n_pos = sum(1 for _, _, l in triples if l == 1)
    n_neg = len(triples) - n_pos
    print(f"  Split ({len(genes_in_split)} genes): "
          f"{n_pos:,} pos  {n_neg:,} neg  (1:{n_neg // max(n_pos, 1)})")
    return triples


def prepare_gene_isoform_splits(df, protein_to_idx, train_data, val_data, test_data,
                                neg_ratio=5, random_state=42,
                                inductive=False, held_out_isoforms=None):
    """Build gene–isoform train/val/test triples.

    Args:
        df: full interaction DataFrame (used to derive gene→isoform membership).
        protein_to_idx: isoform → dense index.
        train_data, val_data, test_data: iso–iso split DataFrames.
        neg_ratio: negatives per positive membership edge.
        random_state: seed (offsets per split for reproducibility).
        inductive: if True, all gene–isoform edges go to training (annotation
            prior, not experimental data) — held-out isoforms still receive
            bipartite anchors.
        held_out_isoforms: optional set, only used for the print line.

    Returns:
        gene_to_idx, train_triples, val_triples, test_triples, neg_pos_ratio
        (val/test are empty in inductive mode).
    """
    gene_to_idx, gene_to_isoforms = build_gene_isoform_graph(df)

    def gene_set(split_df):
        return set(split_df['gene_1']) | set(split_df['gene_2'])

    if inductive:
        if held_out_isoforms:
            print(f"\n  Inductive: gene–isoform edges for {len(held_out_isoforms):,} "
                  f"held-out isoforms included as anchors")
        all_genes = gene_set(train_data) | gene_set(val_data) | gene_set(test_data)
        print("\nSampling gene–isoform edges (all → train):")
        train_triples = sample_gene_isoform_pairs(
            gene_to_isoforms, gene_to_idx, protein_to_idx,
            all_genes, neg_ratio, random_state)
        val_triples, test_triples = [], []
    else:
        print("\nSampling gene–isoform edges per split:")
        train_triples = sample_gene_isoform_pairs(
            gene_to_isoforms, gene_to_idx, protein_to_idx,
            gene_set(train_data), neg_ratio, random_state)
        val_triples = sample_gene_isoform_pairs(
            gene_to_isoforms, gene_to_idx, protein_to_idx,
            gene_set(val_data), neg_ratio, random_state + 1)
        test_triples = sample_gene_isoform_pairs(
            gene_to_isoforms, gene_to_idx, protein_to_idx,
            gene_set(test_data), neg_ratio, random_state + 2)

    n_pos = sum(1 for _, _, l in train_triples if l == 1)
    neg_pos_ratio = (len(train_triples) - n_pos) / max(n_pos, 1)
    return gene_to_idx, train_triples, val_triples, test_triples, neg_pos_ratio


class GeneIsoformDataset(Dataset):
    """Gene–isoform membership triples."""

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
