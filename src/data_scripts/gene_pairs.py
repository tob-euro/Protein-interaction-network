"""
gene_gene_pairs.py — Gene-gene interaction data for the multimodal LDM.

Source: STRING (Search Tool for the Retrieval of Interacting Genes/Proteins)
        data/STRING_protein_pairs_wscores_physical.csv

Pipeline:
  1. Load STRING physical interaction pairs (ENSP IDs, pre-labeled interact column)
  2. Map ENSP → ENSG via data/gene-isoform_mapping_enst_ensp_ensg.csv
  3. Expand gene_to_idx with any STRING genes not already in the isoform vocabulary
  4. Split train/val/test (transductive) or put all in train (inductive)

Inductive mode:
  STRING gene-gene interactions are an external database, always available
  regardless of which isoforms are held out. All edges go to training.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


STRING_PATH  = "data/STRING_protein_pairs_wscores_physical.csv"
MAPPING_PATH = "data/gene-isoform_mapping_enst_ensp_ensg.csv"


# =============================================================================
# Loading and mapping
# =============================================================================

def load_string_pairs(gene_to_idx,
                      string_path=STRING_PATH,
                      mapping_path=MAPPING_PATH):
    """
    Load STRING interaction pairs, expand gene_to_idx with STRING-only genes,
    and map all pairs to gene indices.
    """
    print(f"  Loading STRING pairs from {string_path} ...")
    string_df = pd.read_csv(string_path)

    print(f"  Loading ENSP→ENSG mapping from {mapping_path} ...")
    mapping_df = pd.read_csv(mapping_path)
    ensp_to_ensg = dict(zip(mapping_df['ensp_id'], mapping_df['ensg_id']))

    ensg1 = string_df['protein1'].map(ensp_to_ensg)
    ensg2 = string_df['protein2'].map(ensp_to_ensg)

    mask      = ensg1.notna() & ensg2.notna()
    n_dropped = (~mask).sum()
    print(f"  ENSP→ENSG mappable: {mask.sum():,} / {len(string_df):,} "
          f"(dropped {n_dropped:,} pairs with unmappable ENSP)")

    ensg1  = ensg1[mask].values
    ensg2  = ensg2[mask].values
    labels = string_df['interact'][mask].values

    expanded = dict(gene_to_idx)
    next_idx = len(expanded)
    for ensg in set(ensg1) | set(ensg2):
        if ensg not in expanded:
            expanded[ensg] = next_idx
            next_idx += 1

    n_new = len(expanded) - len(gene_to_idx)
    print(f"  Gene vocabulary: {len(gene_to_idx):,} (isoform) "
          f"+ {n_new:,} (STRING-only) = {len(expanded):,} total")

    idx1 = [expanded[e] for e in ensg1]
    idx2 = [expanded[e] for e in ensg2]

    pairs = [(min(a, b), max(a, b), int(l)) for a, b, l in zip(idx1, idx2, labels)]

    n_pos = sum(1 for _, _, l in pairs if l == 1)
    n_neg = len(pairs) - n_pos
    print(f"  Total pairs: {len(pairs):,}  ({n_pos:,} pos  {n_neg:,} neg  "
          f"global ratio 1:{n_neg / max(n_pos, 1):.1f})")

    return pairs, expanded


# =============================================================================
# Main entry point
# =============================================================================

def prepare_gene_gene_splits(gene_to_idx, train_data, val_data, test_data,
                              test_size=0.1, val_size=0.1, random_state=42,
                              string_path=STRING_PATH,
                              mapping_path=MAPPING_PATH,
                              inductive=False):
    """
    Full pipeline: load STRING → expand gene vocabulary → split.

    Transductive mode (inductive=False):
        Stratified split on the STRING pairs, independent of isoform split.

    Inductive mode (inductive=True):
        ALL STRING gene-gene edges go to training. STRING is an external
        database — always available regardless of which isoforms are held out.
        Val/test triples are empty lists.

    Returns:
        train_triples, val_triples, test_triples,
        expanded_gene_to_idx,
        neg_pos_ratio
    """
    print("\n" + "=" * 70)
    print("GENE-GENE INTERACTION DATA (STRING)")
    print("=" * 70)

    pairs, expanded_gene_to_idx = load_string_pairs(
        gene_to_idx, string_path, mapping_path)

    if inductive:
        print(f"\nInductive mode: all {len(pairs):,} gene-gene pairs → training")
        train_pairs = pairs
        val_pairs   = []
        test_pairs  = []

        n_pos = sum(1 for _, _, l in train_pairs if l == 1)
        n_neg = len(train_pairs) - n_pos
        print(f"  Train: {len(train_pairs):,} pairs  ({n_pos:,} pos  {n_neg:,} neg  "
              f"ratio 1:{n_neg / max(n_pos, 1):.1f})")

    else:
        labels = [l for _, _, l in pairs]

        print("\nSplitting gene-gene interaction pairs (stratified by interact label)...")
        train_val_pairs, test_pairs = train_test_split(
            pairs, test_size=test_size, stratify=labels, random_state=random_state)

        labels_tv = [l for _, _, l in train_val_pairs]
        train_pairs, val_pairs = train_test_split(
            train_val_pairs,
            test_size=val_size / (1 - test_size),
            stratify=labels_tv,
            random_state=random_state,
        )

        for name, split in [('Train', train_pairs), ('Val', val_pairs), ('Test', test_pairs)]:
            n_pos = sum(1 for _, _, l in split if l == 1)
            n_neg = len(split) - n_pos
            print(f"  {name:<6}: {len(split):,} pairs  ({n_pos:,} pos  {n_neg:,} neg  ratio 1:{n_neg/max(n_pos,1):.1f})")

    n_pos_train = sum(1 for _, _, l in train_pairs if l == 1)
    n_neg_train = len(train_pairs) - n_pos_train
    neg_pos_ratio = n_neg_train / max(n_pos_train, 1)
    print(f"  Gene-gene neg/pos ratio (train, used as pos_weight): {neg_pos_ratio:.1f}")

    return train_pairs, val_pairs, test_pairs, expanded_gene_to_idx, neg_pos_ratio


# =============================================================================
# Dataset
# =============================================================================

class GeneGeneDataset(Dataset):
    """Dataset of gene-gene interaction triples (STRING-based)."""

    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        gene_a, gene_b, label = self.triples[idx]
        return (
            torch.tensor(gene_a,  dtype=torch.long),
            torch.tensor(gene_b,  dtype=torch.long),
            torch.tensor(label,   dtype=torch.float32),
        )