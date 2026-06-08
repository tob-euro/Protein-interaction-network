import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def load_string_pairs(gene_to_idx, string_path, mapping_path):
    """Load STRING pairs, expand gene_to_idx with STRING-only genes, map to indices.

    Args:
        gene_to_idx: existing gene → index mapping.
        string_path: STRING physical-interactions CSV (protein1, protein2, interact).
        mapping_path: ENSP→ENSG mapping CSV.

    Returns:
        pairs: list of (gene_idx_a, gene_idx_b, label) with a ≤ b.
        expanded_gene_to_idx: gene_to_idx plus any STRING-only ENSG IDs.
    """
    print(f"  Loading STRING pairs from {string_path} ...")
    string_df = pd.read_csv(string_path)
    print(f"  Loading ENSP→ENSG mapping from {mapping_path} ...")
    mapping_df   = pd.read_csv(mapping_path)
    ensp_to_ensg = dict(zip(mapping_df['ensp_id'], mapping_df['ensg_id']))

    ensg1 = string_df['protein1'].map(ensp_to_ensg)
    ensg2 = string_df['protein2'].map(ensp_to_ensg)
    mask  = ensg1.notna() & ensg2.notna()
    print(f"  ENSP→ENSG mappable: {mask.sum():,} / {len(string_df):,} "
          f"(dropped {(~mask).sum():,} unmappable)")

    ensg1, ensg2 = ensg1[mask].values, ensg2[mask].values
    labels       = string_df['interact'][mask].values

    # Extend the gene vocabulary with STRING-only genes (no isoform-pair coverage).
    expanded = dict(gene_to_idx)
    for ensg in set(ensg1) | set(ensg2):
        expanded.setdefault(ensg, len(expanded))
    n_new = len(expanded) - len(gene_to_idx)
    print(f"  Gene vocabulary: {len(gene_to_idx):,} (isoform) "
          f"+ {n_new:,} (STRING-only) = {len(expanded):,} total")

    pairs = [
        (min(expanded[a], expanded[b]), max(expanded[a], expanded[b]), int(l))
        for a, b, l in zip(ensg1, ensg2, labels)
    ]
    n_pos = sum(1 for _, _, l in pairs if l == 1)
    print(f"  Total pairs: {len(pairs):,}  ({n_pos:,} pos  {len(pairs) - n_pos:,} neg  "
          f"global 1:{(len(pairs) - n_pos) / max(n_pos, 1):.1f})")
    return pairs, expanded


def prepare_gene_gene_splits(gene_to_idx, train_data, val_data, test_data, string_path,
                             mapping_path, test_size=0.2, val_size=0.1, random_state=42,
                             inductive=False):
    """Load STRING, expand gene vocabulary, and split.

    Args:
        gene_to_idx: existing gene → index mapping (extended in the result).
        train_data, val_data, test_data: iso–iso splits (currently unused for
            stratification — STRING is split independently).
        test_size, val_size: stratified split fractions (transductive only).
        random_state: seed for the stratified split.
        string_path, mapping_path: source CSV locations.
        inductive: if True, all STRING edges go to training (external prior,
            available regardless of which isoforms are held out).

    Returns:
        train_pairs, val_pairs, test_pairs, expanded_gene_to_idx, neg_pos_ratio
        (val is empty in inductive mode; test mirrors train so eval still works).
    """
    print("\n--- Gene–gene interaction data (STRING) ---")
    pairs, expanded_gene_to_idx = load_string_pairs(gene_to_idx, string_path, mapping_path)

    if inductive:
        print(f"\nInductive: all {len(pairs):,} gene–gene pairs → training")
        train_pairs = pairs
        val_pairs   = []
        test_pairs  = pairs
        n_pos = sum(1 for _, _, l in train_pairs if l == 1)
        print(f"  Train: {len(train_pairs):,}  ({n_pos:,} pos  "
              f"{len(train_pairs) - n_pos:,} neg  "
              f"1:{(len(train_pairs) - n_pos) / max(n_pos, 1):.1f})")
    else:
        labels = [l for _, _, l in pairs]
        print("\nSplitting gene–gene pairs (stratified)...")
        train_val_pairs, test_pairs = train_test_split(
            pairs, test_size=test_size, stratify=labels, random_state=random_state)
        labels_tv = [l for _, _, l in train_val_pairs]
        train_pairs, val_pairs = train_test_split(
            train_val_pairs, test_size=val_size / (1 - test_size),
            stratify=labels_tv, random_state=random_state)

        for name, split in [('Train', train_pairs), ('Val', val_pairs), ('Test', test_pairs)]:
            n_pos = sum(1 for _, _, l in split if l == 1)
            print(f"  {name:<6}: {len(split):,}  ({n_pos:,} pos  "
                  f"{len(split) - n_pos:,} neg  "
                  f"1:{(len(split) - n_pos) / max(n_pos, 1):.1f})")

    n_pos = sum(1 for _, _, l in train_pairs if l == 1)
    neg_pos_ratio = (len(train_pairs) - n_pos) / max(n_pos, 1)
    print(f"  Gene–gene pos_weight (train neg/pos): {neg_pos_ratio:.1f}")
    return train_pairs, val_pairs, test_pairs, expanded_gene_to_idx, neg_pos_ratio


class GeneGeneDataset(Dataset):
    """Gene–gene STRING interaction triples."""

    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        gene_a, gene_b, label = self.triples[idx]
        return (
            torch.tensor(gene_a, dtype=torch.long),
            torch.tensor(gene_b, dtype=torch.long),
            torch.tensor(label,  dtype=torch.float32),
        )
