"""
Quick smoke-test for complex_dataset.py against the real data.

Run from the project root:
    python verify_complex_dataset.py
"""
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ── 1. Import ─────────────────────────────────────────────────────────────────
print("1. Importing complex_dataset...")
from src.data_scripts.gene_pairs import (
    prepare_gene_gene_splits,
    GeneGeneDataset,
)
print("   OK")

# ── 2. Build gene_to_idx from the real dataset (same as train.py does) ────────
print("\n2. Loading interaction data and building gene_to_idx...")
from src.data_scripts.isoform_pairs import load_and_prepare_data
from src.data_scripts.gene_isoform_pairs import prepare_gene_isoform_splits

DATA_PATH = "data/results_PHYSICAL_Prob_Model_16_02_26.csv"
_, train_data, val_data, test_data, protein_to_idx, num_proteins, _ = \
    load_and_prepare_data(DATA_PATH, test_size=0.1, val_size=0.1, random_state=42)

df_full = pd.read_csv(DATA_PATH)
gene_to_idx, _, _, _, gene_iso_ratio = prepare_gene_isoform_splits(
    df_full, protein_to_idx,
    train_data, val_data, test_data,
    neg_ratio=5, random_state=42,
)
print(f"   gene-isoform neg/pos ratio (auto): {gene_iso_ratio:.1f}")
print(f"   gene_to_idx: {len(gene_to_idx):,} genes")

# ── 3. Run prepare_complex_splits ─────────────────────────────────────────────
print("\n3. Running prepare_gene_gene_splits...")
train_t, val_t, test_t, expanded_gene_to_idx, gene_gene_ratio = prepare_gene_gene_splits(
    gene_to_idx, train_data, val_data, test_data,
)
print(f"   gene-gene neg/pos ratio (auto): {gene_gene_ratio:.1f}")
print(f"   gene vocabulary expanded: {len(gene_to_idx):,} → {len(expanded_gene_to_idx):,}")

# ── 4. Verify triple format ───────────────────────────────────────────────────
print("\n4. Verifying triple format (gene_idx_a, gene_idx_b, label)...")
for name, triples in [("train", train_t), ("val", val_t), ("test", test_t)]:
    assert len(triples) > 0, f"{name} triples is empty!"
    a, b, l = triples[0]
    assert isinstance(a, int), f"gene_idx_a is not int: {type(a)}"
    assert isinstance(b, int), f"gene_idx_b is not int: {type(b)}"
    assert l in (0, 1),        f"label not 0/1: {l}"
    assert a <= b,             f"canonical ordering violated: {a} > {b}"
    assert 0 <= a < len(expanded_gene_to_idx), f"gene_idx_a out of range: {a}"
    assert 0 <= b < len(expanded_gene_to_idx), f"gene_idx_b out of range: {b}"
    print(f"   {name}: {len(triples):,} triples — first={triples[0]}  OK")

# ── 5. Verify Dataset + DataLoader output shapes ──────────────────────────────
print("\n5. Verifying GeneGeneDataset + DataLoader batch shapes...")
loader = DataLoader(GeneGeneDataset(train_t), batch_size=512, shuffle=True)
ga, gb, labels = next(iter(loader))
print(f"   ga:     shape={tuple(ga.shape)}  dtype={ga.dtype}")
print(f"   gb:     shape={tuple(gb.shape)}  dtype={gb.dtype}")
print(f"   labels: shape={tuple(labels.shape)}  dtype={labels.dtype}")
assert ga.dtype     == torch.long,    f"Expected long, got {ga.dtype}"
assert gb.dtype     == torch.long,    f"Expected long, got {gb.dtype}"
assert labels.dtype == torch.float32, f"Expected float32, got {labels.dtype}"
assert ga.max() < len(expanded_gene_to_idx), f"ga index out of range"
assert gb.max() < len(expanded_gene_to_idx), f"gb index out of range"
print("   OK — matches mmldm.py forward_complex(gene_idx_a, gene_idx_b) signature")

# ── 6. Run a forward pass through the real model ─────────────────────────────
print("\n6. Forward pass through MultimodalLDM.forward_complex...")
from model_classes.mm_ldm import MultimodalLDM

model = MultimodalLDM(
    num_proteins=num_proteins,
    num_genes=len(expanded_gene_to_idx),
    latent_dim=32,
)
model.eval()
with torch.no_grad():
    logits = model.forward_complex(ga, gb)
print(f"   logits: shape={tuple(logits.shape)}  dtype={logits.dtype}")
assert logits.shape == (len(ga),), f"Expected ({len(ga)},), got {logits.shape}"
print("   OK")

# ── 7. Check pos_weight note ──────────────────────────────────────────────────
print(f"\n7. Auto pos_weight summary:")
print(f"   isoform-isoform: computed in load_and_prepare_data (global dataset ratio)")
print(f"   gene-isoform:    {gene_iso_ratio:.1f}  (from sampled train triples)")
print(f"   gene-gene:       {gene_gene_ratio:.1f}  (from full STRING dataset)")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)
