"""Evaluate trained gene embeddings on the gene–gene network derived from the
iso-iso dataset (RQ3).

A gene pair (g_a, g_b) is positive iff at least one isoform-pair between them
in the iso-iso CSV has interact=1, negative otherwise. The model is expected
to be trained with `lambda_gene_gene: 0` (no STRING supervision), so the gene
embeddings are shaped only by iso-iso and gene-iso losses.

Two evaluations are reported:

  1. **Gene-embedding distance** — score = −||u_a − u_b||. Asks: how well do
     the per-gene embeddings, alone, recover gene affinity?

  2. **Iso-iso max-aggregation** — score = max over isoform-pairs (a, b) of
     sigmoid(forward_iso_iso(a, b)), aggregated to gene level. Asks: how do
     isoform-level interactions aggregate into gene-level affinity? Uses the
     same max construction as the gene-pair labels themselves.

The gap between the two is the cost of compressing each gene's isoforms into
a single embedding.

Usage:
    python scripts/evaluate_gene_embeddings.py \
        --checkpoint models/IND_MM_<timestamp>/model.pt
    # → writes to models/IND_MM_<timestamp>/rq3/ by default
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader

from src.data_scripts.gene_gene_eval import build_gene_gene_eval_pairs
from src.data_scripts.gene_pairs import GeneGeneDataset
from src.training.evaluate import (
    _compute_and_plot, load_model,
    plot_confusion_matrix, plot_pr_curves, plot_roc_curves,
)


def pick_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def evaluate_iso_iso_aggregated(model, csv_path, protein_to_idx, device, save_dir,
                                batch_size=4096):
    """Aggregate iso-iso predictions to the gene level (max over iso-pairs).

    For each (gene_1, gene_2) row group in the iso-iso CSV:
      * label = max(interact)  — same construction as gene_gene_eval labels
      * score = max(sigmoid(forward_iso_iso(ensp_1, ensp_2)))
    AUC/AP are computed on these per-gene-pair aggregates. Score is a real
    probability, so the standard 0.5 threshold for the confusion matrix
    is meaningful here.
    """
    df = pd.read_csv(csv_path)
    idx_a = df['ensp_1'].map(protein_to_idx)
    idx_b = df['ensp_2'].map(protein_to_idx)
    mask = idx_a.notna() & idx_b.notna()
    n_dropped = (~mask).sum()
    df = df[mask].copy()
    df['idx_a'] = idx_a[mask].astype(int).values
    df['idx_b'] = idx_b[mask].astype(int).values
    print(f"  Iso-iso rows scored: {len(df):,}  (dropped {n_dropped:,} unmappable)")

    model.eval()
    preds = np.empty(len(df), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            chunk = df.iloc[start:start + batch_size]
            a = torch.tensor(chunk['idx_a'].values, dtype=torch.long, device=device)
            b = torch.tensor(chunk['idx_b'].values, dtype=torch.long, device=device)
            preds[start:start + len(chunk)] = (
                torch.sigmoid(model.forward_iso_iso(a, b)).cpu().numpy())
    df['pred'] = preds

    agg = df.groupby(['gene_1', 'gene_2']).agg(
        label=('interact', 'max'),
        score=('pred', 'max'),
    ).reset_index()
    print(f"  Gene-pair aggregates: {len(agg):,}  "
          f"({int(agg['label'].sum()):,} pos, "
          f"{int((1 - agg['label']).sum()):,} neg)")

    auc, ap, _, _ = _compute_and_plot(
        agg['score'].values, agg['label'].values, save_dir,
        title_prefix="Gene-Gene (iso-iso max-agg)",
        fname_suffix="_gg_isoagg",
        confusion_title="Confusion matrix (Gene-Gene iso-iso max-agg)")
    return auc, ap


def evaluate_distance_only(model, loader, device, save_dir):
    """Score gene-gene pairs by −||u_a − u_b||.

    The score is an uncalibrated negative distance, not a probability, so:
      * AUC / AP are computed on the raw score (rank-based — calibration-free).
      * The confusion matrix is reported at the median score (data-driven 50/50
        cut). A 0.5 threshold would be meaningless for an uncalibrated score.
      * The histogram shows the raw embedding-distance distribution per class,
        which is the actually informative view.
    """
    model.eval()
    distances, labels = [], []
    with torch.no_grad():
        for g1, g2, y in loader:
            u1 = model.gene_embeddings(g1.to(device))
            u2 = model.gene_embeddings(g2.to(device))
            distances.extend(torch.norm(u1 - u2, p=2, dim=1).cpu().numpy())
            labels.extend(y.numpy())
    distances = np.asarray(distances)
    labels    = np.asarray(labels)
    scores    = -distances  # higher = more likely positive

    auc = roc_auc_score(labels, scores)
    ap  = average_precision_score(labels, scores)

    threshold = float(np.median(scores))
    preds = scores > threshold
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel().tolist()
    total = tn + fp + fn + tp

    print(f"\nEvaluation Results: Gene-Gene (distance score)")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  Avg Prec:    {ap:.4f}")
    print(f"  (confusion matrix at median-distance threshold = "
          f"{-threshold:.4f})")
    print(f"  Accuracy:    {(tp + tn) / total:.4f}")
    print(f"  Recall:      {tp / max(tp + fn, 1):.4f}")
    print(f"  Precision:   {tp / max(tp + fp, 1):.4f}")
    print(f"  Specificity: {tn / max(tn + fp, 1):.4f}\n")

    fpr, tpr, _ = roc_curve(labels, scores)
    plot_roc_curves([(fpr, tpr, f'AUC = {auc:.4f}')],
                    save_path=f"{save_dir}/roc_curve_gg.png",
                    title="Gene-Gene ROC Curve")
    precisions, recalls, _ = precision_recall_curve(labels, scores)
    plot_pr_curves([(recalls, precisions, f'AP = {ap:.4f}')],
                   save_path=f"{save_dir}/precision_recall_gg.png",
                   title="Gene-Gene Precision-Recall Curve")
    plot_confusion_matrix(
        tn, fp, fn, tp, save_dir=save_dir,
        title="Confusion matrix (Gene-Gene, median threshold)")

    plt.figure(figsize=(8, 6))
    bins = np.linspace(distances.min(), distances.max(), 40)
    plt.hist(distances[labels == 0], bins, label='Negative (0)',
             alpha=0.6, color='blue', density=True)
    plt.hist(distances[labels == 1], bins, label='Positive (1)',
             alpha=0.6, color='red', density=True)
    plt.axvline(-threshold, color='k', linestyle='--',
                label=f'Median threshold = {-threshold:.3f}')
    plt.xlabel('Gene-embedding distance  ||u_a − u_b||')
    plt.ylabel('Density')
    plt.title('Gene-Gene embedding distance distribution by class')
    plt.legend(); plt.grid(True)
    plt.savefig(f"{save_dir}/dist_hist_gg.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved figures to {save_dir}: roc_curve_gg, precision_recall_gg, "
          f"dist_hist_gg, Confusion_matrix_(Gene-Gene,_median_threshold)")
    return auc, ap


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="RQ3 evaluation: gene embeddings on derived gene–gene pairs.")
    parser.add_argument('--checkpoint', required=True,
                        help='Path to a multimodal model checkpoint (model.pt).')
    parser.add_argument('--output', default=None,
                        help='Output directory for figures. '
                             'Defaults to <checkpoint_dir>/rq3.')
    parser.add_argument('--csv', default='data/results_PHYSICAL_Prob_Model_16_02_26.csv',
                        help='Iso-iso CSV to derive gene–gene ground truth from.')
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()

    device = pick_device()
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.checkpoint), 'rq3')
    os.makedirs(args.output, exist_ok=True)
    for stale in glob.glob(os.path.join(args.output, '*.png')):
        os.remove(stale)

    print(f"\n--- RQ3: Gene-embedding link prediction ---")
    print(f"Device: {device}")
    print(f"Output: {args.output}")

    print("\nStep 1: Loading checkpoint...")
    model, protein_to_idx, ckpt = load_model(args.checkpoint, device=device)
    if ckpt.get('model_type') != 'multimodal':
        raise ValueError(f"Expected a multimodal checkpoint, got "
                         f"{ckpt.get('model_type')!r}")
    gene_to_idx = ckpt['gene_to_idx']
    lam_gg = ckpt.get('lambda_gene_gene', None)
    print(f"  λ_gene_gene during training: {lam_gg}  "
          f"({'no STRING supervision' if lam_gg == 0 else 'STRING was used'})")

    print("\nStep 2: Building gene–gene eval set from iso-iso CSV...")
    pairs, _, _ = build_gene_gene_eval_pairs(args.csv, gene_to_idx)
    loader = DataLoader(GeneGeneDataset(pairs),
                        batch_size=args.batch_size, shuffle=False)

    print("\nStep 3: Evaluating gene-embedding distance...")
    auc_emb, ap_emb = evaluate_distance_only(model, loader, device, args.output)

    print("\nStep 4: Evaluating iso-iso predictions aggregated to gene level...")
    auc_iso, ap_iso = evaluate_iso_iso_aggregated(
        model, args.csv, protein_to_idx, device, args.output)

    print(f"\n--- RQ3 results ---")
    print(f"  Gene-embedding distance   AUC: {auc_emb:.4f}   AP: {ap_emb:.4f}")
    print(f"  Iso-iso max-aggregation   AUC: {auc_iso:.4f}   AP: {ap_iso:.4f}")
    print(f"  Figures saved to: {args.output}")


if __name__ == "__main__":
    main()
