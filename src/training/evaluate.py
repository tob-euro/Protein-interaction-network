import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader

from src.data_scripts.isoform_pairs import ProteinInteractionDataset
from src.model_classes.ldm import LatentDistanceModel
from src.model_classes.mm_ldm import MultimodalLDM


def load_model(model_path, device='cpu', only_re=False):
    """Load a trained LDM or MultimodalLDM checkpoint.

    Args:
        model_path: path to the .pt checkpoint file.
        device: torch device to map tensors onto.
        only_re: (LDM only) load weights into a BaselineLDM (random effects only).

    Returns:
        model, protein_to_idx, checkpoint dict
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_type = checkpoint.get('model_type', 'ldm')

    if model_type == 'multimodal':
        esmc_features = checkpoint['model_state_dict'].get('esmc_features', None)
        model = MultimodalLDM(
            num_proteins  = checkpoint['num_proteins'],
            num_genes     = checkpoint['num_genes'],
            latent_dim    = checkpoint['latent_dim'],
            esmc_features = esmc_features,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded MultimodalLDM: {model_path}")
        print(f"  AUC {checkpoint['test_auc']:.4f}  AP {checkpoint['test_ap']:.4f}")
        print(f"  latent={checkpoint['latent_dim']}  "
              f"isoforms={checkpoint['num_proteins']}  genes={checkpoint['num_genes']}")
        print(f"  λ_iso_iso={checkpoint.get('lambda_iso_iso')}  "
              f"λ_gene_iso={checkpoint.get('lambda_gene_iso')}  "
              f"neg_ratio={checkpoint.get('neg_ratio')}")
    else:
        cls = BaselineLDM if only_re else LatentDistanceModel
        model = cls(
            num_proteins=checkpoint['num_proteins'],
            latent_dim=checkpoint['latent_dim'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded {cls.__name__}: {model_path}")
        print(f"  AUC {checkpoint['test_auc']:.4f}  AP {checkpoint['test_ap']:.4f}  "
              f"latent={checkpoint['latent_dim']}")

    model = model.to(device).eval()
    return model, checkpoint['protein_to_idx'], checkpoint


def _plot_curves(curves, xlabel, ylabel, title, save_path=None, diagonal=False):
    plt.figure(figsize=(8, 6))
    for c in curves:
        kw = {'label': c[2]}
        if len(c) > 3:
            kw['color'] = c[3]
        plt.plot(c[0], c[1], **kw)
    if diagonal:
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.legend(); plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_roc_curves(curves, save_path=None, title="ROC Curve"):
    """Plot one or more ROC curves on a single figure.

    Args:
        curves: list of (fpr, tpr, label[, color]) tuples.
        save_path: optional output path for the figure.
        title: plot title.
    """
    _plot_curves(curves, 'False Positive Rate', 'True Positive Rate',
                 title, save_path, diagonal=True)


def plot_pr_curves(curves, save_path=None, title="Precision-Recall Curve"):
    """Plot one or more Precision–Recall curves on a single figure.

    Args:
        curves: list of (recall, precision, label[, color]) tuples.
        save_path: optional output path for the figure.
        title: plot title.
    """
    _plot_curves(curves, 'Recall', 'Precision', title, save_path)


def plot_confusion_matrix(tn, fp, fn, tp, save_dir=None, title="Confusion Matrix"):
    total  = tn + fp + fn + tp
    matrix = np.array([[tn, fp], [fn, tp]])
    labels = [["TN", "FP"], ["FN", "TP"]]

    acc  = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Neg", "Pred Pos"])
    ax.set_yticklabels(["Actual Neg", "Actual Pos"])
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()

    for i in range(2):
        for j in range(2):
            v = matrix[i, j]
            ax.text(j, i, f"{labels[i][j]}\n{v} ({v / total:.0%})",
                    ha="center", va="center", fontsize=12,
                    color="white" if v > matrix.max() * 0.6 else "black")

    ax.set_title(title, pad=20, fontweight="bold")
    fig.text(0.5, 0.02,
             f"Accuracy: {acc:.1%}  |  Precision: {prec:.1%}  |  Recall: {rec:.1%}",
             ha="center", fontsize=10)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if save_dir:
        fname = "_".join(title.split())
        plt.savefig(f"{save_dir}/{fname}.png", dpi=300, bbox_inches='tight')


def _compute_and_plot(all_preds, all_labels, save_dir,
                      title_prefix="", fname_suffix="",
                      confusion_title=None, skip_plots=False):
    """Compute metrics, print summary, optionally save ROC/PR/histogram."""
    all_preds  = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)

    auc = roc_auc_score(all_labels, all_preds)
    ap  = average_precision_score(all_labels, all_preds)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds > 0.5).ravel().tolist()
    total = tn + fp + fn + tp

    sep = " " if title_prefix else ""
    print(f"\nEvaluation Results: {title_prefix}")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  Avg Prec:    {ap:.4f}")
    print(f"  Accuracy:    {(tp + tn) / total:.4f}")
    print(f"  Recall:      {tp / (tp + fn):.4f}")
    print(f"  Precision:   {tp / (tp + fp):.4f}")
    print(f"  Specificity: {tn / (tn + fp):.4f}\n")

    if confusion_title:
        plot_confusion_matrix(tn, fp, fn, tp, save_dir=save_dir, title=confusion_title)

    if not skip_plots:
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        plot_roc_curves(
            [(fpr, tpr, f'AUC = {auc:.4f}')],
            save_path=f"{save_dir}/roc_curve{fname_suffix}.png" if save_dir else None,
            title=f"{title_prefix}{sep}ROC Curve")

        precisions, recalls, _ = precision_recall_curve(all_labels, all_preds)
        plot_pr_curves(
            [(recalls, precisions, f'AP = {ap:.4f}')],
            save_path=f"{save_dir}/precision_recall{fname_suffix}.png" if save_dir else None,
            title=f"{title_prefix}{sep}Precision-Recall Curve")

        bins = np.linspace(0, 1, 21)
        plt.figure(figsize=(8, 6))
        plt.hist(all_preds[all_labels == 0], bins, label='Negative (0)',
                 alpha=0.6, color='blue', density=True)
        plt.hist(all_preds[all_labels == 1], bins, label='Positive (1)',
                 alpha=0.6, color='red',  density=True)
        plt.xlabel('Prediction prob (model output)'); plt.ylabel('Density')
        plt.title(f"{title_prefix}{sep}Prediction probs distribution")
        plt.legend(); plt.grid(True)
        if save_dir:
            plt.savefig(f"{save_dir}/preds_hist{fname_suffix}.png",
                        dpi=300, bbox_inches='tight')
            print(f"Saved figures to {save_dir}: roc_curve{fname_suffix}, "
                  f"precision_recall{fname_suffix}, preds_hist{fname_suffix}")

    return auc, ap, all_preds, all_labels


def evaluate_model(model, test_loader, device='cpu', save_dir=None):
    """Evaluate iso–iso predictions on a dataloader; print metrics and save plots.

    Args:
        model: a trained LDM-family model with a forward(p1_idx, p2_idx) → logits.
        test_loader: DataLoader yielding (p1_idx, p2_idx, label) batches.
        device: torch device for inference.
        save_dir: if set, save ROC/PR/histogram figures here.

    Returns:
        auc, ap, all_preds, all_labels
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for p1, p2, y in test_loader:
            preds = torch.sigmoid(model(p1.to(device), p2.to(device)))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
    return _compute_and_plot(all_preds, all_labels, save_dir)


def evaluate_inductive_model_separately(model, test_data, test_proteins, protein_to_idx,
                                        batch_size=512, num_workers=0,
                                        device='cpu', save_dir=None):
    """Evaluate test pairs split into both-unseen vs one-unseen subsets.

    Args:
        model: a trained inductive LDM-family model.
        test_data: test-split DataFrame.
        test_proteins: set of held-out test isoforms.
        protein_to_idx: isoform → index mapping.
        batch_size, num_workers: DataLoader settings.
        device: torch device for inference.
        save_dir: if set, save ROC/PR/confusion-matrix figures here.

    Returns:
        auc1, ap1 (both-unseen), auc2, ap2 (one-unseen)
    """
    model.eval()

    def _infer(split_df):
        loader = DataLoader(ProteinInteractionDataset(split_df, protein_to_idx),
                            batch_size=batch_size, shuffle=False, num_workers=num_workers)
        preds, labels = [], []
        with torch.no_grad():
            for p1, p2, y in loader:
                preds.extend(torch.sigmoid(model(p1.to(device), p2.to(device))).cpu().numpy())
                labels.extend(y.numpy())
        return np.array(preds), np.array(labels)

    mask = test_data['ensp_1'].isin(test_proteins) & test_data['ensp_2'].isin(test_proteins)
    preds1, labels1 = _infer(test_data[mask])
    preds2, labels2 = _infer(test_data[~mask])

    print("\nEvaluation Results (per pair class):\n")
    auc1, ap1, _, _ = _compute_and_plot(
        preds1, labels1, save_dir,
        title_prefix="Both-unseen", confusion_title="Confusion matrix (Both-unseen)",
        skip_plots=True)
    auc2, ap2, _, _ = _compute_and_plot(
        preds2, labels2, save_dir,
        title_prefix="One-unseen", confusion_title="Confusion matrix (One-unseen)",
        skip_plots=True)

    fpr1, tpr1, _ = roc_curve(labels1, preds1)
    fpr2, tpr2, _ = roc_curve(labels2, preds2)
    plot_roc_curves(
        [(fpr1, tpr1, f'AUC = {auc1:.4f} (both unseen)', 'red'),
         (fpr2, tpr2, f'AUC = {auc2:.4f} (one unseen)',  'blue')],
        save_path=f"{save_dir}/roc_curves_separated.png" if save_dir else None,
    )
    p1, r1, _ = precision_recall_curve(labels1, preds1)
    p2, r2, _ = precision_recall_curve(labels2, preds2)
    plot_pr_curves(
        [(r1, p1, f'AP = {ap1:.4f} (both unseen)', 'red'),
         (r2, p2, f'AP = {ap2:.4f} (one unseen)',  'blue')],
        save_path=f"{save_dir}/precision_recall_separated.png" if save_dir else None,
    )

    if save_dir:
        print(f"Saved: {save_dir}/roc_curves_separated.png, "
              f"precision_recall_separated.png, "
              f"Confusion_matrix_(Both-unseen).png, Confusion_matrix_(One-unseen).png")
    return auc1, ap1, auc2, ap2


def evaluate_gene_gene(model, gene_gene_loader, device, save_dir):
    """Evaluate gene–gene predictions on a STRING test loader.

    Args:
        model: a trained MultimodalLDM (uses model.forward_gene_gene).
        gene_gene_loader: DataLoader yielding (g1_idx, g2_idx, label) batches.
        device: torch device for inference.
        save_dir: if set, save ROC/PR/histogram figures here.

    Returns:
        auc, ap, all_preds, all_labels
    """
    model.eval()
    all_preds, all_labels = [], []
    print("\nEvaluating gene–gene interactions\n")
    with torch.no_grad():
        for g1, g2, y in gene_gene_loader:
            preds = torch.sigmoid(model.forward_gene_gene(g1.to(device), g2.to(device)))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
    return _compute_and_plot(all_preds, all_labels, save_dir,
                             title_prefix="Gene-Gene", fname_suffix="_gg",
                             confusion_title="Confusion matrix (Gene-Gene)")
