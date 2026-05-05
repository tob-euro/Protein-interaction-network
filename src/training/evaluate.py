import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

from src.model_classes.ldm import LatentDistanceModel, BaselineLDM
from src.model_classes.mm_ldm import MultimodalLDM
from src.data_scripts.isoform_pairs import ProteinInteractionDataset


def load_model(model_path, device='cpu', only_re=False):
    """
    Load a trained LDM or MultimodalLDM from a checkpoint file.

    Args:
        model_path: path to .pt checkpoint file
        only_re:    (LDM only) if True, load as BaselineLDM (random effects only)
        device:     device to load onto

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
            use_residuals = checkpoint.get('use_residuals', True),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded MultimodalLDM: {model_path}")
        print(f"  AUC {checkpoint['test_auc']:.4f}  AP {checkpoint['test_ap']:.4f}")
        print(f"  Latent dim: {checkpoint['latent_dim']}  "
              f"Isoforms: {checkpoint['num_proteins']}  Genes: {checkpoint['num_genes']}")
        print(f"  λ_iso: {checkpoint['lambda_iso']}  λ_gene: {checkpoint['lambda_gene']}  "
              f"neg_ratio: {checkpoint['neg_ratio']}")
    else:
        cls   = BaselineLDM if only_re else LatentDistanceModel
        model = cls(
            num_proteins = checkpoint['num_proteins'],
            latent_dim   = checkpoint['latent_dim'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded {'BaselineLDM' if only_re else 'LatentDistanceModel'}: {model_path}")
        print(f"  AUC {checkpoint['test_auc']:.4f}  AP {checkpoint['test_ap']:.4f}")
        print(f"  Latent dim: {checkpoint['latent_dim']}")

    model = model.to(device)
    model.eval()
    return model, checkpoint['protein_to_idx'], checkpoint

def plot_roc_curves(curves, save_path=None, title="ROC Curve"):
    """Plot one or more ROC curves on a single figure.

    curves: list of (fpr, tpr, legend_label[, color]) tuples
    """
    plt.figure(figsize=(8, 6))
    for c in curves:
        fpr, tpr, label = c[0], c[1], c[2]
        kw = {'label': label, 'color': c[3]} if len(c) > 3 else {'label': label}
        plt.plot(fpr, tpr, **kw)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(); plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_pr_curves(curves, save_path=None, title="Precision-Recall Curve"):
    """Plot one or more Precision-Recall curves on a single figure.

    curves: list of (recalls, precisions, legend_label[, color]) tuples
    """
    plt.figure(figsize=(8, 6))
    for c in curves:
        recalls, precisions, label = c[0], c[1], c[2]
        kw = {'label': label, 'color': c[3]} if len(c) > 3 else {'label': label}
        plt.plot(recalls, precisions, **kw)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(); plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def _compute_and_plot(all_preds, all_labels, save_dir,
                      title_prefix="", fname_suffix="", confusion_title=None, skip_plots=False):
    """Compute metrics, print a summary, and optionally save ROC / PR / histogram plots."""
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_preds)
    ap  = average_precision_score(all_labels, all_preds)

    preds_bin      = all_preds > 0.5
    tn, fp, fn, tp = confusion_matrix(all_labels, preds_bin).ravel().tolist()
    total          = tn + fp + fn + tp

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
        plot_roc_curves([(fpr, tpr, f'AUC = {auc:.4f}')],
                        save_path=f"{save_dir}/roc_curve{fname_suffix}.png" if save_dir else None,
                        title=f'{title_prefix}{sep}ROC Curve')

        precisions, recalls, _ = precision_recall_curve(all_labels, all_preds)
        plot_pr_curves([(recalls, precisions, f'AP = {ap:.4f}')],
                       save_path=f"{save_dir}/precision_recall{fname_suffix}.png" if save_dir else None,
                       title=f'{title_prefix}{sep}Precision-Recall Curve')

        neg_preds = all_preds[all_labels == 0]
        pos_preds = all_preds[all_labels == 1]
        bins = np.linspace(0, 1, 21)
        plt.figure(figsize=(8, 6))
        plt.hist(neg_preds, bins, label=f'Negative (0)', alpha=0.6, color="blue", density=True)
        plt.hist(pos_preds, bins, label=f'Positive (1)', alpha=0.6, color="red", density=True)
        plt.xlabel('Prediction prob (model output)')
        plt.ylabel('Density')
        plt.title(f'{title_prefix}{sep}Prediction probs distribution')
        plt.legend(); plt.grid(True)
        if save_dir:
            plt.savefig(f"{save_dir}/preds_hist{fname_suffix}.png", dpi=300, bbox_inches='tight')
            print(f"Saved figures to {save_dir}:\n roc_curve{fname_suffix}.png\n"
                  f" precision_recall{fname_suffix}.png\n preds_hist{fname_suffix}.png")

    return auc, ap, all_preds, all_labels


def evaluate_model(model, test_loader, device='cpu', save_dir=None):
    """
    Evaluate a model on a dataloader. Prints metrics and plots ROC / PR curves.

    Returns:
        auc, ap, all_preds, all_labels
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for protein1_idx, protein2_idx, labels in test_loader:
            predictions = torch.sigmoid(model(protein1_idx.to(device), protein2_idx.to(device)))
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    return _compute_and_plot(all_preds, all_labels, save_dir)

def evaluate_inductive_model_separately(model, test_data, test_proteins, protein_to_idx, batch_size=512, num_workers=0, device='cpu', save_dir=None):
    """
    Evaluate a model trained inductively on the test set, stratified by whether
    both or only one endpoint is unseen. Prints per-class metrics and saves
    combined ROC / PR plots plus a full-set prediction histogram.

    Returns:
        auc1, ap1 (both-unseen), auc2, ap2 (one-unseen)
    """
    model.eval()

    def _infer(split_df):
        loader = DataLoader(ProteinInteractionDataset(split_df, protein_to_idx),
                            batch_size=batch_size, shuffle=False, num_workers=num_workers)
        preds, labels = [], []
        with torch.no_grad():
            for p1, p2, lab in loader:
                preds.extend(torch.sigmoid(model(p1.to(device), p2.to(device))).cpu().numpy())
                labels.extend(lab.numpy())
        return np.array(preds), np.array(labels)

    mask            = test_data['ensp_1'].isin(test_proteins) & test_data['ensp_2'].isin(test_proteins)
    preds1, labels1 = _infer(test_data[mask])
    preds2, labels2 = _infer(test_data[~mask])

    print(f"\nEvaluation Results (for both interaction classes):\n")
    auc1, ap1, _, _ = _compute_and_plot(preds1, labels1, save_dir,
                                         title_prefix="Both-unseen",
                                         confusion_title="Confusion matrix (Both-unseen)",
                                         skip_plots=True)
    auc2, ap2, _, _ = _compute_and_plot(preds2, labels2, save_dir,
                                         title_prefix="One-unseen",
                                         confusion_title="Confusion matrix (One-unseen)",
                                         skip_plots=True)

    fpr1, tpr1, _ = roc_curve(labels1, preds1)
    fpr2, tpr2, _ = roc_curve(labels2, preds2)
    plot_roc_curves(
        [(fpr1, tpr1, f'AUC = {auc1:.4f} (both unseen)', 'red'),
         (fpr2, tpr2, f'AUC = {auc2:.4f} (one unseen)',  'blue')],
        save_path=f"{save_dir}/roc_curves_separated.png" if save_dir else None,
    )

    precisions1, recalls1, _ = precision_recall_curve(labels1, preds1)
    precisions2, recalls2, _ = precision_recall_curve(labels2, preds2)
    plot_pr_curves(
        [(recalls1, precisions1, f'AP = {ap1:.4f} (both unseen)', 'red'),
         (recalls2, precisions2, f'AP = {ap2:.4f} (one unseen)',  'blue')],
        save_path=f"{save_dir}/precision_recall_separated.png" if save_dir else None,
    )

    if save_dir:
        print(f"Saved figures to {save_dir}:\n roc_curves_separated.png\n precision_recall_separated.png\n"
              f" Confusion_matrix_(Both-unseen).png\n Confusion_matrix_(One-unseen).png\n preds_hist.png")

    return auc1, ap1, auc2, ap2

def evaluate_gene_gene(model, gene_gene_loader, device, save_dir):
    model.eval()
    all_preds, all_labels = [], []

    print("\nEvaluating gene-gene interactions\n")

    with torch.no_grad():
        for gene1_idx, gene2_idx, labels in gene_gene_loader:
            predictions = torch.sigmoid(model.forward_gene_gene(gene1_idx.to(device), gene2_idx.to(device)))
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    return _compute_and_plot(all_preds, all_labels, save_dir,
                             title_prefix="Gene-Gene", fname_suffix="_gg",
                             confusion_title="Confusion matrix (Gene-Gene)")

def plot_confusion_matrix(tn, fp, fn, tp, save_dir=None, title="Confusion Matrix"):
    total = tn + fp + fn + tp
    matrix = np.array([[tn, fp], [fn, tp]])
    labels = [["TN", "FP"], ["FN", "TP"]]

    acc  = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Neg", "Pred Pos"])
    ax.set_yticklabels(["Actual Neg", "Actual Pos"])
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            ax.text(j, i, f"{labels[i][j]}\n{val} ({val/total:.0%})",
                    ha="center", va="center", fontsize=12,
                    color="white" if val > matrix.max() * 0.6 else "black")

    ax.set_title(title, pad=20, fontweight="bold")
    fig.text(0.5, 0.02,
             f"Accuracy: {acc:.1%}  |  Precision: {prec:.1%}  |  Recall: {rec:.1%}",
             ha="center", fontsize=10)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if save_dir:
        fname = "_".join(title.split(" "))
        plt.savefig(f"{save_dir}/{fname}.png", dpi=300, bbox_inches='tight')

