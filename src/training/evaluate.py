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

def evaluate_model(model, test_loader, device='cpu', save_dir=None):
    """
    Evaluate a model on a dataloader. Prints metrics and plots ROC / PR curves.

    Predictions are raw logits; threshold > 0 is used for binary metrics
    (equivalent to sigmoid probability > 0.5).

    Args:
        model:       trained model
        test_loader: DataLoader over the evaluation set
        device:      device to run inference on
        save_dir:    if provided, saves roc_curve.png and precision_recall.png here

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

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_preds)
    ap  = average_precision_score(all_labels, all_preds)

    # Threshold at 0.5
    preds_bin            = all_preds > 0.5
    tn, fp, fn, tp       = confusion_matrix(all_labels, preds_bin).ravel().tolist()
    total                = tn + fp + fn + tp

    print(f"\nEvaluation Results:")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  Avg Prec:    {ap:.4f}")
    print(f"  Accuracy:    {(tp + tn) / total:.4f}")
    print(f"  Recall:      {tp / (tp + fn):.4f}")
    print(f"  Precision:   {tp / (tp + fp):.4f}")
    print(f"  Specificity: {tn / (tn + fp):.4f}\n")

    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(); plt.grid(True)
    if save_dir:
        plt.savefig(f"{save_dir}/roc_curve.png", dpi=300, bbox_inches='tight')

    # Precision-Recall curve
    precisions, recalls, _ = precision_recall_curve(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label=f'AP = {ap:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(); plt.grid(True)
    if save_dir:
        plt.savefig(f"{save_dir}/precision_recall.png", dpi=300, bbox_inches='tight')

    # Distribution of prediction probs (model output)
    neg_preds = all_preds[all_labels == 0]
    pos_preds = all_preds[all_labels == 1]
    bins = np.linspace(0,1,21)
    plt.figure(figsize=(8, 6))
    plt.hist(neg_preds, bins, label=f'Negative (0)', alpha=0.6, color="blue", density=True)
    plt.hist(pos_preds, bins, label=f'Positive (1)', alpha=0.6, color="red", density=True)
    plt.xlabel('Prediction prob (model output)')
    plt.ylabel('Density')
    plt.title('Prediction probs distribution')
    plt.legend(); plt.grid(True)
    if save_dir:
        plt.savefig(f"{save_dir}/preds_hist.png", dpi=300, bbox_inches='tight')
        print(f"Saved figures to {save_dir}:\n roc_curve.png\n precision_recall.png\n preds_hist.png")

    return auc, ap, all_preds, all_labels

def evaluate_inductive_model(model, test_data, test_proteins, protein_to_idx, batch_size=512, num_workers=0, device='cpu', save_dir=None):
    """
    Evaluate a model that was trained inductively on a test-set. Prints metrics and plots ROC / PR curves.

    Thresholds predictions from model at 0.5 for binary metrics

    Args:
        model:          trained model
        test_data:      pandas DataFrame of the test interactions
        test_proteins:  frozenset of test proteins
        protein_to_idx: dict mapping (protein ID -> index)
        batch_size:     batch_size used in testloader
        num_workers:    num_workers used in testloader
        device:         device to run inference on
        save_dir:       if provided, saves roc_curve.png and precision_recall.png here

    Returns:
        auc, ap, all_preds, all_labels
    """
    model.eval()
    preds1, labels1, preds2, labels2 = [], [], [], []
    mask = test_data['ensp_1'].isin(test_proteins) & test_data['ensp_2'].isin(test_proteins)
    test_data1 = test_data[mask]
    test_data2 = test_data[~mask]
    dataset1 = ProteinInteractionDataset(test_data1, protein_to_idx)
    dataset2 = ProteinInteractionDataset(test_data2, protein_to_idx)
    test_loader1 = DataLoader(dataset=dataset1, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader2 = DataLoader(dataset=dataset2, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    with torch.no_grad():
        for protein1_idx, protein2_idx, labels in test_loader1:
            predictions = torch.sigmoid(model(protein1_idx.to(device), protein2_idx.to(device)))
            preds1.extend(predictions.cpu().numpy())
            labels1.extend(labels.numpy())
        for protein1_idx, protein2_idx, labels in test_loader2:
            predictions = torch.sigmoid(model(protein1_idx.to(device), protein2_idx.to(device)))
            preds2.extend(predictions.cpu().numpy())
            labels2.extend(labels.numpy())
    preds1 = np.array(preds1)
    labels1 = np.array(labels1)
    preds2 = np.array(preds2)
    labels2 = np.array(labels2)

    auc1 = roc_auc_score(labels1, preds1)
    ap1  = average_precision_score(labels1, preds1)
    auc2 = roc_auc_score(labels2, preds2)
    ap2  = average_precision_score(labels2, preds2)

    # Threshold at 0.5
    preds1_bin            = preds1 > 0.5
    tn1, fp1, fn1, tp1       = confusion_matrix(labels1, preds1_bin).ravel().tolist()
    total1                = tn1 + fp1 + fn1 + tp1

    preds2_bin            = preds2 > 0.5
    tn2, fp2, fn2, tp2       = confusion_matrix(labels2, preds2_bin).ravel().tolist()
    total2                = tn2 + fp2 + fn2 + tp2

    print(f"\nEvaluation Results (for both interaction classes):\n")

    plot_confusion_matrix(tn1, fp1, fn1, tp1, save_dir=save_dir, title="Confusion matrix (Both-unseen)")

    plot_confusion_matrix(tn2, fp2, fn2, tp2, save_dir=save_dir, title="Confusion matrix (One-unseen)")

    print(f"  \t Class-1 (both unseen) \t Class-2 (one unseen)")
    print(f"  AUC-ROC: \t{auc1:.4f} \t\t\t{auc2:.4f}")
    print(f"  Avg Prec: \t{ap1:.4f} \t\t\t{ap2:.4f}")
    print(f"  Accuracy: \t{(tp1 + tn1) / total1:.4f} \t\t\t{(tp2 + tn2) / total2:.4f}")
    print(f"  Recall: \t{tp1 / max(tp1 + fn1, 1):.4f} \t\t\t{tp2 / max(tp2 + fn2, 1):.4f}")
    print(f"  Precision: \t{tp1 / max(tp1 + fp1, 1):.4f} \t\t\t{tp2 / max(tp2 + fp2, 1):.4f}")
    print(f"  Specificity: \t{tn1 / max(tn1 + fp1, 1):.4f} \t\t\t{tn2 / max(tn2 + fp2, 1):.4f}\n")

    # ROC curve
    plt.figure(figsize=(8, 6))
    fpr1, tpr1, _ = roc_curve(labels1, preds1)
    fpr2, tpr2, _ = roc_curve(labels2, preds2)
    plt.plot(fpr1, tpr1, color="red", label=f'AUC = {auc1:.4f} (both unseen)')
    plt.plot(fpr2, tpr2, color="blue", label=f'AUC = {auc2:.4f} (one unseen)')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(); plt.grid(True)
    if save_dir:
        plt.savefig(f"{save_dir}/roc_curves_separated.png", dpi=300, bbox_inches='tight')

    # Precision-Recall curve
    plt.figure(figsize=(8, 6))
    precisions1, recalls1, _ = precision_recall_curve(labels1, preds1)
    precisions2, recalls2, _ = precision_recall_curve(labels2, preds2)
    plt.plot(recalls1, precisions1, color="red", label=f'AP = {ap1:.4f} (both unseen)')
    plt.plot(recalls2, precisions2, color="blue", label=f'AP = {ap2:.4f} (one unseen)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(); plt.grid(True)
    if save_dir:
        plt.savefig(f"{save_dir}/precision_recall_separated.png", dpi=300, bbox_inches='tight')
        print(f"Saved figures to {save_dir}:\n roc_curves_separated.png\n precision_recall_separated.png\n Confusion_matrix_(Both-unseen).png\n Confusion_matrix_(One-unseen).png")

    return auc1, ap1, auc2, ap2

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
        plt.savefig(f"{save_dir}/{"_".join(title.split(" "))}.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    model_dir = "models/IND_MM_20260423_123332"
    model_path = model_dir + "/model.pt"
    model, _, _ = load_model(model_path=model_path)

    from src.data_scripts.isoform_pairs import load_and_prepare_data, load_and_prepare_data_inductive
    csv_path = "data/results_PHYSICAL_Prob_Model_16_02_26.csv"
    _, _, _, test_data, protein_to_idx, _, _, _, _, test_proteins = load_and_prepare_data_inductive(csv_file=csv_path)
    test_loader = DataLoader(ProteinInteractionDataset(test_data, protein_to_idx), batch_size=512, num_workers=4, shuffle=False)

    auc, ap, all_preds, all_labels = evaluate_model(model, test_loader, save_dir=model_dir)
    auc1, ap1, auc2, ap2 = evaluate_inductive_model(model, test_data, test_proteins, protein_to_idx, batch_size=512, num_workers=4, save_dir=model_dir)