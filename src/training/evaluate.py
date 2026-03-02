import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

from src.data_scripts.dataset import ProteinInteractionDataset, load_and_prepare_data
from src.model_classes.latent_distance_model import LatentDistanceModel, BaselineLDM


def load_trained_model(model_path, only_re=False, device='cpu'):
    """
    Load a trained model from a checkpoint file.

    Args:
        model_path: path to .pt checkpoint file
        only_re:    if True, load as BaselineLDM (random effects only, no distance term)
        device:     device to load onto

    Returns:
        model, protein_to_idx, checkpoint dict
    """
    checkpoint = torch.load(model_path, map_location=device)

    cls = BaselineLDM if only_re else LatentDistanceModel
    model = cls(
        num_proteins=checkpoint['num_proteins'],
        latent_dim=checkpoint['latent_dim'],
        distance_metric=checkpoint['distance_metric'],
        random_effects=checkpoint['random_effects'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded: {model_path}")
    print(f"  AUC {checkpoint['test_auc']:.4f}  AP {checkpoint['test_ap']:.4f}  F1 {checkpoint['test_f1']:.4f}")
    print(f"  Latent dim: {checkpoint['latent_dim']}  Metric: {checkpoint['distance_metric']}  RE: {checkpoint['random_effects']}")

    return model, checkpoint['protein_to_idx'], checkpoint


def evaluate_model(model, test_loader, device='cpu', save_dir=None):
    """
    Evaluate a model on a dataloader. Prints metrics and plots ROC / PR curves.

    Args:
        model:       trained model
        test_loader: DataLoader over the evaluation set
        device:      device to run inference on
        save_dir:    if provided, saves roc_curve.png and precision_recall.png here

    Returns:
        auc, ap, f1, all_preds, all_labels
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for protein1_idx, protein2_idx, labels in test_loader:
            predictions = model(protein1_idx.to(device), protein2_idx.to(device))
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    auc = roc_auc_score(all_labels, all_preds)
    ap  = average_precision_score(all_labels, all_preds)
    f1  = f1_score(all_labels, np.array(all_preds) > 0.5)
    tn, fp, fn, tp = confusion_matrix(all_labels, np.array(all_preds) > 0.5).ravel().tolist()
    total = tn + fp + fn + tp

    print(f"\nEvaluation Results:")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  Avg Prec:    {ap:.4f}")
    print(f"  F1:          {f1:.4f}")
    print(f"  Accuracy:    {(tp + tn) / total:.4f}")
    print(f"  Recall:      {tp / (tp + fn):.4f}")
    print(f"  Precision:   {tp / (tp + fp):.4f}")
    print(f"  Specificity: {tn / (tn + fp):.4f}")

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
    plt.show()

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
    plt.show()

    return auc, ap, f1, all_preds, all_labels