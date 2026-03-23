import os
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from src.training.evaluate import load_trained_model, load_trained_mm_model
from src.data_scripts.isoform_pairs import load_and_prepare_data
from src.visualizations.pca import visualize_latent_space_pca, visualize_pca_variance
from src.visualizations.hierarchical_clustering import plot_dendrogram, plot_cluster_sizes, plot_pca_by_cluster

# =============================================================================
MODEL_DIR = "models/MM_dim=32_metric=euclidean_epochs=30_lr=0.001_BS=512_lIso=1.0_lGene=0.5_negR=5"
# =============================================================================

CONFIG = "config/config.yaml"

def detect_model_type(model_dir):
    """Infer model type and checkpoint path from directory contents."""
    mm_pt  = os.path.join(model_dir, 'multimodal_ldm.pt')
    ldm_pt = os.path.join(model_dir, 'latent_distance_model.pt')
    if os.path.exists(mm_pt):
        return 'multimodal', mm_pt
    if os.path.exists(ldm_pt):
        return 'ldm', ldm_pt
    sys.exit(f"No checkpoint found in {model_dir}")


def main():
    model_type, model_pt = detect_model_type(MODEL_DIR)

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    save_dir = os.path.join(MODEL_DIR, 'visualizations')
    os.makedirs(save_dir, exist_ok=True)

    label = 'MULTIMODAL LDM' if model_type == 'multimodal' else 'LDM'
    print(f"\n{'='*70}\nVISUALIZATIONS — {label}\n{'='*70}")
    print(f"Model    : {MODEL_DIR}")
    print(f"Saving to: {save_dir}\n")

    # Load correct model type (CPU — embeddings are numpy anyway)
    if model_type == 'multimodal':
        model, protein_to_idx, _ = load_trained_mm_model(model_pt, device='cpu')
    else:
        model, protein_to_idx, _ = load_trained_model(model_pt, device='cpu')
    model.eval()
    idx_to_protein = {idx: p for p, idx in protein_to_idx.items()}

    # Load data (needed for degree colouring in PCA)
    d = cfg['data']
    _, train_data, val_data, test_data, _, _, _ = load_and_prepare_data(
        d['path'], test_size=d['test_size'], val_size=d['val_size'])
    all_data = pd.concat([train_data, val_data, test_data])

    v = cfg['visualization']

    # =========================================================================
    # PCA
    # =========================================================================
    print("--- PCA ---")

    fig, _, _ = visualize_latent_space_pca(
        model, protein_to_idx, data=all_data,
        n_components=2, idx_to_protein=idx_to_protein)
    plt.savefig(f"{save_dir}/latent_pca_2d.png", dpi=300, bbox_inches='tight')
    plt.close(); print(f"Saved: {save_dir}/latent_pca_2d.png")

    fig, _, _ = visualize_latent_space_pca(
        model, protein_to_idx, data=all_data,
        n_components=3, idx_to_protein=idx_to_protein)
    plt.savefig(f"{save_dir}/latent_pca_3d.png", dpi=300, bbox_inches='tight')
    plt.close(); print(f"Saved: {save_dir}/latent_pca_3d.png")

    fig, _, _ = visualize_pca_variance(model, max_components=v['pca_max_components'])
    plt.savefig(f"{save_dir}/pca_variance.png", dpi=300, bbox_inches='tight')
    plt.close(); print(f"Saved: {save_dir}/pca_variance.png")

    # =========================================================================
    # Hierarchical clustering
    # =========================================================================
    print("\n--- Hierarchical clustering ---")
    n_clusters = v['clustering_n_clusters']

    fig, cluster_labels, _ = plot_dendrogram(
        model, protein_to_idx,
        truncate_level=v['clustering_truncate_level'],
        idx_to_protein=idx_to_protein)
    plt.savefig(f"{save_dir}/hierarchical_dendrogram.png", dpi=300, bbox_inches='tight')
    plt.close(); print(f"Saved: {save_dir}/hierarchical_dendrogram.png")

    fig = plot_cluster_sizes(cluster_labels, n_clusters=n_clusters)
    plt.savefig(f"{save_dir}/cluster_sizes.png", dpi=300, bbox_inches='tight')
    plt.close(); print(f"Saved: {save_dir}/cluster_sizes.png")

    fig, _ = plot_pca_by_cluster(
        model, cluster_labels, n_clusters=n_clusters, idx_to_protein=idx_to_protein)
    plt.savefig(f"{save_dir}/pca_by_cluster.png", dpi=300, bbox_inches='tight')
    plt.close(); print(f"Saved: {save_dir}/pca_by_cluster.png")

    print(f"\n{'='*70}\nDone. All figures saved to: {save_dir}\n{'='*70}\n")


if __name__ == "__main__":
    main()