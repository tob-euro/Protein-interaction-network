import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.training.evaluate import load_model
from src.visualizations.pca import visualize_latent_space_pca, visualize_pca_variance


def _save(fig_path):
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize a trained LDM or MultimodalLDM.')
    parser.add_argument('model_dir', help='Path to model directory (contains model.pt)')
    parser.add_argument('--config', default='config/config.yaml')
    args = parser.parse_args()

    model_pt = os.path.join(args.model_dir, "model.pt")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    save_dir = os.path.join(args.model_dir, 'visualizations')
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n--- Visualizations ---\nModel: {args.model_dir}\nSaving to: {save_dir}\n")

    model, protein_to_idx, _ = load_model(model_pt, device='cpu')
    idx_to_protein = {idx: p for p, idx in protein_to_idx.items()}
    all_data = pd.read_csv(cfg['data']['iso_path'])
    v = cfg['visualization']

    print("--- PCA ---")
    visualize_latent_space_pca(model, protein_to_idx, data=all_data,
                               n_components=2, idx_to_protein=idx_to_protein)
    _save(f"{save_dir}/latent_pca_2d.png")

    visualize_latent_space_pca(model, protein_to_idx, data=all_data,
                               n_components=3, idx_to_protein=idx_to_protein)
    _save(f"{save_dir}/latent_pca_3d.png")

    visualize_pca_variance(model, max_components=v['pca_max_components'])
    _save(f"{save_dir}/pca_variance.png")

    print(f"\nDone. Figures in: {save_dir}\n")


if __name__ == "__main__":
    main()
