import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import fastcluster
from scipy.cluster.hierarchy import dendrogram, fcluster
from sklearn.decomposition import PCA

from src.data.dataset import load_and_prepare_data
from training.evaluate import load_trained_model



def plot_dendrogram(model, protein_to_idx, truncate_level=10, idx_to_protein=None):
    """
    Hierarchical clustering dendrogram of protein latent embeddings using Ward linkage.
    Uses fastcluster.linkage_vector which operates directly on the embedding matrix
    without computing a full pairwise distance matrix — scales to all proteins.

    Args:
        model:           Trained LatentDistanceModel
        protein_to_idx:  Protein to index mapping
        truncate_level:  Show only the top N merged clusters (None = full tree)
        idx_to_protein:  Index to protein mapping (for leaf labels when n <= 100)

    Returns:
        fig, cluster_labels (array of cluster assignments per protein), linkage_matrix Z
    """
    embeddings = model.get_embeddings()
    n = embeddings.shape[0]

    print(f"Computing Ward linkage on all {n} proteins via fastcluster.linkage_vector...")
    Z = fastcluster.linkage_vector(embeddings, method='ward')

    n_clusters = 8
    cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
    print(f"Assigned {n_clusters} clusters")

    fig, ax = plt.subplots(figsize=(14, 7))

    leaf_labels = [idx_to_protein.get(i, str(i)) for i in range(n)] if idx_to_protein and n <= 100 else None

    dendrogram(
        Z,
        ax=ax,
        truncate_mode='level' if truncate_level else None,
        p=truncate_level,
        labels=leaf_labels,
        color_threshold=Z[-n_clusters, 2],
        above_threshold_color='lightgrey',
        leaf_rotation=90,
        leaf_font_size=7 if leaf_labels else 0,
    )

    ax.set_title(f'Hierarchical Clustering — Ward Linkage (latent space, {n} proteins)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Protein' if leaf_labels else 'Protein index', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    return fig, cluster_labels, Z


def plot_cluster_sizes(cluster_labels, n_clusters=8):
    """Bar chart of protein counts per cluster."""
    counts = np.bincount(cluster_labels)[1:]   # fcluster labels are 1-indexed

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(1, n_clusters + 1), counts, edgecolor='black', alpha=0.8)
    ax.bar_label(bars)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Number of Proteins', fontsize=12)
    ax.set_title('Cluster Size Distribution (Ward Linkage)', fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    return fig


def plot_pca_by_cluster(model, cluster_labels, n_clusters=8,
                        idx_to_protein=None, label_top_n=0,
                        colormap='tab10'):
    """
    Project latent embeddings to 2D via PCA and color each point by its
    hierarchical cluster assignment.

    Args:
        model:          Trained LatentDistanceModel
        cluster_labels: 1-indexed cluster array returned by fcluster (shape: n_proteins,)
        n_clusters:     Number of clusters (used for legend and colormap range)
        idx_to_protein: Index to protein name mapping — used if label_top_n > 0
        label_top_n:    Annotate the N proteins closest to each cluster centroid
                        (0 = no labels)
        colormap:       A qualitative matplotlib colormap ('tab10', 'Set1', etc.)

    Returns:
        fig, pca_embeddings (n_proteins × 2)
    """
    embeddings = model.get_embeddings()   # numpy (n, latent_dim)
    n = embeddings.shape[0]

    # ── PCA ──────────────────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(embeddings)
    var = pca.explained_variance_ratio_
    print(f"\nPCA variance explained: PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%  "
          f"total={var.sum()*100:.1f}%")

    # ── Colours: one per cluster (fcluster labels are 1-indexed) ─────────────
    cmap = plt.get_cmap(colormap)
    # Map cluster id (1…n_clusters) → colour
    colours = np.array([cmap((lbl - 1) / n_clusters) for lbl in cluster_labels])

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 9))

    for c in range(1, n_clusters + 1):
        mask = cluster_labels == c
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            s=18, color=cmap((c - 1) / n_clusters),
            alpha=0.65, edgecolors='none',
            label=f'Cluster {c}  (n={mask.sum():,})',
            rasterized=True,   # keeps file size small for large n
        )

    # ── Optional: label the protein closest to each cluster centroid ─────────
    if label_top_n > 0 and idx_to_protein is not None:
        for c in range(1, n_clusters + 1):
            mask = np.where(cluster_labels == c)[0]
            if len(mask) == 0:
                continue
            centroid = emb_2d[mask].mean(axis=0)
            dists = np.linalg.norm(emb_2d[mask] - centroid, axis=1)
            closest = mask[np.argsort(dists)[:label_top_n]]
            for idx in closest:
                name = idx_to_protein.get(idx, str(idx))
                ax.annotate(
                    name,
                    (emb_2d[idx, 0], emb_2d[idx, 1]),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=7, alpha=0.85,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.5),
                )

    ax.set_xlabel(f'PC1  ({var[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2  ({var[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title(
        f'Latent Space (PCA 2D) — coloured by Ward Hierarchical Cluster\n'
        f'{n:,} proteins · {n_clusters} clusters · '
        f'{var.sum()*100:.1f}% variance explained',
        fontsize=13, fontweight='bold',
    )
    ax.legend(loc='best', fontsize=9, markerscale=2, framealpha=0.8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    return fig, emb_2d