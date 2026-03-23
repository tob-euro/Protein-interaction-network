import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import fastcluster
from scipy.cluster.hierarchy import dendrogram, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score


def find_optimal_clusters(Z, embeddings, k_min=2, k_max=20):
    """
    Find the optimal number of clusters from a Ward linkage matrix using two
    independent criteria:

      1. Height gap (acceleration) — free, no extra computation.
         Ward merges at each step by minimising the within-cluster sum-of-squares
         increase. The last rows of Z contain the distances at which the largest
         groups were merged. The biggest jump between consecutive merge heights
         means "Ward paid the most to combine these — they were truly separate."
         We cut just before that jump.

      2. Calinski-Harabasz index (variance ratio) — pick k that maximises
         between-cluster / within-cluster dispersion ratio. Well-suited to Ward
         since both metrics share the same geometric intuition.

    Args:
        Z:          Linkage matrix from fastcluster.linkage_vector (n-1, 4)
        embeddings: The original embedding matrix (n, d)
        k_min:      Minimum k to consider
        k_max:      Maximum k to consider

    Returns:
        k_gap:  Optimal k from height-gap criterion
        k_ch:   Optimal k from Calinski-Harabasz criterion
        gaps:   Array of height gaps for each k in [k_min, k_max]
        ch_scores: Array of CH scores for each k in [k_min, k_max]
    """
    k_range = range(k_min, k_max + 1)

    # ── 1. Height gap criterion ───────────────────────────────────────────────
    # Z[-k, 2] is the merge height threshold that produces k clusters.
    # We look at the height differences between successive k values.
    # The biggest jump at index j means going from k_max-j to k_max-j+1
    # clusters crosses the largest gap → keep k_max-j+1 clusters.
    heights = Z[-(k_max):, 2]          # shape (k_max,), ascending
    gaps    = np.diff(heights)          # shape (k_max-1,)
    # gaps[j] = height[j+1] - height[j]
    # = gap between the k_max-j-1 and k_max-j cluster levels
    j      = gaps.argmax()
    k_gap  = k_max - j                  # cut just before the big jump

    # ── 2. Calinski-Harabasz index ────────────────────────────────────────────
    ch_scores = []
    for k in k_range:
        labels = fcluster(Z, k, criterion='maxclust')
        ch_scores.append(calinski_harabasz_score(embeddings, labels))
    ch_scores = np.array(ch_scores)
    k_ch = k_range[int(ch_scores.argmax())]

    print(f"\nOptimal clusters — height gap: {k_gap}  |  Calinski-Harabasz: {k_ch}")
    return k_gap, k_ch, gaps, ch_scores


def plot_cluster_selection(Z, embeddings, k_min=2, k_max=20):
    """
    Diagnostic plot showing both cluster-selection criteria side by side so
    you can see where they agree and pick with confidence.

    Returns:
        fig, k_gap, k_ch
    """
    k_gap, k_ch, gaps, ch_scores = find_optimal_clusters(
        Z, embeddings, k_min=k_min, k_max=k_max)

    k_range = np.arange(k_min, k_max + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Height gaps ───────────────────────────────────────────────────────────
    # gaps has length k_max-1, corresponding to transitions between
    # k_max, k_max-1, ..., 2 clusters (one gap per pair of consecutive levels)
    gap_ks = np.arange(k_max, k_min, -1)    # k values that the gaps correspond to
    axes[0].bar(gap_ks, gaps, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].axvline(k_gap, color='tomato', linewidth=2,
                    linestyle='--', label=f'Selected k={k_gap}')
    axes[0].set_xlabel('Number of clusters (k)', fontsize=12)
    axes[0].set_ylabel('Height gap (Ward distance increase)', fontsize=11)
    axes[0].set_title('Height Gap Criterion\n(largest = natural cluster boundary)', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, axis='y', alpha=0.3)

    # ── Calinski-Harabasz ─────────────────────────────────────────────────────
    axes[1].plot(k_range, ch_scores, marker='o', color='steelblue',
                 linewidth=2, markersize=5)
    axes[1].axvline(k_ch, color='tomato', linewidth=2,
                    linestyle='--', label=f'Selected k={k_ch}')
    axes[1].set_xlabel('Number of clusters (k)', fontsize=12)
    axes[1].set_ylabel('Calinski-Harabasz score', fontsize=11)
    axes[1].set_title('Calinski-Harabasz Index\n(higher = better-separated clusters)', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Optimal Cluster Selection — Ward Linkage', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig, k_gap, k_ch


def plot_dendrogram(model, protein_to_idx, truncate_level=10, idx_to_protein=None,
                    n_clusters=None, k_min=2, k_max=20):
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

    if n_clusters is None:
        k_gap, k_ch, _, _ = find_optimal_clusters(Z, embeddings, k_min=k_min, k_max=k_max)
        # Prefer height-gap as it is native to hierarchical clustering;
        # fall back to CH if the two agree, otherwise report both and use gap.
        if k_gap == k_ch:
            n_clusters = k_gap
            print(f"Both criteria agree: k={n_clusters}")
        else:
            n_clusters = k_gap
            print(f"Criteria differ (gap={k_gap}, CH={k_ch}) — using height-gap k={n_clusters}")

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


def plot_cluster_sizes(cluster_labels, n_clusters=None):
    """Bar chart of protein counts per cluster."""
    counts = np.bincount(cluster_labels)[1:]   # fcluster labels are 1-indexed
    if n_clusters is None:
        n_clusters = len(counts)

    fig, ax = plt.subplots(figsize=(max(8, n_clusters // 2), 4))
    bars = ax.bar(range(1, n_clusters + 1), counts, edgecolor='black', alpha=0.8)
    ax.bar_label(bars)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Number of Proteins', fontsize=12)
    ax.set_title('Cluster Size Distribution (Ward Linkage)', fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    return fig


def plot_pca_by_cluster(model, cluster_labels, n_clusters=None,
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
    if n_clusters is None:
        n_clusters = len(np.unique(cluster_labels))

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