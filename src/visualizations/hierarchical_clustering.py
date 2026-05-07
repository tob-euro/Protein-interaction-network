import fastcluster
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score


def find_optimal_clusters(Z, embeddings, k_min=2, k_max=20):
    """Pick k by two criteria: largest height gap in the linkage, and
    the Calinski-Harabasz index.

    Args:
        Z: Ward linkage matrix from fastcluster.linkage_vector.
        embeddings: original embedding matrix (n × d).
        k_min, k_max: range of k values to consider.

    Returns:
        k_gap, k_ch, gaps, ch_scores
    """
    k_range = range(k_min, k_max + 1)

    # Height-gap criterion: cut just before the largest jump in merge heights.
    heights = Z[-k_max:, 2]
    gaps    = np.diff(heights)
    k_gap   = k_max - int(gaps.argmax())

    # Calinski-Harabasz: higher = better separation.
    ch_scores = np.array([
        calinski_harabasz_score(embeddings, fcluster(Z, k, criterion='maxclust'))
        for k in k_range
    ])
    k_ch = k_range[int(ch_scores.argmax())]

    print(f"\nOptimal clusters — height gap: {k_gap}  |  Calinski-Harabasz: {k_ch}")
    return k_gap, k_ch, gaps, ch_scores


def plot_cluster_selection(Z, embeddings, k_min=2, k_max=20):
    """Diagnostic plot of both cluster-selection criteria side by side."""
    k_gap, k_ch, gaps, ch_scores = find_optimal_clusters(Z, embeddings, k_min, k_max)
    k_range = np.arange(k_min, k_max + 1)
    gap_ks  = np.arange(k_max, k_min, -1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(gap_ks, gaps, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].axvline(k_gap, color='tomato', linestyle='--', linewidth=2,
                    label=f'k={k_gap}')
    axes[0].set(xlabel='Number of clusters (k)', ylabel='Height gap',
                title='Height Gap Criterion')
    axes[0].legend(); axes[0].grid(True, axis='y', alpha=0.3)

    axes[1].plot(k_range, ch_scores, marker='o', color='steelblue',
                 linewidth=2, markersize=5)
    axes[1].axvline(k_ch, color='tomato', linestyle='--', linewidth=2,
                    label=f'k={k_ch}')
    axes[1].set(xlabel='Number of clusters (k)', ylabel='Calinski-Harabasz score',
                title='Calinski-Harabasz Index')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle('Optimal Cluster Selection — Ward Linkage',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig, k_gap, k_ch


def plot_dendrogram(model, protein_to_idx, truncate_level=10, idx_to_protein=None,
                    n_clusters=None, k_min=2, k_max=20):
    """Ward-linkage dendrogram of latent embeddings (fastcluster.linkage_vector).

    Args:
        model: a trained LDM-family model.
        protein_to_idx: isoform → index mapping (kept for API parity).
        truncate_level: dendrogram truncation level (None = full tree).
        idx_to_protein: index → name mapping; used as leaf labels when n ≤ 100.
        n_clusters: if None, choose via height-gap criterion.
        k_min, k_max: search range when picking n_clusters automatically.

    Returns:
        fig, cluster_labels, linkage_matrix Z
    """
    embeddings = model.get_embeddings()
    n = embeddings.shape[0]
    print(f"Computing Ward linkage on {n} proteins...")
    Z = fastcluster.linkage_vector(embeddings, method='ward')

    if n_clusters is None:
        k_gap, k_ch, _, _ = find_optimal_clusters(Z, embeddings, k_min, k_max)
        n_clusters = k_gap
        print(f"k_gap={k_gap}, k_ch={k_ch} → using k={n_clusters}")

    cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
    print(f"Assigned {n_clusters} clusters")

    leaf_labels = ([idx_to_protein.get(i, str(i)) for i in range(n)]
                   if idx_to_protein and n <= 100 else None)

    fig, ax = plt.subplots(figsize=(14, 7))
    dendrogram(
        Z, ax=ax,
        truncate_mode='level' if truncate_level else None,
        p=truncate_level,
        labels=leaf_labels,
        color_threshold=Z[-n_clusters, 2],
        above_threshold_color='lightgrey',
        leaf_rotation=90,
        leaf_font_size=7 if leaf_labels else 0,
    )
    ax.set(xlabel='Protein' if leaf_labels else 'Protein index',
           ylabel='Distance',
           title=f'Hierarchical Clustering — Ward Linkage ({n} proteins)')
    ax.title.set_fontweight('bold')
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
    ax.set(xlabel='Cluster', ylabel='Number of Proteins',
           title='Cluster Size Distribution (Ward Linkage)')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_pca_by_cluster(model, cluster_labels, n_clusters=None,
                        idx_to_protein=None, label_top_n=0, colormap='tab10'):
    """PCA-2D scatter coloured by hierarchical cluster.

    Args:
        model: a trained LDM-family model.
        cluster_labels: 1-indexed cluster array from fcluster (n_proteins,).
        n_clusters: total number of clusters (inferred from labels if None).
        idx_to_protein: index → name mapping, used when label_top_n > 0.
        label_top_n: annotate the N proteins closest to each cluster centroid.
        colormap: a qualitative matplotlib colormap.

    Returns:
        fig, pca_embeddings (n × 2)
    """
    embeddings = model.get_embeddings()
    n = embeddings.shape[0]
    if n_clusters is None:
        n_clusters = len(np.unique(cluster_labels))

    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(embeddings)
    var = pca.explained_variance_ratio_
    print(f"\nPCA: PC1={var[0] * 100:.1f}%  PC2={var[1] * 100:.1f}%  "
          f"total={var.sum() * 100:.1f}%")

    cmap = plt.get_cmap(colormap)
    fig, ax = plt.subplots(figsize=(11, 9))
    for c in range(1, n_clusters + 1):
        mask = cluster_labels == c
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   s=18, color=cmap((c - 1) / n_clusters),
                   alpha=0.65, edgecolors='none',
                   label=f'Cluster {c}  (n={mask.sum():,})',
                   rasterized=True)

    if label_top_n > 0 and idx_to_protein is not None:
        for c in range(1, n_clusters + 1):
            members = np.where(cluster_labels == c)[0]
            if len(members) == 0:
                continue
            centroid = emb_2d[members].mean(axis=0)
            closest  = members[np.argsort(
                np.linalg.norm(emb_2d[members] - centroid, axis=1))[:label_top_n]]
            for idx in closest:
                ax.annotate(idx_to_protein.get(idx, str(idx)),
                            (emb_2d[idx, 0], emb_2d[idx, 1]),
                            xytext=(4, 4), textcoords='offset points',
                            fontsize=7, alpha=0.85,
                            bbox=dict(boxstyle='round,pad=0.2',
                                      facecolor='white', alpha=0.5))

    ax.set(xlabel=f'PC1  ({var[0] * 100:.1f}% var)',
           ylabel=f'PC2  ({var[1] * 100:.1f}% var)')
    ax.set_title(f'Latent Space (PCA 2D) — coloured by Ward Cluster\n'
                 f'{n:,} proteins · {n_clusters} clusters · '
                 f'{var.sum() * 100:.1f}% variance',
                 fontweight='bold')
    ax.legend(loc='best', fontsize=9, markerscale=2, framealpha=0.8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig, emb_2d
