import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA


def calculate_node_degrees(data, protein_to_idx):
    """Per-protein degree (positive interactions only).

    Args:
        data: DataFrame with ensp_1, ensp_2, interact columns.
        protein_to_idx: isoform → index mapping.

    Returns:
        np.ndarray of shape (num_proteins,) with degree per index.
    """
    G = nx.Graph()
    for _, row in data[data['interact'] == 1].iterrows():
        G.add_edge(row['ensp_1'], row['ensp_2'])
    degrees = np.zeros(len(protein_to_idx))
    for protein, idx in protein_to_idx.items():
        if protein in G:
            degrees[idx] = G.degree(protein)
    return degrees


def visualize_latent_space_pca(model, protein_to_idx, data=None, degrees=None,
                               n_components=2, sample_size=None,
                               idx_to_protein=None, colormap='viridis',
                               show_variance=True):
    """PCA scatter of latent embeddings; point size and colour scale with degree.

    Args:
        model: a trained LDM-family model (provides .get_embeddings()).
        protein_to_idx: isoform → index mapping.
        data: optional interaction DataFrame for degree colouring.
        degrees: pre-computed degree array (overrides `data`).
        n_components: 2 or 3.
        sample_size: if set and smaller than the embedding count, sample randomly.
        idx_to_protein: index → name mapping (currently unused; kept for API parity).
        colormap: matplotlib colormap name.
        show_variance: include total explained variance in the title.

    Returns:
        fig, pca_embeddings (n × n_components), pca_model
    """
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")

    embeddings   = model.get_embeddings()
    original_dim = embeddings.shape[1]
    print(f"\nPCA: {original_dim}D → {n_components}D")

    if degrees is None:
        degrees = (calculate_node_degrees(data, protein_to_idx)
                   if data is not None else np.ones(embeddings.shape[0]))

    if sample_size is not None and embeddings.shape[0] > sample_size:
        idx = np.random.choice(embeddings.shape[0], sample_size, replace=False)
        embeddings, degrees = embeddings[idx], degrees[idx]

    pca = PCA(n_components=n_components, random_state=42)
    emb_pca = pca.fit_transform(embeddings)
    var = pca.explained_variance_ratio_
    print(f"  Explained variance: {var}  total: {var.sum() * 100:.2f}%")

    sizes = (20 + (degrees / degrees.max()) * 480
             if degrees.max() > 0 else np.full(len(degrees), 20))

    if n_components == 2:
        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(emb_pca[:, 0], emb_pca[:, 1],
                             s=sizes, c=degrees, cmap=colormap,
                             alpha=0.6, edgecolors='black', linewidths=0.5)
        plt.colorbar(scatter, ax=ax, label='Number of Interactions')
        ax.set(xlabel=f'PC1 ({var[0] * 100:.1f}% var)',
               ylabel=f'PC2 ({var[1] * 100:.1f}% var)')
        title = f'Latent Space (PCA {original_dim}D → 2D)'
    else:
        fig = plt.figure(figsize=(12, 10))
        ax  = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(emb_pca[:, 0], emb_pca[:, 1], emb_pca[:, 2],
                             s=sizes, c=degrees, cmap=colormap,
                             alpha=0.6, edgecolors='black', linewidths=0.5)
        plt.colorbar(scatter, ax=ax, pad=0.1, label='Number of Interactions')
        ax.set_xlabel(f'PC1 ({var[0] * 100:.1f}%)')
        ax.set_ylabel(f'PC2 ({var[1] * 100:.1f}%)')
        ax.set_zlabel(f'PC3 ({var[2] * 100:.1f}%)')
        title = f'Latent Space (PCA {original_dim}D → 3D)'

    if show_variance:
        title += f'\nTotal variance: {var.sum() * 100:.1f}%'
    ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    return fig, emb_pca, pca


def visualize_pca_variance(model, max_components=50):
    """Per-component and cumulative explained variance plots.

    Args:
        model: a trained LDM-family model (provides .get_embeddings()).
        max_components: max number of PCs to evaluate.

    Returns:
        fig, explained_variance_ratio, cumulative_variance
    """
    embeddings   = model.get_embeddings()
    original_dim = embeddings.shape[1]
    n_comp       = min(max_components, original_dim, embeddings.shape[0])

    pca = PCA(n_components=n_comp).fit(embeddings)
    var = pca.explained_variance_ratio_
    cum = np.cumsum(var)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(1, len(var) + 1), var, alpha=0.7, edgecolor='black')
    axes[0].set(xlabel='Principal Component', ylabel='Explained Variance Ratio',
                title='Variance per Component')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i in range(min(3, len(var))):
        axes[0].text(i + 1, var[i], f'{var[i] * 100:.1f}%',
                     ha='center', va='bottom', fontsize=9)

    axes[1].plot(range(1, len(cum) + 1), cum, 'b-o', linewidth=2, markersize=4)
    axes[1].axhline(0.95, color='r',      linestyle='--', alpha=0.7, label='95% variance')
    axes[1].axhline(0.90, color='orange', linestyle='--', alpha=0.7, label='90% variance')
    axes[1].set(xlabel='Number of Components', ylabel='Cumulative Variance',
                title='Cumulative Variance', ylim=[0, 1.05])
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    n_90 = int(np.argmax(cum >= 0.90)) + 1
    n_95 = int(np.argmax(cum >= 0.95)) + 1
    axes[1].text(n_90, 0.90, f'{n_90} PCs', ha='center', va='bottom', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
    axes[1].text(n_95, 0.95, f'{n_95} PCs', ha='center', va='bottom', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

    plt.tight_layout()
    print(f"\nPCA variance: dim={original_dim}  90% at {n_90} PCs  95% at {n_95} PCs  "
          f"first 3 PCs: {cum[2] * 100:.2f}%")
    return fig, var, cum
