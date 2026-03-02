import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import os
from src.data_scripts.dataset import load_and_prepare_data
from src.training.evaluate import load_trained_model


def calculate_node_degrees(data, protein_to_idx):
    """Calculate degree for each protein
    
    Args:
        data: pandas DataFrame (for calculating degrees)
        protein_to_idx: dictionary (Protein id -> index mapping)
    
    Returns:
        degrees (numpy array)"""
    G = nx.Graph()
    positive_interactions = data[data['interact'] == 1]
    for _, row in positive_interactions.iterrows():
        G.add_edge(row['ensp_1'], row['ensp_2'])
    
    degrees = np.zeros(len(protein_to_idx))
    for protein, idx in protein_to_idx.items():
        if protein in G:
            degrees[idx] = G.degree(protein)
    return degrees


def visualize_latent_space_pca(model, protein_to_idx, data=None, degrees=None,
                                n_components=2, sample_size=None,
                                idx_to_protein=None, colormap='viridis', show_variance=True):
    """
    Visualize high-dimensional latent space using PCA
    
    Args:
        model: Trained LatentDistanceModel
        protein_to_idx: Protein to index mapping
        data: DataFrame (for calculating degrees)
        degrees: Pre-computed degrees (optional)
        n_components: Number of PCA components (2 or 3)
        sample_size: Number of proteins to sample (None = all)
        idx_to_protein: Index to protein mapping (for labels)
        colormap: Matplotlib colormap
        show_variance: Show explained variance in title
    
    Returns:
        fig, pca_embeddings, pca_model
    """
    # Get embeddings
    embeddings = model.get_embeddings()
    original_dim = embeddings.shape[1]
    
    print(f"\nPCA Dimensionality Reduction:")
    print(f"  Original dimension: {original_dim}")
    print(f"  Target dimension: {n_components}")
    
    # Calculate degrees
    if degrees is None and data is not None:
        degrees = calculate_node_degrees(data, protein_to_idx)
    elif degrees is None:
        degrees = np.ones(embeddings.shape[0])
    
    # Sample if needed
    if sample_size is not None and embeddings.shape[0] > sample_size:
        indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
        embeddings_sample = embeddings[indices]
        degrees_sample = degrees[indices]
    else:
        embeddings_sample = embeddings
        degrees_sample = degrees
        indices = np.arange(embeddings.shape[0])
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings_sample)
    
    explained_var = pca.explained_variance_ratio_
    print(f"  Explained variance: {explained_var}")
    print(f"  Total variance explained: {explained_var.sum()*100:.2f}%")
    
    # Normalize degrees for visualization
    min_size, max_size = 20, 500
    if degrees_sample.max() > 0:
        sizes = min_size + (degrees_sample / degrees_sample.max()) * (max_size - min_size)
    else:
        sizes = np.full(len(degrees_sample), min_size)
    
    colors = degrees_sample
    
    # Create visualization
    if n_components == 2:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        scatter = ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1],
                           s=sizes, c=colors, cmap=colormap,
                           alpha=0.6, edgecolors='black', linewidths=0.5)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Number of Interactions', fontsize=12, rotation=270, labelpad=20)
        
        ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', fontsize=12)
        
        title = f'Latent Space Visualization (PCA: {original_dim}D → 2D)'
        if show_variance:
            title += f'\nTotal variance explained: {explained_var.sum()*100:.1f}%'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
    elif n_components == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(embeddings_pca[:, 0],
                           embeddings_pca[:, 1],
                           embeddings_pca[:, 2],
                           s=sizes, c=colors, cmap=colormap,
                           alpha=0.6, edgecolors='black', linewidths=0.5)
        
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Number of Interactions', fontsize=12, rotation=270, labelpad=20)
        
        ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontsize=12)
        ax.set_zlabel(f'PC3 ({explained_var[2]*100:.1f}%)', fontsize=12)
        
        title = f'Latent Space Visualization (PCA: {original_dim}D → 3D)'
        if show_variance:
            title += f'\nTotal variance: {explained_var.sum()*100:.1f}%'
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    else:
        raise ValueError("n_components must be 2 or 3")
    
    plt.tight_layout()
    
    return fig, embeddings_pca, pca


def visualize_pca_variance(model, max_components=50):
    """
    Visualize explained variance across PCA components
    
    Helps decide how many components to use
    """
    embeddings = model.get_embeddings()
    original_dim = embeddings.shape[1]
    
    # Calculate PCA with all components
    n_comp = min(max_components, original_dim, embeddings.shape[0])
    pca = PCA(n_components=n_comp)
    pca.fit(embeddings)
    
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual variance
    ax1 = axes[0]
    ax1.bar(range(1, len(explained_var)+1), explained_var, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Variance Explained by Each Component', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Mark first 3 components
    for i in range(min(3, len(explained_var))):
        ax1.text(i+1, explained_var[i], f'{explained_var[i]*100:.1f}%',
                ha='center', va='bottom', fontsize=9)
    
    # Cumulative variance
    ax2 = axes[1]
    ax2.plot(range(1, len(cumulative_var)+1), cumulative_var, 'b-o', linewidth=2, markersize=4)
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance', alpha=0.7)
    ax2.axhline(y=0.90, color='orange', linestyle='--', label='90% variance', alpha=0.7)
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('Cumulative Variance Explained', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # Find how many components for 90% and 95% variance
    n_90 = np.argmax(cumulative_var >= 0.90) + 1
    n_95 = np.argmax(cumulative_var >= 0.95) + 1
    
    ax2.text(n_90, 0.90, f'{n_90} PCs', ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
    ax2.text(n_95, 0.95, f'{n_95} PCs', ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    plt.tight_layout()
    
    print(f"\nPCA Variance Analysis:")
    print(f"  Original dimensions: {original_dim}")
    print(f"  Components for 90% variance: {n_90}")
    print(f"  Components for 95% variance: {n_95}")
    print(f"  First 3 PCs explain: {cumulative_var[2]*100:.2f}%")
    
    return fig, explained_var, cumulative_var