import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

sns.set_style('whitegrid')

# =============================================================================
# SET THIS
# =============================================================================
FIGURES_DIR = "figures"
# =============================================================================


def load_network(csv_file):
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} protein pairs")
    return df


def create_graph(df, use_interactions_only=False):
    """
    Create a NetworkX graph from protein interaction data.

    Args:
        df:                    DataFrame with protein interactions
        use_interactions_only: if True, only include rows where interact=1
    """
    G = nx.Graph()

    if use_interactions_only:
        df_filtered = df[df['interact'] == 1]
        print(f"Using {len(df_filtered)} positive interactions")
    else:
        df_filtered = df
        print(f"Using all {len(df_filtered)} protein pairs")

    for _, row in df_filtered.iterrows():
        G.add_node(row['ensp_1'])
        G.add_node(row['ensp_2'])
        if row['interact']:
            G.add_edge(row['ensp_1'], row['ensp_2'], interact=row['interact'])

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def analyze_network_statistics(G):
    print("\n" + "="*70)
    print("NETWORK STATISTICS")
    print("="*70)

    degrees = np.array([d for n, d in G.degree()])
    print(f"\nBasic Properties:")
    print(f"  Nodes (proteins): {G.number_of_nodes()}")
    print(f"  Edges (interactions): {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.6f}")
    print(f"\nDegree Statistics:")
    print(f"  Nodes with no interactions: {np.sum(degrees==0)}")
    print(f"  Nodes with interactions: {len(degrees) - np.sum(degrees==0)}")
    print(f"  Mean degree: {np.mean(degrees):.2f}")
    print(f"  Median degree: {np.median(degrees):.2f}")
    print(f"  Max degree: {np.max(degrees)}")

    return {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_degree': np.mean(degrees),
    }


def plot_degree_distribution(G, save_path='degree_distribution.png'):
    degrees = [d for n, d in G.degree()]
    degree_counts = Counter(degrees)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(degree_counts.keys(), degree_counts.values(), alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Degree')
    axes[0].set_ylabel('Count (log scale)')
    axes[0].set_yscale('log')
    axes[0].set_title('Degree Distribution')
    axes[0].grid(True, alpha=0.3)

    degrees_sorted = sorted(degree_counts.keys())
    counts_sorted = [degree_counts[d] for d in degrees_sorted]
    axes[1].loglog(degrees_sorted, counts_sorted, 'o', alpha=0.7)
    axes[1].set_xlabel('Degree (log scale)')
    axes[1].set_ylabel('Count (log scale)')
    axes[1].set_title('Degree Distribution (log-log)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{save_path}", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {FIGURES_DIR}/{save_path}")
    return fig


def find_hub_proteins(G, top_n=10):
    degrees = dict(G.degree())
    sorted_proteins = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'='*70}\nTOP {top_n} HUB PROTEINS\n{'='*70}")
    for i, (protein, degree) in enumerate(sorted_proteins[:top_n], 1):
        print(f"{i:2d}. {protein}: {degree} interactions")

    return sorted_proteins[:top_n]


def plot_adjacency_matrix(G):
    A = nx.to_numpy_array(G)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(A)
    fig.colorbar(im, ax=ax)
    ax.set_title("Adjacency Matrix")
    ax.set_xlabel("Node index")
    ax.set_ylabel("Node index")
    plt.savefig(f"{FIGURES_DIR}/adjacency_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {FIGURES_DIR}/adjacency_matrix.png")


def main(csv_file):
    print("\n" + "="*70)
    print("PROTEIN NETWORK ANALYSIS")
    print("="*70 + "\n")

    df = load_network(csv_file)

    G = create_graph(df, use_interactions_only=False)
    analyze_network_statistics(G)
    plot_degree_distribution(G)
    find_hub_proteins(G, top_n=20)

    G = create_graph(df, use_interactions_only=True)
    plot_adjacency_matrix(G)


if __name__ == "__main__":
    main("data/results_PHYSICAL_Prob_Model_22_01_26.csv")