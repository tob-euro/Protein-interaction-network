import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import seaborn as sns
import os

sns.set_style('whitegrid')

# =============================================================================
# SET THIS
# =============================================================================
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)
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

    return G


def analyze_network_statistics(G):
    print("\n" + "="*70)
    print("NETWORK STATISTICS")
    print("="*70)

    degrees = np.array([d for n, d in G.degree()])

    largest_cc_nodes = max(nx.connected_components(G), key=len)

    # Create subgraph
    G_lcc = G.subgraph(largest_cc_nodes).copy()

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
    print("\nConnected components analysis:")
    print(f"  Connected components: {nx.number_connected_components(G)}")
    print(f"  LCC nodes: {G_lcc.number_of_nodes()}")
    print(f"  LCC edges: {G_lcc.number_of_edges()}")

    return degrees


def plot_degree_distribution(degrees, save_path='degree_distribution.png'):
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

def analyze_bipartite_network(df):
    gene_isoforms = defaultdict(set)
    
    for _, row in df.iterrows():
        gene_isoforms[row['gene_1']].add(row['enst_1'])
        gene_isoforms[row['gene_2']].add(row['enst_2'])
    
    unique_genes = list(gene_isoforms.keys())
    degrees = [len(isoforms) for isoforms in gene_isoforms.values()]
    
    n_unique_genes = len(unique_genes)
    mean_degree   = np.mean(degrees)
    median_degree = np.median(degrees)
    min_degree    = np.min(degrees)
    max_degree    = np.max(degrees)
    
    print(f"Number of unique genes:  {n_unique_genes}")
    print(f"\nGene node degree statistics (# of isoforms per gene):")
    print(f"  Mean:    {mean_degree:.3f}")
    print(f"  Median:  {median_degree:.3f}")
    print(f"  Min:     {min_degree}")
    print(f"  Max:     {max_degree}")
    
    return degrees



def plot_adjacency_matrix(G):
    A = nx.to_numpy_array(G)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(A)
    ax.set_title("Adjacency Matrix")
    ax.set_xlabel("Node index")
    ax.set_ylabel("Node index")
    plt.savefig(f"{FIGURES_DIR}/adjacency_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {FIGURES_DIR}/adjacency_matrix.png")

def dataframe_analysis(df):
    pi = df["pi"].to_numpy()
    print("pi")

def main(csv_file):
    print("\n" + "="*70)
    print("PROTEIN NETWORK ANALYSIS")
    print("="*70 + "\n")

    df = load_network(csv_file)
    dataframe_analysis(df)

    # G = create_graph(df, use_interactions_only=False)
    # degrees = analyze_network_statistics(G)
    # plot_degree_distribution(degrees, save_path="Unipartite_degree_distribution.png")

    # find_hub_proteins(G, top_n=20)
    # G_pos = create_graph(df, use_interactions_only=True)
    # plot_adjacency_matrix(G_pos)

    # print("\n" + "="*70)
    # print("Bipartite Gene-Isoform Graph Analysis")
    # print("="*70 + "\n")

    # degrees_bi = analyze_bipartite_network(df)
    # plot_degree_distribution(degrees_bi, save_path="Bipartite_degree_distribution.png")






if __name__ == "__main__":
    main("data/results_PHYSICAL_Prob_Model_16_02_26_filtered.csv")