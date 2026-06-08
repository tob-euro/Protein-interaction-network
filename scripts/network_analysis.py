import os
import yaml
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')


def load_network(csv_file):
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} protein pairs")

    if 'interact' not in df.columns:
        df['interact'] = (df['pi'] >= 0.5).astype(int)

    return df


def create_graph(df, use_interactions_only=False):
    """Build a NetworkX graph from a protein-pair dataframe."""
    if use_interactions_only:
        df = df[df['interact'] == 1]
        print(f"Using {len(df)} positive interactions")
    else:
        print(f"Using all {len(df)} protein pairs")

    G = nx.Graph()
    G.add_nodes_from(df['ensp_1'])
    G.add_nodes_from(df['ensp_2'])
    pos = df[df['interact'] == 1]
    G.add_edges_from(zip(pos['ensp_1'], pos['ensp_2']))
    return G


def analyze_network_statistics(G):
    print("\n--- Network statistics ---")
    degrees = np.array([d for _, d in G.degree()])
    G_lcc   = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    print(f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}  "
          f"Density: {nx.density(G):.6f}")
    print(f"Degrees — isolated: {np.sum(degrees == 0)}  "
          f"connected: {len(degrees) - np.sum(degrees == 0)}")
    print(f"  mean: {np.mean(degrees):.2f}  median: {np.median(degrees):.2f}  "
          f"max: {np.max(degrees)}")
    print(f"Components: {nx.number_connected_components(G)}  "
          f"LCC: {G_lcc.number_of_nodes()} nodes, {G_lcc.number_of_edges()} edges")
    return degrees


def plot_degree_distribution(degrees, save_path='degree_distribution.png', output_dir='figures'):
    counts = Counter(degrees)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(counts.keys(), counts.values(), alpha=0.7, edgecolor='black')
    axes[0].set(xlabel='Degree', ylabel='Count (log)', title='Degree Distribution')
    axes[0].set_yscale('log'); axes[0].grid(True, alpha=0.3)

    deg_sorted = sorted(counts)
    axes[1].loglog(deg_sorted, [counts[d] for d in deg_sorted], 'o', alpha=0.7)
    axes[1].set(xlabel='Degree (log)', ylabel='Count (log)',
                title='Degree Distribution (log-log)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"{output_dir}/{save_path}"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out}")
    return fig


def find_hub_proteins(G, top_n=10):
    sorted_proteins = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    print(f"\n--- Top {top_n} hub proteins ---")
    for i, (protein, degree) in enumerate(sorted_proteins[:top_n], 1):
        print(f"{i:2d}. {protein}: {degree} interactions")
    return sorted_proteins[:top_n]


def analyze_bipartite_network(df):
    gene_isoforms = defaultdict(set)
    for g, i in zip(df['gene_1'], df['ensp_1']):
        gene_isoforms[g].add(i)
    for g, i in zip(df['gene_2'], df['ensp_2']):
        gene_isoforms[g].add(i)

    degrees = [len(s) for s in gene_isoforms.values()]
    print(f"Unique genes: {len(gene_isoforms)}")
    print(f"Isoforms/gene — mean: {np.mean(degrees):.3f}  "
          f"median: {np.median(degrees):.3f}  "
          f"min: {np.min(degrees)}  max: {np.max(degrees)}")
    return degrees


def dataframe_analysis(df, save_path, output_dir):
    pi = df["pi"].to_numpy()
    print(f"  mean: {pi.mean():.3f}  median: {np.median(pi):.3f}  "
          f"min: {pi.min()}  max: {pi.max()}")

    pi_neg = pi[pi <  0.5]
    pi_pos = pi[pi >= 0.5]
    bins   = np.linspace(0, 1, 101)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pi_neg, bins=bins, color='steelblue', alpha=0.7,
            label=f'Negative (pi < 0.5)  n={len(pi_neg):,}')
    ax.hist(pi_pos, bins=bins, color='tomato', alpha=0.7,
            label=f'Positive (pi ≥ 0.5)  n={len(pi_pos):,}')
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1, label='Threshold (0.5)')
    ax.set(xlabel='Interaction probability (pi)', ylabel='Count (log)',
           title='Distribution of interaction probability (pi)')
    ax.set_yscale('log'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{output_dir}/{save_path}"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out}")


def load_gene_gene_network(string_path, mapping_path, use_interactions_only=True):
    """Load STRING gene–gene pairs, map ENSP→ENSG, return a NetworkX graph."""
    string_df    = pd.read_csv(string_path)
    mapping_df   = pd.read_csv(mapping_path)
    ensp_to_ensg = dict(zip(mapping_df['ensp_id'], mapping_df['ensg_id']))

    ensg1 = string_df['protein1'].map(ensp_to_ensg)
    ensg2 = string_df['protein2'].map(ensp_to_ensg)
    mask  = ensg1.notna() & ensg2.notna()
    print(f"  ENSP→ENSG mappable: {mask.sum():,} / {len(string_df):,} "
          f"(dropped {(~mask).sum():,} unmappable)")

    ensg1, ensg2 = ensg1[mask].values, ensg2[mask].values
    labels       = string_df['interact'][mask].values

    G = nx.Graph()
    G.add_nodes_from(set(ensg1) | set(ensg2))
    for g1, g2, interact in zip(ensg1, ensg2, labels):
        if not use_interactions_only or interact == 1:
            G.add_edge(g1, g2, interact=int(interact))

    mode = "positive only" if use_interactions_only else "all pairs"
    print(f"  Gene-gene graph ({mode}): {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")
    return G


def main(iso_path, string_path, mapping_path, figures_dir):
    print("\n--- Protein network analysis ---")
    df = load_network(iso_path)
    dataframe_analysis(df, "Pi_distribution.png", output_dir=figures_dir)

    G = create_graph(df, use_interactions_only=False)
    plot_degree_distribution(analyze_network_statistics(G),
                             save_path="Unipartite_degree_distribution.png", output_dir=figures_dir)
    find_hub_proteins(G, top_n=20)


    print("\n--- Bipartite (gene-isoform) ---")
    plot_degree_distribution(analyze_bipartite_network(df),
                             save_path="Bipartite_degree_distribution.png", output_dir=figures_dir)

    print("\n--- Gene-gene (STRING) ---")
    G_gene = load_gene_gene_network(string_path, mapping_path)
    plot_degree_distribution(analyze_network_statistics(G_gene),
                             save_path="GeneGene_degree_distribution.png", output_dir=figures_dir)
    find_hub_proteins(G_gene, top_n=20)


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    iso_path = cfg['data']['iso_path']
    string_path = cfg['data']['string_path']
    mapping_path = cfg['data']['mapping_path']
    figures_dir = cfg['paths']['figures_dir']

    main(iso_path, string_path, mapping_path, figures_dir)
