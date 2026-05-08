import pandas as pd


def build_gene_gene_eval_pairs(csv_path, gene_to_idx):
    """Aggregate iso-iso interactions to a gene-level evaluation set.

    A gene pair (g_a, g_b) is positive if any isoform-pair between them has
    interact=1 in the iso-iso CSV; negative otherwise. Only gene pairs whose
    both endpoints are present in `gene_to_idx` are kept.

    Args:
        csv_path: path to the iso-iso interaction CSV (gene_1, gene_2, interact).
        gene_to_idx: gene → index mapping from the trained model checkpoint.

    Returns:
        pairs: list of (gene_idx_a, gene_idx_b, label) with a ≤ b.
        n_total: total number of unique gene pairs in the CSV (before filtering).
        n_kept: number of pairs retained after gene-vocab filtering.
    """
    print(f"  Loading iso-iso interactions from {csv_path} ...")
    df = pd.read_csv(csv_path)

    gene_pairs = (df.groupby(['gene_1', 'gene_2'])['interact']
                    .max()
                    .reset_index())
    n_total = len(gene_pairs)

    in_vocab = (gene_pairs['gene_1'].isin(gene_to_idx)
                & gene_pairs['gene_2'].isin(gene_to_idx))
    kept = gene_pairs[in_vocab]
    n_kept = len(kept)
    n_dropped = n_total - n_kept

    pairs = [
        (min(gene_to_idx[a], gene_to_idx[b]),
         max(gene_to_idx[a], gene_to_idx[b]),
         int(l))
        for a, b, l in zip(kept['gene_1'], kept['gene_2'], kept['interact'])
    ]
    n_pos = sum(1 for _, _, l in pairs if l == 1)
    n_neg = len(pairs) - n_pos

    print(f"  Unique gene pairs in CSV: {n_total:,}")
    print(f"  Kept (both genes in model vocab): {n_kept:,}  "
          f"(dropped {n_dropped:,})")
    print(f"  Positives: {n_pos:,}  Negatives: {n_neg:,}  "
          f"ratio 1:{n_neg / max(n_pos, 1):.1f}")
    print(f"  Distinct genes in eval set: "
          f"{len(set(kept['gene_1']) | set(kept['gene_2'])):,}")

    return pairs, n_total, n_kept
