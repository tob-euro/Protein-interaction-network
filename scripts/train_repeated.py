"""Train with N different random splits; report mean ± std and 95% CI statistics.

Usage:
    python scripts/train_repeated.py --n_seeds 5
    python scripts/train_repeated.py --n_seeds 10 --base_seed 0 --config config/config.yaml

Each run uses a distinct seed that controls both the data split and model weight
initialisation, giving an unbiased estimate of performance variance over splits.
Per-run model checkpoints are saved to individual timestamped directories.
Aggregate statistics and a per-run CSV are written to a shared output directory.
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from src.data_scripts.split_cache import SplitCache
from src.training.runner import run_single


def _stats(values):
    """Return (mean, std, 95% CI half-width) for a sequence of floats."""
    a = np.array(values, dtype=float)
    n = len(a)
    mean = float(a.mean())
    std  = float(a.std(ddof=1))
    ci   = 1.96 * std / np.sqrt(n)
    return mean, std, ci


def _print_table(rows, title):
    """Print a formatted summary table.

    rows: list of (label, auc_values, ap_values)
    """
    col = 24
    print(f"\n{title}")
    print("-" * 76)
    print(f"{'':>{col}}  {'AUC  (mean ± std  [95% CI])':^28}  "
          f"{'AP  (mean ± std  [95% CI])':^28}")
    print("-" * 76)
    for label, auc_vals, ap_vals in rows:
        am, as_, ac = _stats(auc_vals)
        pm, ps_, pc = _stats(ap_vals)
        print(f"  {label:<{col-2}}  {am:.4f} ± {as_:.4f}  [{ac:.4f}]      "
              f"{pm:.4f} ± {ps_:.4f}  [{pc:.4f}]")
    print("-" * 76)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__)
    parser.add_argument('--config',     default='config/config.yaml',
                        help='YAML config file')
    parser.add_argument('--n_seeds',    type=int, default=5,
                        help='Number of random seeds to run')
    parser.add_argument('--base_seed',  type=int, default=42,
                        help='First seed; run i uses seed base_seed + i')
    parser.add_argument('--output_dir', default=None,
                        help='Directory for aggregate CSV and summary. '
                             'Defaults to <models_dir>/repeated_<timestamp>.')
    parser.add_argument('--cache',     action='store_true',
                        help='Enable disk cache for data splits (off by default). '
                             'Strongly recommended when comparing models over the '
                             'same seeds, as it avoids re-reading the CSV and '
                             're-sampling negatives for every run.')
    parser.add_argument('--cache_dir', default='.cache/data_splits',
                        help='Directory for cached split files')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seeds       = list(range(args.base_seed, args.base_seed + args.n_seeds))
    models_dir  = cfg.get('paths', {}).get('models_dir', 'models')
    split_cache = SplitCache(args.cache_dir) if args.cache else None

    if args.output_dir is None:
        timestamp       = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join(models_dir, f'repeated_{timestamp}')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Repeated-split training: {args.n_seeds} runs  "
          f"(seeds {seeds[0]}–{seeds[-1]})")
    print(f"Config:      {args.config}")
    print(f"Split cache: {'enabled → ' + args.cache_dir if args.cache else 'disabled'}")
    print(f"Results:     {args.output_dir}")
    print(f"{'='*70}")

    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n{'='*70}")
        print(f"RUN {i + 1}/{args.n_seeds}  (seed={seed})")
        print(f"{'='*70}")
        metrics = run_single(cfg, seed, args.config, split_cache=split_cache,
                             parent_dir=args.output_dir)
        all_results.append(metrics)

    # ── Aggregate statistics ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("REPEATED-SPLIT SUMMARY")
    print(f"{'='*70}")

    rows = [("Overall test set",
             [r['test_auc'] for r in all_results],
             [r['test_ap']  for r in all_results])]

    if 'both_unseen_auc' in all_results[0]:
        rows += [
            ("Both-unseen",
             [r['both_unseen_auc'] for r in all_results],
             [r['both_unseen_ap']  for r in all_results]),
            ("One-unseen",
             [r['one_unseen_auc'] for r in all_results],
             [r['one_unseen_ap']  for r in all_results]),
        ]

    if 'gene_gene_auc' in all_results[0]:
        rows.append(("Gene-gene",
                     [r['gene_gene_auc'] for r in all_results],
                     [r['gene_gene_ap']  for r in all_results]))

    _print_table(rows, f"Statistics across {args.n_seeds} splits")

    # ── Per-seed breakdown ────────────────────────────────────────────────────
    has_inductive = 'both_unseen_auc' in all_results[0]
    has_gg        = 'gene_gene_auc'   in all_results[0]

    header = f"  {'Seed':>4}  {'AUC':>8}  {'AP':>8}"
    if has_inductive:
        header += f"  {'BU-AUC':>8}  {'BU-AP':>8}  {'OU-AUC':>8}  {'OU-AP':>8}"
    if has_gg:
        header += f"  {'GG-AUC':>8}  {'GG-AP':>8}"
    print(f"\nPer-seed breakdown:\n{header}")

    for r in all_results:
        row = f"  {r['seed']:>4}  {r['test_auc']:>8.4f}  {r['test_ap']:>8.4f}"
        if has_inductive:
            row += (f"  {r['both_unseen_auc']:>8.4f}  {r['both_unseen_ap']:>8.4f}"
                    f"  {r['one_unseen_auc']:>8.4f}  {r['one_unseen_ap']:>8.4f}")
        if has_gg:
            row += f"  {r['gene_gene_auc']:>8.4f}  {r['gene_gene_ap']:>8.4f}"
        print(row)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results).drop(columns=['save_dir'], errors='ignore')
    csv_path = os.path.join(args.output_dir, 'repeated_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nPer-run results saved to: {csv_path}")
    print(f"Individual checkpoints in per-run subdirectories of: {models_dir}\n")


if __name__ == '__main__':
    main()
