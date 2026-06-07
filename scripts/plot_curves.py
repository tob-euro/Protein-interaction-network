import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import t as student_t
from sklearn.metrics import ( average_precision_score, precision_recall_curve, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from src.data_scripts.isoform_pairs import ProteinInteractionDataset
from src.data_scripts.split_cache import SplitCache
from src.training.evaluate import load_model
from src.training.runner import _build_split_key, _load_all_splits


DTU_COLORS = {
    'dtured': '#990000',
    'blue': '#2F3EEA',
    'brightgreen': '#1FD082',
    'navyblue': '#030F4F',
    'yellow': '#F6D04D',
    'orange': '#FC7634',
    'pink': '#F7BBB1',
    'grey': '#DADADA',
    'red': '#E83F48',
    'green': '#008835',
    'purple': '#79238E',
}
DTU_CYCLE = [
    DTU_COLORS['dtured'],
    DTU_COLORS['blue'],
    DTU_COLORS['brightgreen'],
    DTU_COLORS['navyblue'],
    DTU_COLORS['yellow'],
    DTU_COLORS['orange'],
    DTU_COLORS['grey'],
    DTU_COLORS['red'],
    DTU_COLORS['green'],
    DTU_COLORS['purple'],
]


def _apply_dtu_matplotlib_style():
    """Approximate the DTU thesis plot style for rendered PNG figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'axes.prop_cycle': plt.cycler(color=DTU_CYCLE),
        'axes.edgecolor': '#666666',
        'axes.linewidth': 0.8,
        'axes.labelcolor': '#111111',
        'xtick.color': '#111111',
        'ytick.color': '#111111',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.major.size': 0,
        'ytick.major.size': 0,
        'grid.color': DTU_COLORS['grey'],
        'grid.linewidth': 0.5,
        'grid.alpha': 1.0,
        'legend.fontsize': 7,
        'legend.frameon': False,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def _style_dtu_axes(ax, grid_axis='both'):
    ax.grid(True, axis=grid_axis)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _group_sort_key(name):
    """Sort key: numeric latent_dim value when present, else the bare name."""
    m = re.search(r'latent_dim=(\d+)', name)
    return int(m.group(1)) if m else name


def _find_groups(models_dir, search_word):
    """List of group dirs whose basename contains search_word, sorted by latent_dim."""
    matches = [
        name for name in os.listdir(models_dir)
        if search_word in name and os.path.isdir(os.path.join(models_dir, name))
    ]
    matches.sort(key=_group_sort_key)
    return [os.path.join(models_dir, name) for name in matches]


def _find_seed_dirs(group_dir):
    """Sorted list of seed_N subdirectories inside a group directory."""
    return sorted(
        os.path.join(group_dir, name)
        for name in sorted(os.listdir(group_dir))
        if re.match(r'seed_\d+$', name)
        and os.path.isdir(os.path.join(group_dir, name))
    )


def _seed_from_path(path):
    """Extract integer seed from a path ending in 'seed_N'."""
    m = re.search(r'seed_(\d+)$', path.rstrip('/\\'))
    return int(m.group(1)) if m else None


def _curve_label(group_basename):
    """Short display label derived from the group folder name.

    Tries to extract the latent_dim value and returns 'Latent dimension N'.
    Falls back to the full basename if the pattern is absent.
    """
    m = re.search(r'latent_dim=(\d+)', group_basename)
    if m:
        return f"Latent dimension {m.group(1)}"
    return group_basename


def _get_test_data(cfg, seed, cache):
    """Return (test_data DataFrame, protein_to_idx) using cache when possible."""
    d  = cfg['data']
    mm = cfg.get('multimodal', {})
    model_type   = cfg['model']['type']
    is_inductive = d['mode'] == 'inductive'

    key        = _build_split_key(d, mm, seed, model_type)
    split_data = cache.get(key) if cache else None

    if split_data is None:
        split_data = _load_all_splits(d, mm, seed, model_type, is_inductive)
        if cache:
            cache.put(key, split_data)

    return split_data['test_data'], split_data['protein_to_idx']


def _run_inference(model, test_data, protein_to_idx, device, batch_size=1024):
    """Return (preds, labels) as numpy arrays from a test DataFrame."""
    loader = DataLoader(
        ProteinInteractionDataset(test_data, protein_to_idx),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for p1, p2, y in loader:
            out = torch.sigmoid(model(p1.to(device), p2.to(device)))
            preds.extend(out.cpu().numpy())
            labels.extend(y.numpy())
    return np.array(preds), np.array(labels)


def _interp_roc(fpr, tpr, grid):
    return np.interp(grid, fpr, tpr)


def _interp_pr(recall, precision, grid):
    # sklearn returns a step curve with recall in decreasing order and many
    # duplicate recall values. np.interp expects strictly increasing x-values;
    # using duplicate recall coordinates can create artificial one-point spikes
    # in averaged PR plots. Collapse duplicates before interpolation.
    idx = np.argsort(recall)
    recall_sorted = recall[idx]
    precision_sorted = precision[idx]

    recall_unique, inverse = np.unique(recall_sorted, return_inverse=True)
    precision_unique = np.zeros_like(recall_unique, dtype=float)
    np.maximum.at(precision_unique, inverse, precision_sorted)

    # Precision-recall plots are commonly displayed as the precision envelope.
    # This affects only the displayed curve/band, not AP values.
    precision_unique = np.maximum.accumulate(precision_unique[::-1])[::-1]
    return np.interp(grid, recall_unique, precision_unique)


def _ci95(values, axis=0):
    values = np.asarray(values, dtype=float)
    n = values.shape[axis]
    if n < 2:
        return np.zeros_like(values.mean(axis=axis))
    return (
        student_t.ppf(0.975, n - 1)
        * values.std(axis=axis, ddof=1)
        / np.sqrt(n)
    )


# ── Per-group processing ──────────────────────────────────────────────────────

def process_group(group_dir, cache, device, fpr_grid, recall_grid, batch_size):
    seed_dirs = _find_seed_dirs(group_dir)
    if not seed_dirs:
        print(f"  [skip] no seed directories in {group_dir}")
        return None

    dirname     = os.path.basename(group_dir)
    curve_label = _curve_label(dirname)
    print(f"\nGroup: {dirname}  ({len(seed_dirs)} seeds)")

    tpr_curves, prec_curves, aucs, aps = [], [], [], []

    for seed_dir in seed_dirs:
        seed        = _seed_from_path(seed_dir)
        model_path  = os.path.join(seed_dir, 'model.pt')
        config_path = os.path.join(seed_dir, 'config.yaml')

        if not (os.path.exists(model_path) and os.path.exists(config_path)):
            print(f"  [skip] missing files in {seed_dir}")
            continue

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # load_model returns (model, protein_to_idx, checkpoint)
        # Use protein_to_idx from the checkpoint to stay consistent with the
        # model's embedding table even if the cache is rebuilt from a different
        # working directory.
        model, protein_to_idx_ckpt, _ = load_model(model_path, device=device)

        test_data, _ = _get_test_data(cfg, seed, cache)

        preds, labels = _run_inference(
            model,
            test_data,
            protein_to_idx_ckpt,
            device,
            batch_size=batch_size,
        )

        fpr, tpr, _       = roc_curve(labels, preds)
        recall, prec, _   = precision_recall_curve(labels, preds)
        auc = roc_auc_score(labels, preds)
        ap  = average_precision_score(labels, preds)

        tpr_curves.append(_interp_roc(fpr, tpr, fpr_grid))
        prec_curves.append(_interp_pr(recall, prec, recall_grid))
        aucs.append(auc)
        aps.append(ap)
        print(f"  seed={seed:>3}  AUC={auc:.4f}  AP={ap:.4f}")

    if not tpr_curves:
        return None

    n = len(aucs)
    mean_auc, std_auc = np.mean(aucs), np.std(aucs, ddof=1)
    mean_ap,  std_ap  = np.mean(aps),  np.std(aps,  ddof=1)
    ci_auc = _ci95(aucs)
    ci_ap  = _ci95(aps)

    tpr_arr  = np.vstack(tpr_curves)
    prec_arr = np.vstack(prec_curves)

    print(f"  → AUC {mean_auc:.4f}  [95% CI ±{ci_auc:.4f}]  "
          f"AP {mean_ap:.4f}  [95% CI ±{ci_ap:.4f}]")

    return {
        'curve_label': curve_label,
        'mean_auc': mean_auc, 'std_auc': std_auc, 'ci_auc': ci_auc,
        'mean_ap':  mean_ap,  'std_ap':  std_ap,  'ci_ap':  ci_ap,
        'aucs': aucs,
        'aps':  aps,
        'mean_tpr':  tpr_arr.mean(axis=0),  'ci_tpr':  _ci95(tpr_arr, axis=0),
        'mean_prec': prec_arr.mean(axis=0), 'ci_prec': _ci95(prec_arr, axis=0),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_band(ax, x, mean, ci, label, color, alpha=0.15):
    ax.plot(x, mean, color=color, label=label, linewidth=1.8)
    lower = np.clip(mean - ci, 0, 1)
    upper = np.clip(mean + ci, 0, 1)
    ax.fill_between(x, lower, upper, color=color, alpha=alpha)


def make_figures(results, fpr_grid, recall_grid, title):
    _apply_dtu_matplotlib_style()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig_roc, ax_roc = plt.subplots(figsize=(6.3, 4.2))
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='#666666', label='Random', linewidth=1)

    fig_pr, ax_pr = plt.subplots(figsize=(6.3, 4.2))

    for i, r in enumerate(results):
        color = colors[i % len(colors)]

        lbl_roc = (
            f"{r['curve_label']} "
            f"(AUC {r['mean_auc']:.3f} ± {r['ci_auc']:.3f})"
        )
        lbl_pr = (
            f"{r['curve_label']} "
            f"(AP {r['mean_ap']:.3f} ± {r['ci_ap']:.3f})"
        )

        _plot_band(ax_roc, fpr_grid,    r['mean_tpr'],  r['ci_tpr'],  lbl_roc, color)
        _plot_band(ax_pr,  recall_grid, r['mean_prec'], r['ci_prec'], lbl_pr,  color)

    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc='lower center', bbox_to_anchor=(0.5, -0.36), ncol=2)
    _style_dtu_axes(ax_roc, grid_axis='both')
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1)

    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.legend(loc='lower center', bbox_to_anchor=(0.5, -0.28), ncol=2)
    _style_dtu_axes(ax_pr, grid_axis='both')
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1)

    fig_roc.tight_layout(rect=(0, 0.16, 1, 1))
    fig_pr.tight_layout(rect=(0, 0.10, 1, 1))
    return fig_roc, fig_pr


def make_boxplot(results, title):
    """Box plots showing the per-seed AUC and AP distributions per group."""
    _apply_dtu_matplotlib_style()
    # Extract just the numeric part of each label for tick marks, e.g. "0", "2"
    tick_labels = []
    for r in results:
        m = re.search(r'(\d+)$', r['curve_label'])
        tick_labels.append(m.group(1) if m else r['curve_label'])

    auc_data = [r['aucs'] for r in results]
    ap_data  = [r['aps']  for r in results]
    colors   = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, (ax_auc, ax_ap) = plt.subplots(1, 2, figsize=(6.6, 3.3))

    for ax, data, metric in ((ax_auc, auc_data, 'AUC-ROC'),
                             (ax_ap,  ap_data,  'Average Precision')):
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color='black', linewidth=1.5))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Overlay individual seed points with a small jitter to avoid overlap
        rng = np.random.default_rng(0)
        for j, vals in enumerate(data, start=1):
            x = rng.uniform(j - 0.08, j + 0.08, size=len(vals))
            ax.scatter(x, vals, color=colors[(j - 1) % len(colors)],
                       s=25, zorder=3, alpha=0.8)

        ax.set_xticks(range(1, len(tick_labels) + 1))
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_xlabel('Latent dimension')
        ax.set_ylabel(metric)
        _style_dtu_axes(ax, grid_axis='y')
        ax.set_ylim(bottom=max(0, min(min(v) for v in data) - 0.05))

    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__)
    parser.add_argument('search',
                        help='Substring to match against model group directory names')
    parser.add_argument('--models_dir',  default='models',
                        help='Root directory containing model group folders')
    parser.add_argument('--cache_dir',   default='.cache/data_splits',
                        help='Directory for split cache files')
    parser.add_argument('--output_dir',  default=None,
                        help='Where to save figures (default: models_dir/plots)')
    parser.add_argument('--no_cache',    action='store_true',
                        help='Disable split cache — forces re-reading the CSV')
    parser.add_argument('--n_points',    type=int, default=1000,
                        help='Resolution of the interpolation grid')
    parser.add_argument('--batch_size',  type=int, default=16384,
                        help='Inference batch size used when scoring test pairs')
    parser.add_argument('--title',       default=None,
                        help='Title for all plots (defaults to the search word)')
    args = parser.parse_args()

    device = ('cuda' if torch.cuda.is_available()
              else 'mps' if torch.backends.mps.is_available()
              else 'cpu')
    print(f"Device: {device}")

    cache = None if args.no_cache else SplitCache(args.cache_dir)

    groups = _find_groups(args.models_dir, args.search)
    if not groups:
        print(f"No model groups matching '{args.search}' in '{args.models_dir}'")
        sys.exit(1)

    print(f"\nFound {len(groups)} group(s) matching '{args.search}':")
    for g in groups:
        print(f"  {os.path.basename(g)}")

    fpr_grid    = np.linspace(0, 1, args.n_points)
    recall_grid = np.linspace(0, 1, args.n_points)

    results = []
    for group_dir in groups:
        r = process_group(
            group_dir,
            cache,
            device,
            fpr_grid,
            recall_grid,
            args.batch_size,
        )
        if r is not None:
            results.append(r)

    if not results:
        print("No results to plot.")
        sys.exit(1)

    title = args.title or args.search

    fig_roc, fig_pr = make_figures(results, fpr_grid, recall_grid, title)
    fig_box          = make_boxplot(results, title)

    output_dir = args.output_dir or os.path.join(args.models_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)

    safe     = re.sub(r'[\\/:*?"<>|]', '_', args.search)
    roc_path = os.path.join(output_dir, f'roc_{safe}.png')
    pr_path  = os.path.join(output_dir, f'pr_{safe}.png')
    box_path = os.path.join(output_dir, f'boxplot_{safe}.png')

    fig_roc.savefig(roc_path, dpi=300, bbox_inches='tight')
    fig_pr.savefig(pr_path,  dpi=300, bbox_inches='tight')
    fig_box.savefig(box_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved:\n  {roc_path}\n  {pr_path}\n  {box_path}")
    plt.close(fig_roc)
    plt.close(fig_pr)
    plt.close(fig_box)


if __name__ == '__main__':
    main()
