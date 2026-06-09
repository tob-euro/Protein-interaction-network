"""Print thesis result summaries, statistical tests, and selected figures."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import t as student_t
from scipy.stats import wilcoxon
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.data_scripts.gene_isoform_pairs import build_gene_isoform_graph
from src.data_scripts.isoform_pairs import load_esmc_features
from src.data_scripts.split_cache import SplitCache
from src.model_classes.ldm import LatentDistanceModel
from src.model_classes.mm_ldm import MultimodalLDM
from src.training.runner import _build_split_key, _load_all_splits


SEEDS = list(range(42, 52))
DIMS = [0, 2, 8, 32]

TRANS_GROUPS = {d: f"trans_ldm_seeds=10_latent_dim={d}" for d in DIMS}
II_ONLY_GROUPS = {d: f"ind_mmldm_seeds=10_latent_dim={d}" for d in DIMS}
II_GI_GROUPS = {d: f"ind_mmldm_withGI_seeds=10_latent_dim={d}" for d in DIMS}
FULL_GROUPS = {d: f"ind_mmldm_full_seeds=10_latent_dim={d}" for d in DIMS}

DTU_COLORS = {
    "dtured": "#990000",
    "blue": "#2F3EEA",
    "brightgreen": "#1FD082",
    "navyblue": "#030F4F",
    "yellow": "#F6D04D",
    "orange": "#FC7634",
    "grey": "#DADADA",
    "red": "#E83F48",
    "green": "#008835",
    "purple": "#79238E",
}
DTU_CYCLE = [
    DTU_COLORS["dtured"],
    DTU_COLORS["blue"],
    DTU_COLORS["brightgreen"],
    DTU_COLORS["navyblue"],
    DTU_COLORS["yellow"],
    DTU_COLORS["orange"],
    DTU_COLORS["grey"],
    DTU_COLORS["red"],
    DTU_COLORS["green"],
    DTU_COLORS["purple"],
]
DTU_TEXT = "#111111"


@dataclass(frozen=True)
class Summary:
    mean: float
    ci: float
    n: int


@dataclass(frozen=True)
class TestResult:
    group: str
    comparison: str
    delta: float
    ci_low: float
    ci_high: float
    p: float
    p_holm: float | None = None


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    group: str


DIAGNOSTIC_MODELS = [
    ModelSpec("trans_d32", "Transductive LDM, d=32", TRANS_GROUPS[32]),
    ModelSpec("full_ind_d32", "Full inductive MMLDM, d=32", FULL_GROUPS[32]),
]


def load_config(path: Path = Path("config.yaml")) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def apply_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.prop_cycle": plt.cycler(color=DTU_CYCLE),
        "axes.edgecolor": "#666666",
        "axes.linewidth": 0.8,
        "axes.labelcolor": DTU_TEXT,
        "xtick.color": DTU_TEXT,
        "ytick.color": DTU_TEXT,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "grid.color": DTU_COLORS["grey"],
        "grid.linewidth": 0.5,
        "grid.alpha": 1.0,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axes(ax, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", length=0)


def t_ci(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    return float(student_t.ppf(0.975, len(arr) - 1) * arr.std(ddof=1) / math.sqrt(len(arr)))


def summarize_values(values: Iterable[float]) -> Summary | None:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    return Summary(float(arr.mean()), t_ci(arr), len(arr))


def summarize(df: pd.DataFrame | None, column: str) -> Summary | None:
    if df is None or column not in df or not valid_seeds(df):
        return None
    return summarize_values(df[column].astype(float).to_numpy())


def fmt_summary(summary: Summary | None, decimals: int = 3) -> str:
    if summary is None:
        return "--"
    return f"{summary.mean:.{decimals}f} +/- {summary.ci:.{decimals}f}"


def fmt_num(value: float | None, decimals: int = 3, signed: bool = True) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    if abs(value) < 0.5 * 10 ** (-decimals):
        value = 0.0
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value:.{decimals}f}"


def fmt_ci(low: float | None, high: float | None, decimals: int = 3) -> str:
    if low is None or high is None or not (np.isfinite(low) and np.isfinite(high)):
        return "--"
    if abs(low) < 0.5 * 10 ** (-decimals):
        low = 0.0
    if abs(high) < 0.5 * 10 ** (-decimals):
        high = 0.0
    return f"[{low:.{decimals}f}, {high:.{decimals}f}]"


def fmt_p(value: float | None, decimals: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    if value < 0.001:
        return "<0.001"
    return f"{value:.{decimals}f}"


def print_section(title: str) -> None:
    print(f"\n--- {title} ---")


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    text_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in text_rows)) if text_rows else len(headers[i])
        for i in range(len(headers))
    ]
    header = "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    print(header)
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in text_rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def save_figure(fig, figures_dir: Path, stem: str) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ["png", "pdf"]:
        path = figures_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    return paths


def read_repeated(models_dir: Path, group: str) -> pd.DataFrame | None:
    path = models_dir / group / "repeated_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path).sort_values("seed").reset_index(drop=True)


def valid_seeds(df: pd.DataFrame | None) -> bool:
    if df is None or "seed" not in df:
        return False
    return sorted(df["seed"].astype(int).unique().tolist()) == SEEDS


def load_repeated_groups(models_dir: Path) -> dict[str, dict[int, pd.DataFrame | None]]:
    return {
        "trans": {d: read_repeated(models_dir, group) for d, group in TRANS_GROUPS.items()},
        "full": {d: read_repeated(models_dir, group) for d, group in FULL_GROUPS.items()},
        "ii_only": {d: read_repeated(models_dir, group) for d, group in II_ONLY_GROUPS.items()},
        "ii_gi": {d: read_repeated(models_dir, group) for d, group in II_GI_GROUPS.items()},
    }


def paired_test(
    df_a: pd.DataFrame | None,
    df_b: pd.DataFrame | None,
    column: str,
    group: str,
    comparison: str,
) -> TestResult | None:
    if not (valid_seeds(df_a) and valid_seeds(df_b)):
        return None
    if column not in df_a or column not in df_b:
        return None

    merged = (
        df_a[["seed", column]].rename(columns={column: "a"})
        .merge(df_b[["seed", column]].rename(columns={column: "b"}), on="seed")
        .sort_values("seed")
    )
    if sorted(merged["seed"].astype(int).tolist()) != SEEDS:
        return None
    return difference_test(merged["a"].to_numpy(), merged["b"].to_numpy(), group, comparison)


def difference_test(
    values: Iterable[float],
    baseline: Iterable[float],
    group: str,
    comparison: str,
) -> TestResult:
    diff = np.asarray(list(values), dtype=float) - np.asarray(list(baseline), dtype=float)
    delta = float(diff.mean())
    ci = t_ci(diff)
    if np.allclose(diff, 0):
        p_value = 1.0
    else:
        p_value = float(wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue)
    return TestResult(group, comparison, delta, delta - ci, delta + ci, p_value)


def holm(results: list[TestResult | None]) -> list[TestResult | None]:
    indexed = [(i, result.p) for i, result in enumerate(results) if result is not None]
    adjusted: dict[int, float] = {}
    running = 0.0
    m = len(indexed)
    for rank, (idx, p_value) in enumerate(sorted(indexed, key=lambda item: item[1])):
        value = min((m - rank) * p_value, 1.0)
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    out = []
    for i, result in enumerate(results):
        if result is None:
            out.append(None)
        else:
            out.append(TestResult(
                result.group,
                result.comparison,
                result.delta,
                result.ci_low,
                result.ci_high,
                result.p,
                adjusted[i],
            ))
    return out


def print_tests(title: str, results: list[TestResult | None]) -> None:
    rows = []
    for result in results:
        if result is None:
            rows.append(["missing", "--", "--", "--", "--"])
        else:
            rows.append([
                result.comparison,
                fmt_num(result.delta),
                fmt_ci(result.ci_low, result.ci_high),
                fmt_p(result.p),
                fmt_p(result.p_holm),
            ])
    print_section(title)
    print_table(["Comparison", "Delta AP", "95% CI", "p", "p Holm"], rows)


def print_coverage(dfs: dict[str, dict[int, pd.DataFrame | None]]) -> None:
    families = [
        ("Transductive LDM", "trans"),
        ("Full inductive MMLDM", "full"),
        ("II-only ablation", "ii_only"),
        ("II+GI/no-GG ablation", "ii_gi"),
    ]
    rows = []
    for label, key in families:
        statuses = []
        for dim in DIMS:
            statuses.append(f"d={dim}: {'ok' if valid_seeds(dfs[key][dim]) else 'missing'}")
        rows.append([label, ", ".join(statuses)])
    print_section("Coverage")
    print_table(["Family", "Repeated-result CSVs"], rows)


def print_default_summaries(dfs: dict[str, dict[int, pd.DataFrame | None]]) -> None:
    print_section("Transductive LDM")
    rows = []
    for dim in DIMS:
        df = dfs["trans"][dim]
        ap_delta = None
        auc_delta = None
        if dim != 0:
            ap_test = paired_test(df, dfs["trans"][0], "test_ap", "", "")
            auc_test = paired_test(df, dfs["trans"][0], "test_auc", "", "")
            ap_delta = ap_test.delta if ap_test else None
            auc_delta = auc_test.delta if auc_test else None
        rows.append([
            dim,
            fmt_summary(summarize(df, "test_ap")),
            fmt_summary(summarize(df, "test_auc")),
            "--" if dim == 0 else fmt_num(ap_delta),
            "--" if dim == 0 else fmt_num(auc_delta),
        ])
    print_table(["d", "AP", "ROC-AUC", "Delta AP vs d=0", "Delta AUC vs d=0"], rows)

    print_section("Full Inductive MMLDM")
    rows = []
    for dim in DIMS:
        df = dfs["full"][dim]
        rows.append([
            dim,
            fmt_summary(summarize(df, "test_ap")),
            fmt_summary(summarize(df, "test_auc")),
            fmt_summary(summarize(df, "one_unseen_ap")),
            fmt_summary(summarize(df, "both_unseen_ap")),
        ])
    print_table(["d", "Overall AP", "ROC-AUC", "One-unseen AP", "Two-unseen AP"], rows)

    print_section("Modality Ablation")
    rows = []
    variants = [
        ("II only", "ii_only"),
        ("II + GI", "ii_gi"),
        ("II + GI + GG", "full"),
    ]
    for dim in DIMS:
        for label, key in variants:
            df = dfs[key][dim]
            delta = None
            if key != "ii_only":
                test = paired_test(df, dfs["ii_only"][dim], "test_ap", "", "")
                delta = test.delta if test else None
            rows.append([
                dim,
                label,
                fmt_summary(summarize(df, "test_ap")),
                fmt_summary(summarize(df, "test_auc")),
                fmt_summary(summarize(df, "one_unseen_ap")),
                fmt_summary(summarize(df, "both_unseen_ap")),
                "--" if key == "ii_only" else fmt_num(delta),
            ])
    print_table([
        "d",
        "Variant",
        "Overall AP",
        "ROC-AUC",
        "One-unseen AP",
        "Two-unseen AP",
        "Delta AP vs II only",
    ], rows)


def print_default_tests(dfs: dict[str, dict[int, pd.DataFrame | None]]) -> None:
    print_tests("Transductive Dimension Tests", holm([
        paired_test(dfs["trans"][dim], dfs["trans"][0], "test_ap",
                    "trans", f"d={dim} vs d=0")
        for dim in [2, 8, 32]
    ]))

    print_tests("Full Inductive Dimension Tests", holm([
        paired_test(dfs["full"][dim], dfs["full"][0], "test_ap",
                    "full", f"d={dim} vs d=0")
        for dim in [2, 8, 32]
    ]))

    for dim in DIMS:
        print_tests(f"Modality Ablation Tests, d={dim}", holm([
            paired_test(dfs["ii_gi"][dim], dfs["ii_only"][dim], "test_ap",
                        "ablation", "II + GI vs II only"),
            paired_test(dfs["full"][dim], dfs["ii_gi"][dim], "test_ap",
                        "ablation", "II + GI + GG vs II + GI"),
            paired_test(dfs["full"][dim], dfs["ii_only"][dim], "test_ap",
                        "ablation", "II + GI + GG vs II only"),
        ]))

    print_tests("One-Unseen vs Two-Unseen Tests", holm([
        paired_test(
            dfs["full"][dim].rename(columns={"one_unseen_ap": "metric"}),
            dfs["full"][dim].rename(columns={"both_unseen_ap": "metric"}),
            "metric",
            "unseen",
            f"d={dim}: one-unseen vs two-unseen",
        )
        for dim in DIMS
    ]))


def plot_ablation(dfs: dict[str, dict[int, pd.DataFrame | None]], figures_dir: Path) -> list[Path]:
    apply_plot_style()
    rows = []
    for label, key in [("II only", "ii_only"), ("II + GI", "ii_gi"), ("II + GI + GG", "full")]:
        for dim in DIMS:
            summary = summarize(dfs[key][dim], "test_ap")
            if summary is not None:
                rows.append({"variant": label, "dimension": dim, "mean": summary.mean, "ci": summary.ci})
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.3, 3.25))
    colors = {
        "II only": DTU_COLORS["dtured"],
        "II + GI": DTU_COLORS["blue"],
        "II + GI + GG": DTU_COLORS["brightgreen"],
    }
    markers = {"II only": "o", "II + GI": "s", "II + GI + GG": "^"}
    for label in ["II only", "II + GI", "II + GI + GG"]:
        sub = plot_df[plot_df["variant"] == label].sort_values("dimension")
        ax.errorbar(
            sub["dimension"],
            sub["mean"],
            yerr=sub["ci"],
            marker=markers[label],
            markersize=5,
            linewidth=1.6,
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
            color=colors[label],
            label=label,
        )
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Average precision")
    ax.set_xticks(DIMS)
    style_axes(ax, "y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    paths = save_figure(fig, figures_dir, "inductive_ablation_ap_ci")
    plt.close(fig)
    return paths


def protein_order_hash(protein_to_idx: dict[str, int]) -> str:
    ordered = "\n".join(sorted(protein_to_idx, key=protein_to_idx.get))
    return hashlib.sha256(ordered.encode()).hexdigest()


def load_model_cached(
    model_path: Path,
    device: str,
    esmc_cache: dict[tuple[str, str], torch.Tensor],
):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state = dict(checkpoint["model_state_dict"])
    model_type = checkpoint.get("model_type", "ldm")

    if model_type == "multimodal":
        esmc_features = state.pop("esmc_features", None)
        if esmc_features is None:
            cache_key = (
                str(Path(checkpoint["esmc_path"]).resolve()),
                protein_order_hash(checkpoint["protein_to_idx"]),
            )
            if cache_key not in esmc_cache:
                esmc_cache[cache_key] = load_esmc_features(
                    checkpoint["esmc_path"], checkpoint["protein_to_idx"]
                )
            esmc_features = esmc_cache[cache_key]
        model = MultimodalLDM(
            num_proteins=checkpoint["num_proteins"],
            num_genes=checkpoint["num_genes"],
            latent_dim=checkpoint["latent_dim"],
            esmc_features=esmc_features,
            proj_hidden_dim=checkpoint.get("proj_hidden_dim", 42),
        )
        model.load_state_dict(state)
    else:
        model = LatentDistanceModel(
            num_proteins=checkpoint["num_proteins"],
            latent_dim=checkpoint["latent_dim"],
        )
        model.load_state_dict(state)

    return model.to(device).eval(), checkpoint


def get_split_data(cfg: dict, seed: int, cache: SplitCache | None):
    d = cfg["data"]
    mm = cfg.get("multimodal", {})
    model_type = cfg["model"]["type"]
    is_inductive = d["mode"] == "inductive"
    key = _build_split_key(d, mm, seed, model_type)
    split_data = cache.get(key) if cache else None
    if split_data is None:
        split_data = _load_all_splits(d, mm, seed, model_type, is_inductive)
        if cache:
            cache.put(key, split_data)
    return split_data


def infer_isoform_scores(
    model,
    test_data: pd.DataFrame,
    protein_to_idx: dict[str, int],
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx1 = test_data["ensp_1"].map(protein_to_idx).astype(np.int64).to_numpy()
    idx2 = test_data["ensp_2"].map(protein_to_idx).astype(np.int64).to_numpy()
    labels = test_data["interact"].astype(int).to_numpy()
    scores = np.empty(len(test_data), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(test_data), batch_size):
            end = min(start + batch_size, len(test_data))
            p1 = torch.from_numpy(idx1[start:end]).to(device)
            p2 = torch.from_numpy(idx2[start:end]).to(device)
            scores[start:end] = torch.sigmoid(model(p1, p2)).cpu().numpy()
    return scores, labels


def confusion_from_predictions(labels: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel().tolist()
    total = tn + fp + fn + tp
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred, average="binary", zero_division=0
    )
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "n": total,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tn_rate": tn / total if total else 0.0,
        "fp_rate": fp / total if total else 0.0,
        "fn_rate": fn / total if total else 0.0,
        "tp_rate": tp / total if total else 0.0,
    }


def confusion_from_scores(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    return confusion_from_predictions(labels, scores >= threshold)


def print_confusion_summary(title: str, rows: list[dict[str, object]], keys: list[str]) -> None:
    df = pd.DataFrame(rows)
    out = []
    for key_values, sub in df.groupby(keys, sort=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        label = " / ".join(str(value) for value in key_values)
        out.append([
            label,
            fmt_summary(summarize_values(sub["n"]), 0),
            fmt_summary(summarize_values(sub["accuracy"])),
            fmt_summary(summarize_values(sub["precision"])),
            fmt_summary(summarize_values(sub["recall"])),
            fmt_summary(summarize_values(sub["specificity"])),
            fmt_summary(summarize_values(sub["f1"])),
            fmt_summary(summarize_values(100 * sub["tn_rate"]), 1),
            fmt_summary(summarize_values(100 * sub["fp_rate"]), 1),
            fmt_summary(summarize_values(100 * sub["fn_rate"]), 1),
            fmt_summary(summarize_values(100 * sub["tp_rate"]), 1),
        ])
    print_section(title)
    print_table(
        ["Panel", "n", "Accuracy", "Precision", "Recall", "Specificity", "F1", "TN%", "FP%", "FN%", "TP%"],
        out,
    )


def confusion_rate_matrix(df: pd.DataFrame, filters: dict[str, str]) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for key, value in filters.items():
        mask &= df[key].to_numpy() == value
    sub = df[mask]
    if sub.empty:
        raise ValueError(f"No confusion rows for {filters}")
    return np.array([
        [sub["tn_rate"].mean(), sub["fp_rate"].mean()],
        [sub["fn_rate"].mean(), sub["tp_rate"].mean()],
    ])


def mean_panel_metrics(df: pd.DataFrame, filters: dict[str, str]) -> dict[str, float]:
    mask = np.ones(len(df), dtype=bool)
    for key, value in filters.items():
        mask &= df[key].to_numpy() == value
    sub = df[mask]
    return {key: float(sub[key].mean()) for key in ["accuracy", "precision", "recall", "specificity"]}


def draw_confusion_panel(
    ax,
    matrix: np.ndarray,
    title: str,
    cmap: LinearSegmentedColormap,
    norm: PowerNorm,
    show_y_labels: bool,
    metrics: dict[str, float],
) -> None:
    ax.imshow(matrix, cmap=cmap, norm=norm)
    ax.set_title(title, pad=7)
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"] if show_y_labels else ["", ""])
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                f"{labels[i][j]}\n{100 * value:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > 0.35 else DTU_TEXT,
            )
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.5,
        -0.18,
        (
            f"Acc {metrics['accuracy']:.3f}    Prec {metrics['precision']:.3f}\n"
            f"Rec {metrics['recall']:.3f}    Spec {metrics['specificity']:.3f}"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color=DTU_TEXT,
        linespacing=1.5,
    )


def confusion_cmap() -> tuple[LinearSegmentedColormap, PowerNorm]:
    cmap = LinearSegmentedColormap.from_list(
        "dtu_confusion",
        ["#F7F7F7", "#E7E7E7", "#F0CFCF", "#C87878", DTU_COLORS["dtured"]],
    )
    return cmap, PowerNorm(gamma=0.40, vmin=0, vmax=1)


def plot_prediction_diagnostics(rows: list[dict[str, object]], figures_dir: Path) -> list[Path]:
    apply_plot_style()
    df = pd.DataFrame(rows)
    cmap, norm = confusion_cmap()
    paths = []

    trans = "Transductive LDM, d=32"
    fig, ax = plt.subplots(1, 1, figsize=(3.25, 3.55))
    draw_confusion_panel(
        ax,
        confusion_rate_matrix(df, {"model": trans, "subset": "overall"}),
        "Transductive LDM (d=32)",
        cmap,
        norm,
        True,
        mean_panel_metrics(df, {"model": trans, "subset": "overall"}),
    )
    fig.tight_layout()
    paths.extend(save_figure(fig, figures_dir, "confusion_transductive_ldm"))
    plt.close(fig)

    ind = "Full inductive MMLDM, d=32"
    panels = [("overall", "Overall", True), ("one-unseen", "One unseen", False), ("two-unseen", "Both unseen", False)]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.55))
    for ax, (subset, title, show_y) in zip(axes, panels):
        filters = {"model": ind, "subset": subset}
        draw_confusion_panel(
            ax,
            confusion_rate_matrix(df, filters),
            title,
            cmap,
            norm,
            show_y,
            mean_panel_metrics(df, filters),
        )
    fig.tight_layout()
    paths.extend(save_figure(fig, figures_dir, "confusion_inductive_mmldm_subsets"))
    plt.close(fig)
    return paths


def run_prediction_diagnostics(args: argparse.Namespace) -> None:
    cache = SplitCache(str(args.cache_dir))
    esmc_cache: dict[tuple[str, str], torch.Tensor] = {}
    rows: list[dict[str, object]] = []

    print_section("Prediction Diagnostics")
    for spec in DIAGNOSTIC_MODELS:
        print(f"Scoring {spec.group}")
        group_dir = args.models_dir / spec.group
        for seed in SEEDS:
            seed_dir = group_dir / f"seed_{seed}"
            model_path = seed_dir / "model.pt"
            config_path = seed_dir / "config.yaml"
            if not model_path.exists() or not config_path.exists():
                raise FileNotFoundError(f"Missing model/config in {seed_dir}")

            with config_path.open() as handle:
                cfg = yaml.safe_load(handle)
            split_data = get_split_data(cfg, seed, cache)
            test_data = split_data["test_data"].reset_index(drop=True)
            model, checkpoint = load_model_cached(model_path, args.device, esmc_cache)
            scores, labels = infer_isoform_scores(
                model, test_data, checkpoint["protein_to_idx"], args.device, args.batch_size
            )
            rows.append({
                "model": spec.label,
                "subset": "overall",
                "seed": seed,
                **confusion_from_scores(scores, labels),
            })

            if checkpoint.get("split_mode") == "inductive":
                test_proteins = set(split_data["test_proteins"])
                both_mask = (
                    test_data["ensp_1"].isin(test_proteins)
                    & test_data["ensp_2"].isin(test_proteins)
                ).to_numpy()
                for subset, mask in [("two-unseen", both_mask), ("one-unseen", ~both_mask)]:
                    rows.append({
                        "model": spec.label,
                        "subset": subset,
                        "seed": seed,
                        **confusion_from_scores(scores[mask], labels[mask]),
                    })
            del model
            print(f"  seed={seed} done")

    print_confusion_summary("Fixed Threshold Diagnostics", rows, ["model", "subset"])
    paths = plot_prediction_diagnostics(rows, args.figures_dir)
    print("\nSaved figures:")
    for path in paths:
        print(f"  {path}")


def load_string_gene_labels(string_csv: Path, mapping_csv: Path) -> pd.DataFrame:
    string_df = pd.read_csv(string_csv, usecols=["protein1", "protein2", "interact"])
    mapping_df = pd.read_csv(mapping_csv, usecols=["ensp_id", "ensg_id"])
    ensp_to_ensg = dict(zip(mapping_df["ensp_id"], mapping_df["ensg_id"]))
    g1 = string_df["protein1"].map(ensp_to_ensg)
    g2 = string_df["protein2"].map(ensp_to_ensg)
    ok = g1.notna() & g2.notna()
    out = pd.DataFrame({
        "gene_1": g1[ok].astype(str).to_numpy(),
        "gene_2": g2[ok].astype(str).to_numpy(),
        "label": string_df.loc[ok, "interact"].astype(int).to_numpy(),
    })
    lo = np.minimum(out["gene_1"], out["gene_2"])
    hi = np.maximum(out["gene_1"], out["gene_2"])
    out["gene_1"], out["gene_2"] = lo, hi
    return out.groupby(["gene_1", "gene_2"], as_index=False)["label"].max()


def load_gene_to_isoforms(iso_csv: Path) -> dict[str, set[str]]:
    df = pd.read_csv(iso_csv, usecols=["gene_1", "gene_2", "ensp_1", "ensp_2"])
    _, gene_to_isoforms = build_gene_isoform_graph(df)
    return gene_to_isoforms


def build_candidate_pool(string_labels: pd.DataFrame, gene_to_idx: dict[str, int]) -> pd.DataFrame:
    model_genes = set(gene_to_idx)
    in_vocab = string_labels["gene_1"].isin(model_genes) & string_labels["gene_2"].isin(model_genes)
    pool = string_labels[in_vocab].reset_index(drop=True).copy()
    pool["idx1"] = pool["gene_1"].map(gene_to_idx).to_numpy()
    pool["idx2"] = pool["gene_2"].map(gene_to_idx).to_numpy()
    return pool


def build_isoform_pair_index(
    pool: pd.DataFrame,
    gene_to_isoforms: dict[str, set[str]],
    protein_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gid_to_iso: dict[str, np.ndarray] = {}
    genes = pd.unique(pd.concat([pool["gene_1"], pool["gene_2"]]))
    for gene_id in genes:
        isoforms = gene_to_isoforms.get(gene_id, ())
        gid_to_iso[gene_id] = np.fromiter(
            (protein_to_idx[iso] for iso in isoforms if iso in protein_to_idx),
            dtype=np.int32,
        )

    g1 = pool["gene_1"].to_numpy()
    g2 = pool["gene_2"].to_numpy()
    block_sizes = np.empty(len(pool), dtype=np.int64)
    iso_i_parts = []
    iso_j_parts = []
    for idx in range(len(pool)):
        a = gid_to_iso[g1[idx]]
        b = gid_to_iso[g2[idx]]
        block_sizes[idx] = a.size * b.size
        iso_i_parts.append(np.repeat(a, b.size))
        iso_j_parts.append(np.tile(b, a.size))
    if block_sizes.min() == 0:
        raise ValueError("Some candidate gene pair has no scorable isoform pair.")

    iso_i = np.concatenate(iso_i_parts)
    iso_j = np.concatenate(iso_j_parts)
    seg_start = np.empty(len(pool), dtype=np.int64)
    seg_start[0] = 0
    np.cumsum(block_sizes[:-1], out=seg_start[1:])
    return iso_i, iso_j, seg_start, block_sizes


def load_rq3_model(checkpoint: dict, esmc_cache: dict[str, torch.Tensor], device: str) -> MultimodalLDM:
    state = dict(checkpoint["model_state_dict"])
    state.pop("esmc_features", None)
    ordered = sorted(checkpoint["protein_to_idx"], key=checkpoint["protein_to_idx"].get)
    key = f"{checkpoint['esmc_path']}::{hashlib.md5(chr(0).join(ordered).encode()).hexdigest()}"
    if key not in esmc_cache:
        esmc_cache[key] = load_esmc_features(checkpoint["esmc_path"], checkpoint["protein_to_idx"])
    model = MultimodalLDM(
        num_proteins=checkpoint["num_proteins"],
        num_genes=checkpoint["num_genes"],
        latent_dim=checkpoint["latent_dim"],
        esmc_features=esmc_cache[key],
        proj_hidden_dim=checkpoint.get("proj_hidden_dim", 42),
    )
    model.load_state_dict(state)
    return model.to(device).eval()


def isoform_z_and_r(model: MultimodalLDM, device: str):
    with torch.no_grad():
        idx = torch.arange(model.esmc_features.shape[0], device=device)
        r = model._random_effect(idx).cpu().numpy().astype(np.float64)
        alpha = float(model.alpha_iso_iso.detach().cpu())
        if model.latent_dim > 0:
            z = model._isoform_latent(idx).cpu().numpy().astype(np.float64)
            beta = float(F.softplus(model.beta_iso_iso).detach().cpu())
        else:
            z, beta = None, 0.0
    return z, r, alpha, beta


def distance_scores(gene_emb: torch.Tensor, pool: pd.DataFrame) -> np.ndarray:
    idx1 = torch.from_numpy(pool["idx1"].to_numpy(dtype=np.int64))
    idx2 = torch.from_numpy(pool["idx2"].to_numpy(dtype=np.int64))
    u1 = gene_emb[idx1]
    u2 = gene_emb[idx2]
    return -torch.norm(u1 - u2, p=2, dim=1).numpy()


def aggregation_scores(z, r, alpha, beta, index, chunk: int = 2_000_000) -> np.ndarray:
    iso_i, iso_j, seg_start, _ = index
    logit = np.empty(iso_i.size, dtype=np.float64)
    for start in range(0, iso_i.size, chunk):
        end = min(start + chunk, iso_i.size)
        i = iso_i[start:end]
        j = iso_j[start:end]
        val = alpha + r[i] + r[j]
        if z is not None:
            diff = z[i] - z[j]
            val = val - beta * np.sqrt(np.einsum("ij,ij->i", diff, diff))
        logit[start:end] = val
    return np.maximum.reduceat(logit, seg_start)


def top_prevalence_predictions(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n_positive = int(labels.sum())
    pred = np.zeros(len(scores), dtype=bool)
    if n_positive:
        pred[np.argsort(scores)[::-1][:n_positive]] = True
    return pred


def rq3_tests(metrics: pd.DataFrame) -> list[TestResult]:
    results: list[TestResult] = []
    baseline_tests: list[TestResult | None] = []
    for dim in DIMS:
        sub = metrics[metrics["dimension"] == dim].sort_values("seed")
        baseline_tests.append(difference_test(
            sub["aggregation_ap"],
            sub["baseline_ap"],
            "rq3_baseline",
            f"Aggregation vs baseline, d={dim}",
        ))
        if dim != 0:
            baseline_tests.append(difference_test(
                sub["distance_ap"],
                sub["baseline_ap"],
                "rq3_baseline",
                f"Distance vs baseline, d={dim}",
            ))
    results.extend(result for result in holm(baseline_tests) if result is not None)

    readout_tests = []
    for dim in [2, 8, 32]:
        sub = metrics[metrics["dimension"] == dim].sort_values("seed")
        readout_tests.append(difference_test(
            sub["aggregation_ap"],
            sub["distance_ap"],
            "rq3_readout",
            f"Aggregation vs distance, d={dim}",
        ))
    results.extend(result for result in holm(readout_tests) if result is not None)

    dim_tests = []
    base = metrics[metrics["dimension"] == 0].sort_values("seed")
    for dim in [2, 8, 32]:
        sub = metrics[metrics["dimension"] == dim].sort_values("seed")
        dim_tests.append(difference_test(
            sub["aggregation_ap"],
            base["aggregation_ap"],
            "rq3_dimension",
            f"Aggregation d={dim} vs d=0",
        ))
    results.extend(result for result in holm(dim_tests) if result is not None)
    return results


def print_rq3_metrics(metrics: pd.DataFrame) -> None:
    baseline = float(metrics["baseline_ap"].mean())
    rows = [["baseline", f"{baseline:.3f}", f"{baseline:.3f}", "0.500", "0.500"]]
    for dim in DIMS:
        sub = metrics[metrics["dimension"] == dim]
        rows.append([
            f"d={dim}",
            "--" if dim == 0 else fmt_summary(summarize_values(sub["distance_ap"])),
            fmt_summary(summarize_values(sub["aggregation_ap"])),
            "--" if dim == 0 else fmt_summary(summarize_values(sub["distance_auc"])),
            fmt_summary(summarize_values(sub["aggregation_auc"])),
        ])
    print_section("RQ3 STRING Reconstruction")
    print_table(["Dimension", "Distance AP", "Aggregation AP", "Distance AUC", "Aggregation AUC"], rows)


def plot_rq3_confusion(confusion_rows: list[dict[str, object]], figures_dir: Path) -> list[Path]:
    apply_plot_style()
    df = pd.DataFrame(confusion_rows)
    cmap, norm = confusion_cmap()
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.55))
    for ax, readout, title, show_y in [
        (axes[0], "distance", "Distance", True),
        (axes[1], "aggregation", "Aggregation", False),
    ]:
        filters = {"readout": readout}
        draw_confusion_panel(
            ax,
            confusion_rate_matrix(df, filters),
            title,
            cmap,
            norm,
            show_y,
            mean_panel_metrics(df, filters),
        )
    fig.tight_layout()
    paths = save_figure(fig, figures_dir, "confusion_rq3_distance_aggregation_d32_prevalence")
    plt.close(fig)
    return paths


def run_rq3(args: argparse.Namespace, cfg: dict) -> None:
    data_cfg = cfg["data"]
    string_labels = load_string_gene_labels(Path(data_cfg["string_path"]), Path(data_cfg["mapping_path"]))
    gene_to_isoforms = load_gene_to_isoforms(Path(data_cfg["iso_path"]))

    pool = None
    index = None
    labels = None
    esmc_cache: dict[str, torch.Tensor] = {}
    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []

    print_section("RQ3 Scoring")
    for dim in DIMS:
        group = II_GI_GROUPS[dim]
        for seed in SEEDS:
            ckpt_path = args.models_dir / group / f"seed_{seed}" / "model.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(ckpt_path)
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            if pool is None:
                pool = build_candidate_pool(string_labels, checkpoint["gene_to_idx"])
                index = build_isoform_pair_index(pool, gene_to_isoforms, checkpoint["protein_to_idx"])
                labels = pool["label"].astype(int).to_numpy()
                print(
                    f"Candidate pool: {len(pool):,} gene pairs, "
                    f"positive rate = {labels.mean():.3f}"
                )

            model = load_rq3_model(checkpoint, esmc_cache, args.device)
            agg_scores = aggregation_scores(*isoform_z_and_r(model, args.device), index)
            del model

            gene_emb = checkpoint["model_state_dict"].get("gene_embeddings.weight")
            dist_scores = distance_scores(gene_emb, pool) if gene_emb is not None else None
            metric_rows.append({
                "dimension": dim,
                "seed": seed,
                "baseline_ap": float(labels.mean()),
                "distance_ap": (
                    float(average_precision_score(labels, dist_scores))
                    if dist_scores is not None else math.nan
                ),
                "distance_auc": (
                    float(roc_auc_score(labels, dist_scores))
                    if dist_scores is not None else math.nan
                ),
                "aggregation_ap": float(average_precision_score(labels, agg_scores)),
                "aggregation_auc": float(roc_auc_score(labels, agg_scores)),
            })

            if dim == 32:
                for readout, scores in [("aggregation", agg_scores), ("distance", dist_scores)]:
                    if scores is None:
                        continue
                    confusion_rows.append({
                        "dimension": dim,
                        "seed": seed,
                        "readout": readout,
                        **confusion_from_predictions(labels, top_prevalence_predictions(scores, labels)),
                    })
            print(f"  d={dim:<2} seed={seed} done")

    metrics = pd.DataFrame(metric_rows)
    print_rq3_metrics(metrics)
    print_tests("RQ3 Statistical Tests", rq3_tests(metrics))
    print_confusion_summary("RQ3 Top-Prevalence Confusion, d=32", confusion_rows, ["readout"])
    paths = plot_rq3_confusion(confusion_rows, args.figures_dir)
    print("\nSaved figures:")
    for path in paths:
        print(f"  {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--models-dir", default=Path("models"), type=Path)
    parser.add_argument("--figures-dir", default=Path("figures"), type=Path)
    parser.add_argument("--cache-dir", default=Path(".cache/data_splits"), type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", default=4096, type=int)
    parser.add_argument("--rq3", action="store_true", help="Run STRING reconstruction for RQ3.")
    parser.add_argument(
        "--prediction-diagnostics",
        action="store_true",
        help="Run fixed-threshold diagnostics for selected RQ1/RQ2 checkpoints.",
    )
    parser.add_argument("--all", action="store_true", help="Run default summaries, RQ3, and diagnostics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    if args.all:
        args.rq3 = True
        args.prediction_diagnostics = True

    dfs = load_repeated_groups(args.models_dir)
    print_coverage(dfs)
    print_default_summaries(dfs)
    print_default_tests(dfs)
    paths = plot_ablation(dfs, args.figures_dir)
    print("\nSaved figures:")
    for path in paths:
        print(f"  {path}")

    if args.rq3:
        run_rq3(args, cfg)
    if args.prediction_diagnostics:
        run_prediction_diagnostics(args)


if __name__ == "__main__":
    main()
