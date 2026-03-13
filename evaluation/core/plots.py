"""
Shared plotting utilities for variant pathogenicity evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

from .metrics import try_numeric, flip_if_inverse

COLORS = {
    "ours":  "#d62728",
    "revel": "#ff7f0e",
    "alpha": "#2ca02c",
    "other": "#4878d0",
}


def col_color(col: str) -> str:
    c = col.lower()
    if c == "ours":              return COLORS["ours"]
    if "revel" in c:             return COLORS["revel"]
    if "alphamissense" in c:     return COLORS["alpha"]
    return COLORS["other"]


def _plot_curve(ax, df, col, labels, curve_fn, color, lw=1.0, label_suffix=""):
    s = try_numeric(df[col])
    mask = s.notna()
    if mask.sum() < 10:
        return
    s_f, _ = flip_if_inverse(s[mask], labels[mask])
    try:
        x, y, _ = curve_fn(labels[mask], s_f)
        auc = roc_auc_score(labels[mask], s_f)
        ax.plot(x, y, color=color, lw=lw, alpha=0.85,
                label=f"{col}{label_suffix} ({auc:.3f})")
    except Exception:
        pass


def plot_roc_curves(df: pd.DataFrame, top_cols: list, our_col: str,
                    label_col: str, out_path: Path, n: int = 20,
                    subtitle: str = "") -> None:
    labels = df[label_col]
    fig, ax = plt.subplots(figsize=(10, 8))

    s = try_numeric(df[our_col])
    mask = s.notna()
    s_f, _ = flip_if_inverse(s[mask], labels[mask])
    fpr, tpr, _ = roc_curve(labels[mask], s_f)
    auc = roc_auc_score(labels[mask], s_f)
    ax.plot(fpr, tpr, color=COLORS["ours"], lw=2.5, zorder=10,
            label=f"Ours ({auc:.3f})")

    for col in top_cols[:n]:
        _plot_curve(ax, df, col, labels, roc_curve, col_color(col),
                    lw=2.0 if col.lower() in ("revel_score", "alphamissense_score") else 1.0)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    title = "ROC Curves — Top methods (shared subset)"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_pr_curves(df: pd.DataFrame, top_cols: list, our_col: str,
                   label_col: str, out_path: Path, n: int = 20,
                   subtitle: str = "") -> None:
    labels = df[label_col]
    fig, ax = plt.subplots(figsize=(10, 8))

    s = try_numeric(df[our_col])
    mask = s.notna()
    s_f, _ = flip_if_inverse(s[mask], labels[mask])
    rec, prec, _ = precision_recall_curve(labels[mask], s_f)
    from sklearn.metrics import average_precision_score
    auc = average_precision_score(labels[mask], s_f)
    ax.plot(rec, prec, color=COLORS["ours"], lw=2.5, zorder=10,
            label=f"Ours ({auc:.3f})")

    for col in top_cols[:n]:
        _plot_curve(ax, df, col, labels, precision_recall_curve, col_color(col),
                    lw=2.0 if col.lower() in ("revel_score", "alphamissense_score") else 1.0)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    title = "Precision-Recall Curves — Top methods"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_auroc_barplot(metrics_df: pd.DataFrame, our_auroc: float,
                       anchor_cols: list, out_path: Path, n: int = 20) -> None:
    top = (metrics_df[metrics_df["column"] != "ours"]
           .dropna(subset=["auroc"]).nlargest(n, "auroc"))
    our_row = metrics_df[metrics_df["column"] == "ours"]

    plot_df = pd.concat([our_row, top]).drop_duplicates("column").reset_index(drop=True)
    colors  = [col_color(c) for c in plot_df["column"]]

    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.4)))
    bars = ax.barh(plot_df["column"], plot_df["auroc"], color=colors, edgecolor="white", lw=0.4)
    ax.axvline(our_auroc, color=COLORS["ours"], lw=1.5, ls="--", alpha=0.7)
    ax.axvline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("AUROC", fontsize=11)
    ax.set_title(f"AUROC Comparison — Top {n} + Ours", fontsize=12)
    ax.invert_yaxis()
    for bar, val in zip(bars, plot_df["auroc"]):
        if not np.isnan(val):
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_metrics_heatmap(metrics_df: pd.DataFrame, out_path: Path, n: int = 20) -> None:
    metric_cols = ["auroc", "prauc", "pauroc_fpr10", "mcc", "f1", "balanced_acc"]
    our_row = metrics_df[metrics_df["column"] == "ours"]
    top     = (metrics_df[metrics_df["column"] != "ours"]
               .dropna(subset=["auroc"]).nlargest(n, "auroc"))
    top     = pd.concat([our_row, top]).drop_duplicates("column").reset_index(drop=True)

    data    = top[metric_cols].values.astype(float)
    fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.4)))
    im      = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["column"], fontsize=8)
    ax.get_yticklabels()[0].set_color("red")
    for i in range(len(top)):
        for j in range(len(metric_cols)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    plt.colorbar(im, ax=ax, shrink=0.6)
    ax.set_title(f"Metrics Heatmap — Top {n} + Ours", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_auroc_scatter(metrics_df: pd.DataFrame, our_auroc: float, out_path: Path) -> None:
    df = metrics_df[metrics_df["column"] != "ours"].dropna(subset=["auroc"])
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(df["n_variants"], df["auroc"],
                    c=df["auroc"], cmap="RdYlGn", vmin=0.4, vmax=1.0,
                    alpha=0.7, edgecolors="gray", lw=0.4, s=60)
    for _, row in df[df["column"].str.lower()
                     .isin(["revel_score", "alphamissense_score"])].iterrows():
        ax.scatter(row["n_variants"], row["auroc"],
                   color=col_color(row["column"]), s=150, zorder=5,
                   edgecolors="black", lw=1)
        ax.annotate(row["column"], (row["n_variants"], row["auroc"]),
                    fontsize=8, xytext=(5, 3), textcoords="offset points")
    ax.axhline(our_auroc, color=COLORS["ours"], ls="--", lw=1.5,
               label=f"Ours ({our_auroc:.3f})")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    plt.colorbar(sc, ax=ax, label="AUROC")
    ax.set_xlabel("N variants with score", fontsize=11)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_title("AUROC vs N variants", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_comparison_across_strata(comparison_df: pd.DataFrame,
                                  out_path: Path, metric: str = "auroc") -> None:
    """
    Bar chart comparing one metric across multiple strata (filter/stratify conditions)
    for each method. comparison_df must have columns: [stratum, column, <metric>].
    """
    strata  = comparison_df["stratum"].unique().tolist()
    methods = comparison_df["column"].unique().tolist()

    x      = np.arange(len(methods))
    width  = 0.8 / len(strata)

    fig, ax = plt.subplots(figsize=(max(10, len(methods) * 0.5), 6))
    for i, stratum in enumerate(strata):
        sub  = comparison_df[comparison_df["stratum"] == stratum].set_index("column")
        vals = [sub.loc[m, metric] if m in sub.index else np.nan for m in methods]
        ax.bar(x + i * width, vals, width, label=stratum, alpha=0.85)

    ax.set_xticks(x + width * (len(strata) - 1) / 2)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric.upper(), fontsize=11)
    ax.set_title(f"{metric.upper()} by Stratum", fontsize=12)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_af_distribution(df: pd.DataFrame, af_cols: list, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(af_cols), figsize=(6 * len(af_cols), 4))
    if len(af_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, af_cols):
        if col not in df.columns:
            continue
        vals = try_numeric(df[col]).dropna()
        vals = vals[vals > 0]
        if len(vals) == 0:
            continue
        ax.hist(np.log10(vals + 1e-10), bins=50, color="#4878d0", alpha=0.7)
        ax.set_xlabel("log10(AF)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"{col}\nn={len(vals):,}", fontsize=10)
    fig.suptitle("Allele Frequency Distribution (rare subset)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")
