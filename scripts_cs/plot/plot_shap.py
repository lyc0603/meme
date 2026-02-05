"""
Script to plot SHAP summary plots with matching importance bars.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.patches import Patch

from environ.constants import FIGURE_PATH, PROCESSED_DATA_CS_PATH

# -----------------------------
# Feature name mapping (canonical / paper order)
# -----------------------------
FEATURE_MAP: List[Tuple[str, str, str]] = [
    ("first_txn_amount", "Trader Purchase Amount", r"$A$"),
    ("t_stat", r"$t$-stat", r"$t$"),
    ("ten_to_six_ret", "Return (6th--10th)", r"$\mathbf{\bar{R}_{\text{6th-10th}}}$"),
    ("five_to_one_ret", "Return (1st--5th)", r"$\mathbf{\bar{R}_{\text{1st-5th}}}$"),
    ("std_ret", "Return Standard Deviation", r"$std$"),
    ("time_since_launch", "Time Since Launch", r"$T_{\mathrm{Launch}}$"),
    (
        "fifteen_to_eleven_ret",
        "Return (11th--15th)",
        r"$\mathbf{\bar{R}_{\text{11th-15th}}}$",
    ),
    ("last_ret", "Return (1st)", r"$\mathbf{\bar{R}_{\text{1st}}}$"),
    ("average_ret", "Return (all)", r"$\mathbf{\bar{R}_{\text{all}}}$"),
    ("first_txn_quantity", "Trader Purchase Quantity", r"$Q$"),
    ("time_since_first_trade", "Time Since First Trade", r"$T_{\mathrm{First}}$"),
    ("num_trades", "Number of Trades", r"$\#Trade$"),
    ("first_txn_price", "Trader Purchase Price", r"$P$"),
    ("wash_trading_bot", "Bump Bot", r"$\text{Bump Bot}$"),
    ("sniper_bot", "Sniper Bot", r"$\text{Sniper Bot}$"),
    ("launch_bundle", "Bundle Bot", r"$\text{Bundle Bot}$"),
    ("time_since_last_trade", "Time Since Last Trade", r"$T_{\mathrm{Last}}$"),
    ("comment_bot", "Comment Bot", r"$\text{Comment Bot}$"),
]


def _build_display_name_map(feature_names: list[str]) -> dict[str, str]:
    """raw_feature -> human display label"""
    raw_to_disp = {raw: disp for (raw, disp, _) in FEATURE_MAP}
    return {f: raw_to_disp.get(f, f) for f in feature_names}


# -----------------------------
# Reordering to canonical order
# -----------------------------
def reorder_to_feature_map(
    feature_names_raw: list[str],
    X: np.ndarray,
    shap_vals: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Reorder columns/features of X and shap_vals to match FEATURE_MAP order.

    Returns:
      feature_names_new, X_new, shap_vals_new
    """
    name_to_idx = {f: i for i, f in enumerate(feature_names_raw)}

    # canonical order, keep only those present
    order = [name_to_idx[raw] for (raw, _, _) in FEATURE_MAP if raw in name_to_idx]

    # reorder X
    X2 = np.asarray(X)[:, order]

    # reorder SHAP
    sv = np.asarray(shap_vals)
    if sv.ndim == 2:  # (n, d)
        sv2 = sv[:, order]
    elif sv.ndim == 3:  # (n, d, k)
        sv2 = sv[:, order, :]
    else:
        raise ValueError(f"Unexpected shap_vals shape={sv.shape} (ndim={sv.ndim})")

    feat2 = [feature_names_raw[i] for i in order]
    return feat2, X2, sv2


# -----------------------------
# SHAP -> importance (consistent with beeswarm)
# -----------------------------
def mean_abs_shap_per_feature(shap_vals: np.ndarray) -> np.ndarray:
    """
    Compute per-feature mean(|SHAP|) from the SAME SHAP array used for summary_plot.

    Supports:
      - (n, d)        single output
      - (n, d, k)     multi-class / multi-output
    Returns:
      - (d,)
    """
    sv = np.asarray(shap_vals)
    if sv.ndim == 2:
        return np.mean(np.abs(sv), axis=0)
    if sv.ndim == 3:
        return np.mean(np.abs(sv), axis=(0, 2))
    raise ValueError(f"Unexpected shap_vals shape={sv.shape} (ndim={sv.ndim})")


def normalize_importance_sum(imp: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize to sum to 1.0 (table-friendly)."""
    s = float(np.sum(imp))
    if s < eps:
        return np.zeros_like(imp, dtype=float)
    return imp / s


# -----------------------------
# Plot helpers
# -----------------------------
def _add_bars_from_importance(
    ax: plt.Axes,
    ref_ax: plt.Axes,
    importance_map: Dict[str, float],  # raw_feature -> normalized importance
    display_map: Dict[str, str],  # raw_feature -> display label
    bar_span_frac: float = 0.35,
    alpha: float = 0.30,
    fmt: str = "{:.4f}",
    text_frac_pad: float = 0.012,
    fontsize: int = 10,
) -> None:
    """
    Add left-anchored horizontal bars aligned with beeswarm y-ticks.
    Assumes ALL panels share the same y tick ordering (we enforce sort=False + reordering).
    """
    # display label -> y position from reference axis
    label_to_y: Dict[str, float] = {}
    for tl, y in zip(ref_ax.get_yticklabels(), ref_ax.get_yticks()):
        lab = tl.get_text().strip()
        if lab:
            label_to_y[lab] = y

    pairs: list[tuple[float, float, str]] = []
    for raw_feature, _, _ in FEATURE_MAP:
        if raw_feature not in importance_map:
            continue
        disp_label = display_map.get(raw_feature, raw_feature)
        if disp_label not in label_to_y:
            continue
        pairs.append(
            (label_to_y[disp_label], float(importance_map[raw_feature]), disp_label)
        )

    if not pairs:
        return

    y_positions = [p[0] for p in pairs]
    vals = np.array([p[1] for p in pairs], dtype=float)
    max_val = float(vals.max()) if float(vals.max()) > 0 else 1.0

    xmin, xmax = ax.get_xlim()
    span = xmax - xmin
    bar_span = span * bar_span_frac
    widths = (vals / max_val) * bar_span

    ax.barh(
        y_positions,
        widths,
        left=xmin,
        height=0.82,
        color="#9b59b6",
        alpha=alpha,
        zorder=0,
    )


def plot_beeswarm_panels_with_bars(
    shap_vals_dict: Dict[str, np.ndarray],
    X_dict: Dict[str, np.ndarray],
    feature_names_raw: list[str],
    outpath: Path,
) -> Dict[str, Dict[str, float]]:
    """
    Plot beeswarm panels with matching bars.
    Returns:
      importance_norm: model -> raw_feature -> normalized importance
    """
    model_names = list(shap_vals_dict.keys())

    # Reorder every model into FEATURE_MAP order and compute importance on that same order
    reordered: Dict[str, dict] = {}
    importance_norm: Dict[str, Dict[str, float]] = {}

    for m in model_names:
        feat2, X2, sv2 = reorder_to_feature_map(
            feature_names_raw, X_dict[m], shap_vals_dict[m]
        )
        display_map2 = _build_display_name_map(feat2)
        feature_names_display2 = [display_map2[f] for f in feat2]

        imp_vec = mean_abs_shap_per_feature(sv2)
        imp_norm = normalize_importance_sum(imp_vec)
        importance_norm[m] = {f: float(v) for f, v in zip(feat2, imp_norm)}

        reordered[m] = {
            "feat_raw": feat2,
            "feat_disp": feature_names_display2,
            "display_map": display_map2,
            "X": X2,
            "sv": sv2,
        }

    # Use the reordered feature list for plotting
    fig, axes = plt.subplots(
        1, len(model_names), figsize=(5.8 * len(model_names), 7.8), sharey=True
    )
    if len(model_names) == 1:
        axes = [axes]  # type: ignore[list-item]

    ref_ax: plt.Axes | None = None

    for i, m in enumerate(model_names):
        ax = axes[i]
        plt.sca(ax)

        shap.summary_plot(
            reordered[m]["sv"],
            features=reordered[m]["X"],
            feature_names=reordered[m]["feat_disp"],
            plot_type="dot",
            max_display=len(reordered[m]["feat_disp"]),
            show=False,
            color_bar=(i == len(model_names) - 1),
            sort=False,  # <<< CRITICAL: keep identical row order across panels
        )

        if i == 0:
            ref_ax = ax
        else:
            ax.tick_params(axis="y", labelleft=False)

        assert ref_ax is not None
        _add_bars_from_importance(
            ax=ax,
            ref_ax=ref_ax,
            importance_map=importance_norm[m],
            display_map=reordered[m]["display_map"],
        )

        ax.set_title(m)
        ax.set_xlabel("SHAP value")
        ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.35)

    legend_patch = Patch(facecolor="#9b59b6", alpha=0.30, label=r"Mean($|$SHAP$|$)")
    fig.legend(
        handles=[legend_patch],
        loc="lower left",
        bbox_to_anchor=(0.08, 0.001),
        frameon=False,
        fontsize=13,
        handlelength=1.2,
        handleheight=1.2,
    )

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.show()

    return importance_norm


if __name__ == "__main__":
    base = Path(PROCESSED_DATA_CS_PATH)
    out_fig = Path(FIGURE_PATH) / "shap_summary_combined_with_bars.pdf"

    data = np.load(base / "shap_plot_data.npz", allow_pickle=True)
    feature_names_raw: list[str] = data["feature_names"].tolist()
    model_names: list[str] = data["model_names"].tolist()

    shap_vals_dict = {m: data[f"shap__{m}"] for m in model_names}
    X_val_dict = {m: data[f"Xval__{m}"] for m in model_names}

    # Plot + get importance (normalized) computed from the SAME reordered SHAP arrays used in the plot
    importance_norm = plot_beeswarm_panels_with_bars(
        shap_vals_dict=shap_vals_dict,
        X_dict=X_val_dict,
        feature_names_raw=feature_names_raw,
        outpath=out_fig,
    )
