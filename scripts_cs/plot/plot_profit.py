from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from environ.constants import FIGURE_PATH, PROCESSED_DATA_CS_PATH

INPUT_JSON = Path(PROCESSED_DATA_CS_PATH) / "res.json"
OUT_PDF = Path(FIGURE_PATH) / "avg_return_test.pdf"

N_THRESH = 201
MIN_VAL_COVERAGE = 0.05
FALLBACK_THRESHOLD = 0.50

MODEL_ORDER = ["Lasso", "NN", "XGBoost", "MAS (Zero Shot)", "MAS"]

COLORS = {
    "XGBoost": "crimson",
    "NN": "royalblue",
    "Lasso": "dimgray",
    "MAS": "darkorange",
    "MAS (Zero Shot)": "seagreen",
    "All test": "black",
}

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
    }
)


def _to_np(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def choose_threshold_max_avg_return(
    proba_val: np.ndarray,
    ret_val: np.ndarray,
    n_thresh: int = 201,
    min_coverage: float = 0.05,
    fallback: float = 0.50,
) -> tuple[float, float, float]:
    thresholds = np.linspace(0.0, 1.0, n_thresh)

    best_t = None
    best_mean = -np.inf
    best_cov = 0.0

    n = len(proba_val)
    if n == 0:
        return fallback, float("nan"), 0.0

    for t in thresholds:
        mask = proba_val >= t
        cov = mask.mean()
        if cov < min_coverage:
            continue

        mean_ret = float(np.nanmean(ret_val[mask])) if mask.any() else -np.inf
        if mean_ret > best_mean:
            best_mean = mean_ret
            best_t = float(t)
            best_cov = float(cov)

    if best_t is None:
        return float(fallback), float("nan"), 0.0

    return best_t, float(best_mean), float(best_cov)


def safe_mkdir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _mean_se(x: np.ndarray) -> tuple[float, float, int]:
    """Return (mean, standard_error, n_effective) ignoring NaNs."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(x))
    # sample std with ddof=1 when possible
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 0 else float("nan")
    return mean, float(se), n


def main() -> None:
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        res = json.load(f)

    labels: list[str] = []
    test_means: list[float] = []
    test_ses: list[float] = []
    copy_test_means: list[float] = []
    copy_test_ses: list[float] = []
    diagnostics = {}

    for model_name in MODEL_ORDER:
        if model_name not in res:
            continue

        block = res[model_name]
        proba_val = _to_np(block["val_proba"])
        ret_val = _to_np(block["ret_val"])
        proba_test = _to_np(block["test_proba"])
        ret_test = _to_np(block["ret_test"])
        copy_trading_ret_test = _to_np(block["copy_trading_ret_test"])

        t_star, val_mean, val_cov = choose_threshold_max_avg_return(
            proba_val=proba_val,
            ret_val=ret_val,
            n_thresh=N_THRESH,
            min_coverage=MIN_VAL_COVERAGE,
            fallback=FALLBACK_THRESHOLD,
        )

        mask_test = proba_test >= t_star
        test_cov = float(mask_test.mean()) if len(mask_test) else 0.0

        sel_ret = ret_test[mask_test] if mask_test.any() else np.array([], dtype=float)
        sel_copy = (
            copy_trading_ret_test[mask_test]
            if mask_test.any()
            else np.array([], dtype=float)
        )

        test_mean, test_se, n_test = _mean_se(sel_ret)
        copy_mean, copy_se, n_copy = _mean_se(sel_copy)

        labels.append(model_name)
        test_means.append(test_mean)
        test_ses.append(test_se)
        copy_test_means.append(copy_mean)
        copy_test_ses.append(copy_se)

        diagnostics[model_name] = {
            "threshold": t_star,
            "val_mean_selected": val_mean,
            "val_coverage": val_cov,
            "test_mean_selected": test_mean,
            "test_se_selected": test_se,
            "copy_test_mean_selected": copy_mean,
            "copy_test_se_selected": copy_se,
            "test_coverage": test_cov,
            "n_test_selected": n_test,
        }

    safe_mkdir(OUT_PDF)

    #
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    colors = [COLORS.get(lbl, "black") for lbl in labels]

    # --- bars (NO error bars) ---
    bars1 = ax.bar(
        x - width / 2,
        test_means,
        width=width,
        color="#B2B2FF",
        label="Smart Money Return",
    )
    bars2 = ax.bar(
        x + width / 2,
        copy_test_means,
        width=width,
        color="#81A1C1",
        linewidth=0.6,
        label="Copier Return",
    )

    # --- per-model connecting lines ---
    for i in range(len(x)):
        ax.plot(
            [x[i] - width / 2, x[i] + width / 2],
            [test_means[i], copy_test_means[i]],
            color="#2E3440",
            linewidth=1.0,
            linestyle="--",
            marker="s",
            markersize=6,
            zorder=5,
            label="Imitation Penalty" if i == 0 else None,
        )

    pretty_labels = [
        "MAS\n(Zero Shot)" if lbl == "MAS (Zero Shot)" else lbl for lbl in labels
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(pretty_labels, fontsize=14)
    ax.set_ylabel("Average Return (Test Set)", fontsize=14)

    ax.axhline(0.0, linewidth=1.0)

    # dynamic y-limits including error bars
    all_vals = np.array(test_means + copy_test_means, dtype=float)
    all_errs = np.array(test_ses + copy_test_ses, dtype=float)
    finite = np.isfinite(all_vals)
    if finite.any():
        vmin = float(
            np.min(all_vals[finite] - np.nan_to_num(all_errs[finite], nan=0.0))
        )
        vmax = float(
            np.max(all_vals[finite] + np.nan_to_num(all_errs[finite], nan=0.0))
        )
        pad = 0.10 * (vmax - vmin + 1e-12)
        ax.set_ylim(vmin - pad, vmax + pad)

    def annotate(bars):
        y_min, y_max = ax.get_ylim()
        y_offset = 0.01 * (y_max - y_min)  # 2% of y-range

        for b in bars:
            v = b.get_height()
            if not np.isfinite(v):
                continue

            x0 = b.get_x() + b.get_width() / 2

            if v >= 0:
                y = v + y_offset
                va = "bottom"
            else:
                y = v - y_offset
                va = "top"

            ax.text(
                x0,
                y,
                f"{v:.3f}",
                ha="center",
                va=va,
                fontsize=12,
            )

    annotate(bars1)
    annotate(bars2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False, labelsize=14)
    handles, labels_ = ax.get_legend_handles_labels()
    order = [i for i, l in enumerate(labels_) if l != "Imitation Penalty"] + [
        i for i, l in enumerate(labels_) if l == "Imitation Penalty"
    ]

    ax.legend(
        [handles[i] for i in order],
        [labels_[i] for i in order],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=False,
        fontsize=12,
        ncol=1,
        borderaxespad=0.0,
        handlelength=2.2,
    )

    fig.tight_layout()

    plt.show()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    for m, d in diagnostics.items():
        print(
            f"{m}: threshold={d['threshold']:.3f} | "
            f"VAL mean={d['val_mean_selected']:.6f}, cov={d['val_coverage']:.2%} | "
            f"TEST mean={d['test_mean_selected']:.6f} (SE={d['test_se_selected']:.6f}, n={d['n_test_selected']}) | "
            f"COPY mean={d['copy_test_mean_selected']:.6f} (SE={d['copy_test_se_selected']:.6f}) | "
            f"cov={d['test_coverage']:.2%}"
        )


if __name__ == "__main__":
    main()
