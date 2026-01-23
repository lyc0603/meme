from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from environ.constants import FIGURE_PATH, PROCESSED_DATA_CS_PATH

INPUT_JSON = Path(PROCESSED_DATA_CS_PATH) / "res.json"
OUT_PDF = Path(FIGURE_PATH) / "avg_return_test_bar.pdf"

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


def main() -> None:
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        res = json.load(f)

    any_model_key = next(iter(res.keys()))
    ret_test_all = _to_np(res[any_model_key]["ret_test"])
    baseline_test_mean = float(np.nanmean(ret_test_all))

    labels = ["All test"]
    test_means = [baseline_test_mean]
    diagnostics = {}

    for model_name in MODEL_ORDER:
        if model_name not in res:
            continue

        block = res[model_name]
        proba_val = _to_np(block["val_proba"])
        ret_val = _to_np(block["ret_val"])
        proba_test = _to_np(block["test_proba"])
        ret_test = _to_np(block["ret_test"])

        t_star, val_mean, val_cov = choose_threshold_max_avg_return(
            proba_val=proba_val,
            ret_val=ret_val,
            n_thresh=N_THRESH,
            min_coverage=MIN_VAL_COVERAGE,
            fallback=FALLBACK_THRESHOLD,
        )

        mask_test = proba_test >= t_star
        test_cov = float(mask_test.mean()) if len(mask_test) else 0.0
        test_mean = (
            float(np.nanmean(ret_test[mask_test])) if mask_test.any() else float("nan")
        )

        labels.append(model_name)
        test_means.append(test_mean)

        diagnostics[model_name] = {
            "threshold": t_star,
            "val_mean_selected": val_mean,
            "val_coverage": val_cov,
            "test_mean_selected": test_mean,
            "test_coverage": test_cov,
        }

    safe_mkdir(OUT_PDF)

    x = np.arange(len(labels))
    bar_colors = [COLORS.get(lbl, "black") for lbl in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, test_means, color=bar_colors, width=0.9)

    pretty_labels = [
        "MAS\n(ZeroShot)" if lbl == "MAS (Zero Shot)" else lbl for lbl in labels
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(pretty_labels, fontsize=14)
    ax.set_ylabel("Average Return (Test Set)", fontsize=14)

    for i, v in enumerate(test_means):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=14)

    # Remove top/right frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # keep ticks only on left/bottom
    ax.tick_params(top=False, right=False, labelsize=14)
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Baseline (All test) avg return: {baseline_test_mean:.6f}\n")
    for m, d in diagnostics.items():
        print(
            f"{m}: threshold={d['threshold']:.3f} | "
            f"VAL mean={d['val_mean_selected']:.6f}, cov={d['val_coverage']:.2%} | "
            f"TEST mean={d['test_mean_selected']:.6f}, cov={d['test_coverage']:.2%}"
        )


if __name__ == "__main__":
    main()
