"""Script to plot ROC curves and precision/F1 vs threshold for ML + MAS models (two figures total)."""

import json

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import precision_recall_curve, roc_curve

from environ.constants import FIGURE_PATH, PROCESSED_DATA_CS_PATH

plt.rcParams.update(
    {
        "font.size": 22,
        "font.weight": "bold",
        "axes.grid": True,
    }
)

# Define model groups
ML_MODELS = ["XGBoost", "NN", "Lasso"]
MAS_MODELS = ["MAS", "MAS (Zero Shot)"]

ALL_MODELS = ML_MODELS + MAS_MODELS

colors = {
    "XGBoost": "crimson",
    "NN": "royalblue",
    "Lasso": "dimgray",
    "MAS": "darkorange",
    "MAS (Zero Shot)": "seagreen",
}

with open(PROCESSED_DATA_CS_PATH / "res.json", "r", encoding="utf-8") as f:
    res_dict = json.load(f)


def collect(model_names: list[str]) -> tuple[dict, dict, dict]:
    """Collect proba / auc / y for a given list of models."""
    proba_dict, auc_dict, y_dict = {}, {}, {}
    for name in model_names:
        info = res_dict[name]
        proba_dict[name] = info["val_proba"]
        auc_dict[name] = info["val_auc"]
        y_dict[name] = info["y_val"]
    return proba_dict, auc_dict, y_dict


def plot_prec_f1_vs_threshold(proba_dict: dict, y_dict: dict, outpath) -> None:
    """Plot precision and F1 vs threshold curves for multiple models."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, proba in proba_dict.items():
        prec, rec, thr = precision_recall_curve(y_dict[name], proba)

        # Drop the artificial last point (precision=1, recall=0) with no threshold
        prec = prec[:-1]
        rec = rec[:-1]

        eps = 1e-8
        f1 = 2 * prec * rec / (prec + rec + eps)

        ax.plot(thr, prec, linestyle="--", linewidth=1.8, color=colors[name])
        ax.plot(thr, f1, linestyle="-", linewidth=1.8, color=colors[name])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Threshold", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")

    # Legend: measure
    measure_lines = [
        plt.Line2D([0], [0], linestyle="--", color="black", lw=1.8, label="Precision"),
        plt.Line2D(
            [0], [0], linestyle="-", color="black", lw=1.8, label=r"$\mathbf{F_1}$"
        ),
    ]
    leg1 = ax.legend(
        handles=measure_lines,
        title="Measure",
        loc="upper left",
        frameon=False,
        fontsize=18,
        title_fontsize=18,
    )
    ax.add_artist(leg1)

    # Legend: models
    model_patches = [
        Patch(facecolor=colors[name], edgecolor="none", label=name)
        for name in proba_dict.keys()
    ]
    leg2 = ax.legend(
        handles=model_patches,
        title="Model",
        loc="lower left",
        frameon=False,
        handlelength=1,
        handleheight=1,
        fontsize=18,
        title_fontsize=18,
    )
    ax.add_artist(leg2)

    ax.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()


def plot_roc_curves(proba_dict: dict, y_dict: dict, auc_dict: dict, outpath) -> None:
    """Plot ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_dict[name], proba)
        ax.plot(fpr, tpr, lw=1.8, color=colors[name], label=name)

    ax.plot([0, 1], [0, 1], linestyle=":", color="black", lw=1.2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate", fontweight="bold")
    ax.set_ylabel("True positive rate", fontweight="bold")

    model_patches = [
        Patch(
            facecolor=colors[name],
            edgecolor="none",
            label=f"{name} (AUC={auc_dict[name]:.4f})",
        )
        for name in proba_dict.keys()
    ]
    leg = ax.legend(
        handles=model_patches,
        title="Model",
        loc="lower right",
        frameon=False,
        handlelength=1,
        handleheight=1,
        fontsize=17,
        title_fontsize=17,
    )
    ax.add_artist(leg)

    ax.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()


# ---- Two plots total: all models together ----
proba_all, auc_all, y_all = collect(ALL_MODELS)

plot_prec_f1_vs_threshold(proba_all, y_all, FIGURE_PATH / "prec_f1_vs_threshold.pdf")
plot_roc_curves(proba_all, y_all, auc_all, FIGURE_PATH / "roc_curves.pdf")
