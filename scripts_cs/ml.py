"""Script to train ML models to classify traders based on their features."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_CS_PATH

SEED = 42

warnings.filterwarnings("ignore", category=RuntimeWarning)

CONTINUOUS_FEATURES = [
    "average_ret",
    "std_ret",
    "last_ret",
    "five_to_one_ret",
    "ten_to_six_ret",
    "fifteen_to_eleven_ret",
    "t_stat",
    "num_trades",
    "time_since_last_trade",
    "time_since_first_trade",
    "first_txn_price",
    "time_since_launch",
]
DUMMY_FEATURES = ["launch_bundle", "sniper_bot", "wash_trading_bot", "comment_bot"]
all_features = CONTINUOUS_FEATURES + DUMMY_FEATURES


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Per-feature winsorization at given lower/upper quantiles.
    Cutoffs are computed from training data and applied to all data.
    """

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X: np.ndarray, y=None) -> "Winsorizer":
        """Fit the transformer on X."""
        X = np.asarray(X, dtype=float)
        # store per-feature quantiles
        self.lower_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X by applying winsorization."""
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)


def grid_search(
    name: str,
    base_estimator: Pipeline,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[Pipeline, dict, float]:
    """Perform grid search over param_grid for the given base_estimator."""

    best_model = None
    best_params = None
    best_auc = -np.inf

    for params in tqdm(ParameterGrid(param_grid), desc=f"Grid search for {name}"):
        model = clone(base_estimator)
        model.set_params(**params)
        model.fit(X_train, y_train)

        val_proba = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model
            best_params = params

    return best_model, best_params, best_auc


# Preprocessor
pre = ColumnTransformer(
    transformers=[
        (
            "cont",
            Pipeline(
                [
                    ("winsor", Winsorizer(lower=0.025, upper=0.975)),
                    ("scale", StandardScaler()),
                ]
            ),
            CONTINUOUS_FEATURES,
        ),
        ("dummy", "passthrough", DUMMY_FEATURES),
    ]
)

# LASSO
pipe_lasso = Pipeline(
    [
        ("preprocessor", pre),
        (
            "lasso",
            LogisticRegression(
                penalty="l1",
                solver="liblinear",
                class_weight="balanced",
                max_iter=5000,
                random_state=SEED,
            ),
        ),
    ]
)
grid_lasso = {"lasso__C": np.logspace(1, 4, 10)}

# Random Forest
pipe_rf = Pipeline(
    [
        ("preprocessor", pre),
        ("rf", RandomForestClassifier(random_state=SEED, class_weight="balanced")),
    ]
)
grid_rf = {
    "rf__max_depth": list(range(1, 11)),
    "rf__n_estimators": [300],
    "rf__max_features": [2, 4, 8, 16],
}


# MLP
pipe_mlp = Pipeline(
    [
        ("preprocessor", pre),
        (
            "mlp",
            MLPClassifier(
                batch_size=500,
                max_iter=100,
                early_stopping=True,
                n_iter_no_change=5,
                solver="adam",
                random_state=SEED,
            ),
        ),
    ]
)

# Treat hidden_layer_sizes as a hyperparameter
grid_mlp = {
    "mlp__hidden_layer_sizes": [
        (32,),
        (32, 16),
        (32, 16, 8),
        (32, 16, 8, 4),
        (32, 16, 8, 4, 2),
    ],
    "mlp__alpha": np.logspace(-5, -3, 3),
    "mlp__learning_rate_init": [0.0001, 0.0005, 0.001, 0.01],
}


if __name__ == "__main__":

    # Load the data
    df = pd.read_csv(PROCESSED_DATA_CS_PATH / "trader_features_merged.csv")
    df["date"] = pd.to_datetime(df["date"])
    df.dropna(inplace=True)

    # Binary label
    df["label_cls"] = (df["label"] > 0).astype(int)

    # Chronological split
    df = df.sort_values("date")
    cutoff_1 = df["date"].quantile(0.70)
    cutoff_2 = df["date"].quantile(0.85)

    train_df = df[df["date"] <= cutoff_1]
    val_df = df[(df["date"] > cutoff_1) & (df["date"] <= cutoff_2)]
    test_df = df[df["date"] > cutoff_2]

    X_train, y_train = train_df[all_features], train_df["label_cls"]
    X_val, y_val = val_df[all_features], val_df["label_cls"]
    X_test, y_test = test_df[all_features], test_df["label_cls"]

    # models and grids
    model_specs = [
        ("LASSO", pipe_lasso, grid_lasso),
        ("RF", pipe_rf, grid_rf),
        ("NN", pipe_mlp, grid_mlp),
    ]

    # Train and evaluate models
    results = {}
    for name, base_estimator, param_grid in model_specs:
        best_model, best_params, best_val_auc = grid_search(
            name,
            base_estimator,
            param_grid,
            X_train,
            y_train,
            X_val,
            y_val,
        )
        proba_test = best_model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, proba_test)
        results[name] = {
            "model": best_model,
            "best_params": best_params,
            "val_auc": best_val_auc,
            "test_auc": test_auc,
        }

    # collect test probabilities and AUCs
    proba_dict = {}
    auc_dict = {}

    for name, info in results.items():
        model = info["model"]
        proba = model.predict_proba(X_test)[:, 1]
        proba_dict[name] = proba
        auc_dict[name] = info["test_auc"]

    plt.rcParams.update(
        {
            "font.size": 22,
            "font.weight": "bold",
            "axes.grid": True,
        }
    )

    colors = {
        "LASSO": "dimgray",
        "RF": "crimson",
        "NN": "royalblue",
    }

    fig1, ax1 = plt.subplots(figsize=(8, 7))

    for name, proba in proba_dict.items():
        prec, rec, thr = precision_recall_curve(y_test, proba)

        # Drop the artificial last point (precision=1, recall=0) with no threshold
        prec = prec[:-1]
        rec = rec[:-1]

        eps = 1e-8
        f1 = 2 * prec * rec / (prec + rec + eps)

        # No need to append anything to thr
        ax1.plot(thr, prec, linestyle="--", linewidth=1.8, color=colors[name])
        ax1.plot(thr, f1, linestyle="-", linewidth=1.8, color=colors[name])

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Threshold", fontweight="bold")
    ax1.set_ylabel("Score", fontweight="bold")

    measure_lines = [
        plt.Line2D([0], [0], linestyle="--", color="black", lw=1.8, label="Precision"),
        plt.Line2D(
            [0], [0], linestyle="-", color="black", lw=1.8, label=r"$\mathbf{F_1}$"
        ),
    ]
    leg1 = ax1.legend(
        handles=measure_lines,
        title="Measure",
        loc="upper left",
        frameon=False,
        fontsize=18,
        title_fontsize=18,
    )
    ax1.add_artist(leg1)

    model_patches = [
        Patch(facecolor=colors[name], edgecolor="none", label=name)
        for name, _ in proba_dict.items()
    ]
    leg2 = ax1.legend(
        handles=model_patches,
        title="Model",
        loc="lower left",
        frameon=False,
        handlelength=1,
        handleheight=1,
        fontsize=18,
        title_fontsize=18,
    )
    ax1.add_artist(leg2)
    ax1.grid(False)

    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_CS_PATH / "prec_f1_vs_threshold.pdf", dpi=300)
    plt.show()

    # ROC Curves
    fig2, ax2 = plt.subplots(figsize=(8, 7))

    for name, proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax2.plot(
            fpr,
            tpr,
            lw=1.8,
            color=colors[name],
            label=f"{name} (AUC = {auc_dict[name]:.4f})",
        )

    ax2.plot([0, 1], [0, 1], linestyle=":", color="black", lw=1.2)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("False positive rate", fontweight="bold")
    ax2.set_ylabel("True positive rate", fontweight="bold")
    model_patches = [
        Patch(
            facecolor=colors[name],
            edgecolor="none",
            label=f"{name} (AUC = {auc_dict[name]:.4f})",
        )
        for name, _ in proba_dict.items()
    ]
    leg = ax2.legend(
        handles=model_patches,
        title="Model",
        loc="lower right",
        frameon=False,
        handlelength=1,
        handleheight=1,
        fontsize=18,
        title_fontsize=18,
    )
    ax2.add_artist(leg)
    ax2.grid(False)

    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_CS_PATH / "roc_curves.pdf", dpi=300)
    plt.show()
    plt.show()
