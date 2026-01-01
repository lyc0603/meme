#!/usr/bin/env python3
"""
LASSO-Logit + Random Forest + Neural Network with styled plots:
Fig 1: Precision (dashed) & F1 (solid) vs Threshold (all models)
Fig 2: ROC curves with AUC (all models)

Train:    used to fit models.
Val:      used to choose hyperparameters.
Test:     used only once for final evaluation.
"""

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

from environ.constants import FIGURE_PATH, PROCESSED_DATA_CS_PATH

warnings.filterwarnings("ignore")


# ---------- Winsorizer transformer ----------
class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Per-feature winsorization at given lower/upper quantiles.
    Applied column-wise along axis=0.
    """

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        # store per-feature quantiles
        self.lower_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)


# ---------- Load & prep ----------
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

# ---------- Features ----------
continuous_features = [
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
dummy_features = ["launch_bundle", "sniper_bot", "wash_trading_bot", "comment_bot"]
all_features = continuous_features + dummy_features

X_train, y_train = train_df[all_features], train_df["label_cls"]
X_val, y_val = val_df[all_features], val_df["label_cls"]
X_test, y_test = test_df[all_features], test_df["label_cls"]

# ---------- Preprocessor for LASSO / RF ----------
pre = ColumnTransformer(
    transformers=[
        (
            "cont",
            Pipeline(
                steps=[
                    ("winsor", Winsorizer(lower=0.025, upper=0.975)),  # 2.5% / 97.5%
                    ("scale", StandardScaler()),
                ]
            ),
            continuous_features,
        ),
        ("pass", "passthrough", dummy_features),
    ]
)

# ========= LASSO-Logit (L1) with manual grid on VAL =========
lasso_pipe = Pipeline(
    [
        ("preprocessor", pre),
        (
            "lasso",
            LogisticRegression(
                penalty="l1",
                solver="liblinear",  # supports L1 for binary
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            ),
        ),
    ]
)

param_grid_lasso = {"lasso__C": np.logspace(-3, 2, 10)}  # 0.001 ... 100

best_lasso_auc = -np.inf
best_lasso_params = None
best_lasso_model = None

for params in ParameterGrid(param_grid_lasso):
    model = clone(lasso_pipe)
    model.set_params(**params)
    # fit ONLY on train
    model.fit(X_train, y_train)

    # evaluate on validation
    val_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)

    if val_auc > best_lasso_auc:
        best_lasso_auc = val_auc
        best_lasso_params = params
        best_lasso_model = model

lasso = best_lasso_model
coef = lasso.named_steps["lasso"].coef_.ravel()
nonzero = (coef != 0).sum()
print(f"[LASSO] Best params (val AUC={best_lasso_auc:.4f}): {best_lasso_params}")
print(f"[LASSO] Non-zero coefficients = {nonzero}/{coef.size}")

# ========= Random Forest (manual grid on VAL) =========
rf_pipe = Pipeline(
    [
        ("preprocessor", pre),
        ("rf", RandomForestClassifier(random_state=42, class_weight="balanced")),
    ]
)

param_grid_rf = {
    "rf__n_estimators": [200],
    "rf__max_depth": [5, 10],
    "rf__min_samples_leaf": [5, 10],
    "rf__max_features": ["sqrt", "log2"],
}

best_rf_auc = -np.inf
best_rf_params = None
best_rf_model = None

for params in ParameterGrid(param_grid_rf):
    model = clone(rf_pipe)
    model.set_params(**params)
    # fit ONLY on train
    model.fit(X_train, y_train)

    # evaluate on validation
    val_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)

    if val_auc > best_rf_auc:
        best_rf_auc = val_auc
        best_rf_params = params
        best_rf_model = model

rf = best_rf_model
print(f"[RF] Best params (val AUC={best_rf_auc:.4f}): {best_rf_params}")

# ========= Neural Network (MLPClassifier, manual grid) =========
# Separate preprocessor for NN (same structure, but different winsor limits)
pre_nn = ColumnTransformer(
    transformers=[
        (
            "cont",
            Pipeline(
                steps=[
                    ("winsor", Winsorizer(lower=0.01, upper=0.99)),
                    ("scale", StandardScaler()),
                ]
            ),
            continuous_features,
        ),
        ("pass", "passthrough", dummy_features),
    ]
)

# fit preprocessor on TRAIN only
pre_nn.fit(X_train)
X_train_nn = pre_nn.transform(X_train)
X_val_nn = pre_nn.transform(X_val)
X_test_nn = pre_nn.transform(X_test)

param_grid_nn = {
    "hidden_layer_sizes": [(32,), (64,), (32, 16)],
    "alpha": [1e-4, 1e-3, 1e-2],  # L2 penalty
    "learning_rate_init": [0.001, 0.0005],
}

best_auc_nn = -np.inf
best_params_nn = None
best_nn_model = None

for params in ParameterGrid(param_grid_nn):
    model = MLPClassifier(
        random_state=42,
        max_iter=500,
        **params,
    )
    # train ONLY on train (already preprocessed)
    model.fit(X_train_nn, y_train)

    # Evaluate on validation set
    val_proba = model.predict_proba(X_val_nn)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)

    if val_auc > best_auc_nn:
        best_auc_nn = val_auc
        best_params_nn = params
        best_nn_model = model

print(f"[NN] Best params (val AUC={best_auc_nn:.4f}): {best_params_nn}")

# ---------- Test-set probabilities (test used only here) ----------
proba_lasso = lasso.predict_proba(X_test)[:, 1]
proba_rf = rf.predict_proba(X_test)[:, 1]
proba_nn = best_nn_model.predict_proba(X_test_nn)[:, 1]

# ---------- ROC data ----------
auc_lasso = roc_auc_score(y_test, proba_lasso)
auc_rf = roc_auc_score(y_test, proba_rf)
auc_nn = roc_auc_score(y_test, proba_nn)

fpr_lasso, tpr_lasso, _ = roc_curve(y_test, proba_lasso)
fpr_rf, tpr_rf, _ = roc_curve(y_test, proba_rf)
fpr_nn, tpr_nn, _ = roc_curve(y_test, proba_nn)

print(f"[TEST] LASSO AUC = {auc_lasso:.4f}")
print(f"[TEST] RF    AUC = {auc_rf:.4f}")
print(f"[TEST] NN    AUC = {auc_nn:.4f}")

# ---------- Precision / F1 vs Threshold ----------
prec_lasso, rec_lasso, thr_lasso = precision_recall_curve(y_test, proba_lasso)
prec_rf, rec_rf, thr_rf = precision_recall_curve(y_test, proba_rf)
prec_nn, rec_nn, thr_nn = precision_recall_curve(y_test, proba_nn)

eps = 1e-8
f1_lasso = 2 * (prec_lasso * rec_lasso) / (prec_lasso + rec_lasso + eps)
f1_rf = 2 * (prec_rf * rec_rf) / (prec_rf + rec_rf + eps)
f1_nn = 2 * (prec_nn * rec_nn) / (prec_nn + rec_nn + eps)

# Extend thresholds to 1.0 for plotting consistency
thr_lasso = np.append(thr_lasso, 1.0)
thr_rf = np.append(thr_rf, 1.0)
thr_nn = np.append(thr_nn, 1.0)

# ---------- Styling ----------
plt.rcParams.update(
    {
        "font.size": 22,
        "font.weight": "bold",
        "axes.grid": True,
    }
)
color_lasso = "dimgray"
color_rf = "crimson"
color_nn = "royalblue"

# ---------- FIGURE 1: Precision & F1 vs Threshold ----------
fig1, ax1 = plt.subplots(figsize=(6, 5))

# LASSO
ax1.plot(thr_lasso, prec_lasso, linestyle="--", linewidth=1.8, color=color_lasso)
ax1.plot(thr_lasso, f1_lasso, linestyle="-", linewidth=1.8, color=color_lasso)
# RF
ax1.plot(thr_rf, prec_rf, linestyle="--", linewidth=1.8, color=color_rf)
ax1.plot(thr_rf, f1_rf, linestyle="-", linewidth=1.8, color=color_rf)
# NN
ax1.plot(thr_nn, prec_nn, linestyle="--", linewidth=1.8, color=color_nn)
ax1.plot(thr_nn, f1_nn, linestyle="-", linewidth=1.8, color=color_nn)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel("Threshold", fontweight="bold")
ax1.set_ylabel("Score", fontweight="bold")

measure_lines = [
    plt.Line2D([0], [0], linestyle="--", color="black", lw=1.8, label="Precision"),
    plt.Line2D([0], [0], linestyle="-", color="black", lw=1.8, label=r"$\mathbf{F_1}$"),
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
    Patch(facecolor=color_lasso, edgecolor="none", label="LASSO"),
    Patch(facecolor=color_rf, edgecolor="none", label="RF"),
    Patch(facecolor=color_nn, edgecolor="none", label="NN"),
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
plt.savefig(FIGURE_PATH / "prec_f1_vs_threshold.pdf", dpi=300)
plt.show()

# ---------- FIGURE 2: ROC Curves ----------
fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.plot(
    fpr_lasso,
    tpr_lasso,
    color=color_lasso,
    lw=1.8,
    label=f"LASSO (AUC = {auc_lasso:.4f})",
)
ax2.plot(
    fpr_rf,
    tpr_rf,
    color=color_rf,
    lw=1.8,
    label=f"RF (AUC = {auc_rf:.4f})",
)
ax2.plot(
    fpr_nn,
    tpr_nn,
    color=color_nn,
    lw=1.8,
    label=f"NN (AUC = {auc_nn:.4f})",
)
ax2.plot([0, 1], [0, 1], linestyle=":", color="black", lw=1.2)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xlabel("False positive rate", fontweight="bold")
ax2.set_ylabel("True positive rate", fontweight="bold")
ax2.legend(loc="lower right", frameon=False, fontsize=18, title_fontsize=18)
ax2.grid(False)

plt.tight_layout()
plt.savefig(FIGURE_PATH / "roc_curves.pdf", dpi=300)
plt.show()
