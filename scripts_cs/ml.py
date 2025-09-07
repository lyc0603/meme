#!/usr/bin/env python3
"""
LASSO-Logit + Random Forest with styled plots:
Fig 1: Precision (dashed) & F1 (solid) vs Threshold (both models)
Fig 2: ROC curves with AUC (both models)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from environ.constants import PROCESSED_DATA_CS_PATH, FIGURE_PATH


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
X_val, y_val = (
    val_df[all_features],
    val_df["label_cls"],
)  # (unused here, but kept if you want threshold tuning)
X_test, y_test = test_df[all_features], test_df["label_cls"]

# ---------- Preprocessor ----------
pre = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), continuous_features),
        ("pass", "passthrough", dummy_features),
    ]
)

# ========= LASSO-Logit (L1) with CV on C =========
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
gs_lasso = GridSearchCV(
    lasso_pipe,
    param_grid_lasso,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=0,
)
gs_lasso.fit(X_train, y_train)
lasso = gs_lasso.best_estimator_

# Report chosen C and sparsity
best_C = gs_lasso.best_params_["lasso__C"]
coef = lasso.named_steps["lasso"].coef_.ravel()
nonzero = (coef != 0).sum()
print(f"[LASSO] Best C = {best_C:.4g} | Non-zero coefficients = {nonzero}/{coef.size}")

# ========= Random Forest (GridSearch) =========
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
gs_rf = GridSearchCV(
    rf_pipe, param_grid_rf, cv=3, scoring="roc_auc", n_jobs=-1, verbose=0
)
gs_rf.fit(X_train, y_train)
rf = gs_rf.best_estimator_
print(f"[RF] Best params: {gs_rf.best_params_}")

# ---------- Test-set probabilities ----------
proba_lasso = lasso.predict_proba(X_test)[:, 1]
proba_rf = rf.predict_proba(X_test)[:, 1]

# ---------- ROC data ----------
auc_lasso = roc_auc_score(y_test, proba_lasso)
auc_rf = roc_auc_score(y_test, proba_rf)
fpr_lasso, tpr_lasso, _ = roc_curve(y_test, proba_lasso)
fpr_rf, tpr_rf, _ = roc_curve(y_test, proba_rf)

# ---------- Precision / F1 vs Threshold ----------
prec_lasso, rec_lasso, thr_lasso = precision_recall_curve(y_test, proba_lasso)
prec_rf, rec_rf, thr_rf = precision_recall_curve(y_test, proba_rf)
eps = 1e-8
f1_lasso = 2 * (prec_lasso * rec_lasso) / (prec_lasso + rec_lasso + eps)
f1_rf = 2 * (prec_rf * rec_rf) / (prec_rf + rec_rf + eps)
thr_lasso = np.append(thr_lasso, 1.0)
thr_rf = np.append(thr_rf, 1.0)

# ---------- FIGURE 1: Precision & F1 vs Threshold ----------
plt.rcParams.update({"font.size": 11, "axes.grid": True})
color_lasso = "dimgray"
color_rf = "crimson"

fig1, ax1 = plt.subplots(figsize=(5, 3))

# LASSO
ax1.plot(thr_lasso, prec_lasso, linestyle="--", linewidth=1.8, color=color_lasso)
ax1.plot(thr_lasso, f1_lasso, linestyle="-", linewidth=1.8, color=color_lasso)
# RF
ax1.plot(thr_rf, prec_rf, linestyle="--", linewidth=1.8, color=color_rf)
ax1.plot(thr_rf, f1_rf, linestyle="-", linewidth=1.8, color=color_rf)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel("Threshold")
ax1.set_ylabel("Score")

# Legends
measure_lines = [
    plt.Line2D([0], [0], linestyle="--", color="black", lw=1.8, label="Precision"),
    plt.Line2D([0], [0], linestyle="-", color="black", lw=1.8, label=r"$F_1$"),
]
leg1 = ax1.legend(
    handles=measure_lines, title="Measure", loc="upper left", frameon=False
)
ax1.add_artist(leg1)

model_patches = [
    Patch(
        facecolor=color_lasso,
        edgecolor="none",
        label="LASSO",
        linewidth=0,
        path_effects=[],
    ),
    Patch(
        facecolor=color_rf, edgecolor="none", label="RF", linewidth=0, path_effects=[]
    ),
]
leg2 = ax1.legend(
    handles=model_patches,
    title="Model",
    loc="lower left",
    frameon=False,
    handlelength=1,
    handleheight=1,
)
ax1.add_artist(leg1)

plt.tight_layout()
plt.savefig(FIGURE_PATH / "prec_f1_vs_threshold.pdf", dpi=300)
plt.show()

# ---------- FIGURE 2: ROC Curves ----------
fig2, ax2 = plt.subplots(figsize=(5, 3))
ax2.plot(
    fpr_lasso,
    tpr_lasso,
    color=color_lasso,
    lw=1.8,
    label=f"LASSO (AUC = {auc_lasso:.4f})",
)
ax2.plot(fpr_rf, tpr_rf, color=color_rf, lw=1.8, label=f"RF (AUC = {auc_rf:.4f})")
ax2.plot([0, 1], [0, 1], linestyle=":", color="black", lw=1.2)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xlabel("False positive rate")
ax2.set_ylabel("True positive rate")
ax2.legend(loc="lower right", frameon=False)
ax2.grid(False)

plt.tight_layout()
plt.savefig(FIGURE_PATH / "roc_curves.pdf", dpi=300)
plt.show()
