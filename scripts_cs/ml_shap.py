"""Script to train random forest multi-classifier on the processed features."""

import time

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from environ.constants import PROCESSED_DATA_CS_PATH

# Load features npy
features = np.load(PROCESSED_DATA_CS_PATH / "train" / "features.npy")

# Load gt npy
gt = np.load(PROCESSED_DATA_CS_PATH / "train" / "gt.npy")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    features, gt, test_size=0.2, random_state=42, stratify=gt
)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Initialize and train the Random Forest Classifier with progress bar
clf = RandomForestClassifier(
    n_estimators=50,
    max_depth=15,
    min_samples_leaf=50,
    n_jobs=-1,
)

clf.fit(X_train, y_train)
# Make predictions on the test set
y_pred = clf.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
report = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Weighted F1 Score:", f1)
print("Classification Report:\n", report)


# ---------- Config ----------
start = time.time()
FIRST_K = 64
# TreeSHAP on large X_test can be slow; sample a subset for reporting
MAX_SHAP_SAMPLES = 2000  # adjust if you want (e.g., 1000/2000/10000)

# Sample rows for SHAP (no strat needed, but you can if you want)
n_test = X_test.shape[0]
idx = np.random.RandomState(42).choice(
    n_test, size=min(MAX_SHAP_SAMPLES, n_test), replace=False
)
# X_shap = X_test[idx]
X_shap = X_test

print(f"\nComputing TreeSHAP on {X_shap.shape[0]} test samples...")

# ---------- TreeExplainer ----------
explainer = shap.TreeExplainer(clf)

# For multi-class sklearn RF, shap_values is usually a list: [ (n, d), ... ] per class
shap_values = explainer.shap_values(X_shap, check_additivity=False)

# ---------- Aggregate to "one number per feature" ----------
# We compute mean(|SHAP|) over samples, and also average across classes (if multi-class).
if isinstance(shap_values, list):
    # list length = n_classes, each array shape (n_samples, n_features)
    sv = np.stack(shap_values, axis=0)  # (n_classes, n_samples, n_features)
    mean_abs = np.mean(np.abs(sv), axis=(0, 1))  # (n_features,)
else:
    # binary or newer API sometimes returns (n_samples, n_features) or (n_samples, n_features, n_outputs)
    sv = np.array(shap_values)
    if sv.ndim == 2:
        mean_abs = np.mean(np.abs(sv), axis=0)  # (n_features,)
    elif sv.ndim == 3:
        # (n_samples, n_features, n_outputs) -> average outputs
        mean_abs = np.mean(np.abs(sv), axis=(0, 2))  # (n_features,)
    else:
        raise ValueError(f"Unexpected shap_values shape: {sv.shape}")

# Report first 64 dims
print("\n===== TreeSHAP mean(|SHAP|) for first 64 dims =====")
for j in range(FIRST_K):
    print(f"Dim {j:02d}: {mean_abs[j]:.8f}")

# Top-10 within first 64
top_idx = np.argsort(-mean_abs[:FIRST_K])[:10]
print("\nTop 10 dims among first 64:")
for j in top_idx:
    print(f"Dim {j:02d} -> {mean_abs[j]:.8f}")


def plot_shap_grouped(mean_abs: np.ndarray, save_path: str | None = None):
    assert (
        mean_abs.ndim == 1 and len(mean_abs) == 136
    ), f"mean_abs shape={mean_abs.shape}"

    groups = [
        ("Wallet graph emb (0–63)", 0, 64),
        ("Time emb (64–87)", 64, 24),
        ("DST emb (88–111)", 88, 24),
        ("Non-DST emb (112–135)", 112, 24),
    ]

    # ---------- 1) Group-level bars: SUM ----------
    group_names = [g[0] for g in groups]
    group_sums = [float(np.sum(mean_abs[s : s + l])) for _, s, l in groups]
    group_means = [float(np.mean(mean_abs[s : s + l])) for _, s, l in groups]

    plt.figure(figsize=(10, 4))
    plt.bar(group_names, group_sums)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Sum of mean(|SHAP|)")
    plt.title("Group-level SHAP importance (sum)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace(".pdf", "_group_sum.pdf"), bbox_inches="tight")
    plt.show()

    # ---------- 2) Group-level bars: MEAN (per-dim) ----------
    plt.figure(figsize=(10, 4))
    plt.bar(group_names, group_means)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Average mean(|SHAP|) per dim")
    plt.title("Group-level SHAP importance (average per dimension)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace(".pdf", "_group_mean.pdf"), bbox_inches="tight")
    plt.show()

    # ---------- 3) Per-dimension bar chart (all 136) with separators ----------
    x = np.arange(len(mean_abs))
    plt.figure(figsize=(12, 4))
    plt.bar(x, mean_abs)
    for _, s, _ in groups:
        plt.axvline(s - 0.5, linestyle="--")
    plt.axvline(136 - 0.5, linestyle="--")
    plt.xlabel("Embedding dimension")
    plt.ylabel("mean(|SHAP|)")
    plt.title("Per-dimension SHAP importance with group boundaries")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace(".pdf", "_per_dim.pdf"), bbox_inches="tight")
    plt.show()

    # ---------- 4) Cumulative contribution (how quickly importance accumulates) ----------
    order = np.argsort(-mean_abs)
    cum = np.cumsum(mean_abs[order]) / np.sum(mean_abs)

    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(1, len(mean_abs) + 1), cum)
    plt.xlabel("#Top dimensions")
    plt.ylabel("Cumulative fraction of total importance")
    plt.title("Cumulative SHAP importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace(".pdf", "_cumulative.pdf"), bbox_inches="tight")
    plt.show()

    # Print quick group contribution summary
    total = float(np.sum(mean_abs))
    print("\n===== Group contribution (share of total) =====")
    for (name, s, l), gs in zip(groups, group_sums):
        print(f"{name:28s}: {gs/total:.3%}")


# Usage after you compute mean_abs:
# plot_shap_grouped(mean_abs, save_path="shap_importance.pdf")  # or None
plot_shap_grouped(mean_abs, save_path=None)

end = time.time()
print(f"\nTotal time: {end - start:.2f} seconds")
