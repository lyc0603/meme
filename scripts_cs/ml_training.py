"""Script to train ML models to classify traders based on their features."""

import json
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier

from environ.constants import PROCESSED_DATA_CS_PATH
from scripts_cs.ml_preprocess import X_test, X_train, X_val, y_test, y_train, y_val

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
    "first_txn_amount",
    "first_txn_quantity",
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
) -> tuple[Pipeline, dict, float, np.ndarray]:
    """Perform grid search over param_grid for the given base_estimator."""

    best_model = None
    best_params = None
    best_auc = -np.inf
    best_val_proba = None

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
            best_val_proba = val_proba

    return best_model, best_params, best_auc, best_val_proba


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
                l1_ratio=1.0,
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
        (
            "rf",
            RandomForestClassifier(
                random_state=SEED, class_weight="balanced", n_jobs=-1
            ),
        ),
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
                max_iter=1000,
                early_stopping=False,
                solver="adam",
                random_state=SEED,
            ),
        ),
    ]
)

# XGBoost
pipe_xgb = Pipeline(
    [
        ("preprocessor", pre),
        (
            "xgb",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                random_state=SEED,
                n_jobs=-1,
                tree_method="hist",
                verbosity=0,
            ),
        ),
    ]
)

grid_xgb = {
    "xgb__n_estimators": [300, 600],
    "xgb__max_depth": [2, 3, 4, 6],
    "xgb__learning_rate": [0.01, 0.05, 0.1],
    "xgb__subsample": [0.7, 0.9, 1.0],
    "xgb__colsample_bytree": [0.7, 0.9, 1.0],
    "xgb__min_child_weight": [1, 5, 10],
    "xgb__reg_lambda": [1.0, 5.0, 10.0],
}

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


def compute_feature_importance(
    name: str,
    fitted_pipeline: Pipeline,
    feature_names: list[str],
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
) -> dict:
    """
    Returns a JSON-serializable dict:
      {
        "method": "...",
        "importance": {feature: value, ...}  # sorted desc by value
      }
    """
    # after pipeline.fit, you can access final estimator by step name
    if name.lower() == "lasso":
        est = fitted_pipeline.named_steps["lasso"]
        imp = np.abs(est.coef_.ravel())

        method = "abs_coef"

    elif name.lower() == "rf":
        est = fitted_pipeline.named_steps["rf"]
        imp = est.feature_importances_

        method = "gini_importance"

    elif name.lower() == "xgboost":
        est = fitted_pipeline.named_steps["xgb"]
        imp = est.feature_importances_

        method = "xgb_gain_or_split_importance"

    else:
        # NN: permutation importance on validation set
        r = permutation_importance(
            fitted_pipeline,
            X_val,
            y_val,
            scoring="roc_auc",
            n_repeats=10,
            random_state=random_state,
            n_jobs=-1,
        )
        imp = r.importances_mean
        method = "permutation_importance_auc"

    # Build sorted mapping (desc)
    pairs = sorted(zip(feature_names, imp), key=lambda x: float(x[1]), reverse=True)
    return {
        "method": method,
        "importance": {k: float(v) for k, v in pairs},
    }


if __name__ == "__main__":

    # models and grids
    model_specs = [
        ("Lasso", pipe_lasso, grid_lasso),
        ("RF", pipe_rf, grid_rf),
        ("NN", pipe_mlp, grid_mlp),
        ("XGBoost", pipe_xgb, grid_xgb),
    ]

    # Train and evaluate models
    results = {}
    for name, base_estimator, param_grid in model_specs:
        best_model, best_params, best_val_auc, best_val_proba = grid_search(
            name,
            base_estimator,
            param_grid,
            X_train,
            y_train,
            X_val,
            y_val,
        )

        # --- NEW: train metrics ---
        proba_train = best_model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, proba_train)

        # existing
        proba_val = best_val_proba
        proba_test = best_model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, proba_test)

        # feature importance
        feat_imp = compute_feature_importance(
            name=name,
            fitted_pipeline=best_model,
            feature_names=list(CONTINUOUS_FEATURES) + list(DUMMY_FEATURES),
            X_val=X_val,
            y_val=y_val,
            random_state=SEED,
        )

        results[name] = {
            "best_params": best_params,
            "train_auc": float(train_auc),
            "val_auc": float(best_val_auc),
            "test_auc": float(test_auc),
            "train_proba": proba_train.tolist(),
            "val_proba": proba_val.tolist(),
            "test_proba": proba_test.tolist(),
            "y_train": y_train.tolist(),
            "y_val": y_val.tolist(),
            "y_test": y_test.tolist(),
            "ret_train": X_train["label"].tolist(),
            "ret_val": X_val["label"].tolist(),
            "ret_test": X_test["label"].tolist(),
            "copy_trading_ret_train": X_train["copy_trading_ret"].tolist(),
            "copy_trading_ret_val": X_val["copy_trading_ret"].tolist(),
            "copy_trading_ret_test": X_test["copy_trading_ret"].tolist(),
            "feature_importance": feat_imp,
        }

    with open(PROCESSED_DATA_CS_PATH / "ml_res.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
