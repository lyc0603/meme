"""Script to train ML models to classify traders based on their features."""

import json
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
                max_iter=100,
                early_stopping=True,
                n_iter_no_change=5,
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


if __name__ == "__main__":

    # models and grids
    model_specs = [
        ("LASSO", pipe_lasso, grid_lasso),
        ("RF", pipe_rf, grid_rf),
        ("NN", pipe_mlp, grid_mlp),
        ("XGBoost", pipe_xgb, grid_xgb),
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
    res_dict = {}
    for name, info in results.items():
        model = info["model"]
        proba = model.predict_proba(X_test)[:, 1]
        # proba_dict[name] = proba
        # auc_dict[name] = info["test_auc"]
        res_dict[name] = {
            "best_params": info["best_params"],
            "test_auc": info["test_auc"],
            "proba": proba.tolist(),
            "y_test": y_test.tolist(),
        }

    with open(PROCESSED_DATA_CS_PATH / "ml_res.json", "w", encoding="utf-8") as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)
