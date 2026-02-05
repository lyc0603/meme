"""
Train models, compute SHAP on FULL validation set, and SAVE plot-ready data.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
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
ALL_FEATURES = CONTINUOUS_FEATURES + DUMMY_FEATURES


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X: np.ndarray, y=None) -> "Winsorizer":
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)


def grid_search(name, base_estimator, param_grid, X_train, y_train, X_val, y_val):
    best_model, best_params, best_auc, best_val_proba = None, None, -np.inf, None
    for params in tqdm(ParameterGrid(param_grid), desc=f"Grid search for {name}"):
        model = clone(base_estimator)
        model.set_params(**params)
        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_proba)
        if auc > best_auc:
            best_auc, best_model, best_params, best_val_proba = (
                auc,
                model,
                params,
                val_proba,
            )
    return best_model, best_params, float(best_auc), best_val_proba


def _get_X_transformed_and_names(fitted_pipeline: Pipeline, X_raw: pd.DataFrame):
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    X_t = preprocessor.transform(X_raw)
    names = list(CONTINUOUS_FEATURES) + list(DUMMY_FEATURES)
    return X_t, names


def compute_shap_values_and_names(
    name: str,
    fitted_pipeline: Pipeline,
    X_bg_raw: pd.DataFrame,
    X_explain_raw: pd.DataFrame,
    random_state: int = 42,
    max_background: int | None = 200,
):
    est_name = name.lower()

    X_explain_t, feat_names = _get_X_transformed_and_names(
        fitted_pipeline, X_explain_raw
    )
    X_bg_t, _ = _get_X_transformed_and_names(fitted_pipeline, X_bg_raw)

    if max_background is None or X_bg_t.shape[0] <= max_background:
        X_bg = X_bg_t
    else:
        rs = np.random.RandomState(random_state)
        idx = rs.choice(X_bg_t.shape[0], max_background, replace=False)
        X_bg = X_bg_t[idx]

    if est_name in ["xgboost", "xgb"]:
        model = fitted_pipeline.named_steps["xgb"]
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_explain_t)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_vals = np.asarray(shap_vals)

    elif est_name == "lasso":
        model = fitted_pipeline.named_steps["lasso"]
        explainer = shap.LinearExplainer(
            model, X_bg, feature_perturbation="interventional"
        )
        shap_vals = explainer.shap_values(X_explain_t)
        shap_vals = np.asarray(shap_vals)

    else:
        model = fitted_pipeline.named_steps["mlp"]

        def f(X):
            return model.predict_proba(X)[:, 1]

        explainer = shap.PermutationExplainer(f, X_bg)
        shap_vals = np.asarray(explainer(X_explain_t).values)

    return shap_vals, X_explain_t, feat_names


def compute_feature_importance_shap_from_vals(feat_names, shap_vals):
    imp = np.mean(np.abs(shap_vals), axis=0)
    pairs = sorted(zip(feat_names, imp), key=lambda x: float(x[1]), reverse=True)
    return {"method": "shap_mean_abs", "importance": {k: float(v) for k, v in pairs}}


def main():
    out_dir = Path(PROCESSED_DATA_CS_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)

    pre = ColumnTransformer(
        transformers=[
            (
                "cont",
                Pipeline(
                    [("winsor", Winsorizer(0.025, 0.975)), ("scale", StandardScaler())]
                ),
                CONTINUOUS_FEATURES,
            ),
            ("dummy", "passthrough", DUMMY_FEATURES),
        ]
    )

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

    model_specs = [
        ("Lasso", pipe_lasso, grid_lasso),
        ("NN", pipe_mlp, grid_mlp),
        ("XGBoost", pipe_xgb, grid_xgb),
    ]

    results = {}
    shap_vals_dict = {}
    X_val_t_dict = {}
    feat_names_final = None

    for name, base_estimator, param_grid in model_specs:
        best_model, best_params, best_val_auc, best_val_proba = grid_search(
            name, base_estimator, param_grid, X_train, y_train, X_val, y_val
        )

        proba_train = best_model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, proba_train)

        proba_test = best_model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, proba_test)

        shap_vals, X_val_t, feat_names = compute_shap_values_and_names(
            name=name,
            fitted_pipeline=best_model,
            X_bg_raw=X_train,
            X_explain_raw=X_val,
            random_state=SEED,
            max_background=None,  # set e.g. 200 for speed if needed
        )
        feat_names_final = feat_names

        shap_vals_dict[name] = shap_vals
        X_val_t_dict[name] = X_val_t

        feat_imp = compute_feature_importance_shap_from_vals(feat_names, shap_vals)

        results[name] = {
            "best_params": best_params,
            "train_auc": float(train_auc),
            "val_auc": float(best_val_auc),
            "test_auc": float(test_auc),
            "feature_importance": feat_imp,
        }

    # Save JSON metrics
    with open(out_dir / "ml_res_shap.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    np.savez_compressed(
        out_dir / "shap_plot_data.npz",
        feature_names=np.array(feat_names_final, dtype=object),
        model_names=np.array(list(shap_vals_dict.keys()), dtype=object),
        **{f"shap__{k}": v for k, v in shap_vals_dict.items()},
        **{f"Xval__{k}": v for k, v in X_val_t_dict.items()},
    )


if __name__ == "__main__":
    main()
