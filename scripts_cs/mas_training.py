"""Script to merge model results from multiple training runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from environ.constants import PROCESSED_DATA_CS_PATH
from scripts_cs.ml_preprocess import X_test, X_val

with open(PROCESSED_DATA_CS_PATH / "ml_res.json", "r", encoding="utf-8") as f:
    res_dict = json.load(f)


NAME_MAPPING = {
    "mas_20260114": "MAS",
    "mas_zero_shot_20260114": "MAS (Zero Shot)",
}


def simplex_grid(step: float = 0.05):
    vals = np.arange(0.0, 1.0 + 1e-12, step)
    out = []
    for a in vals:
        for b in vals:
            c = 1.0 - a - b
            if c >= 0:
                out.append((float(a), float(b), float(c)))
    return out


def load_agent_results(path: Path):
    coin, wallet, timing = {}, {}, {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)

            cid = r["custom_id"]
            msg = r["response"]["body"]["choices"][0]
            res = json.loads(msg["message"]["content"])

            lp = msg["logprobs"]["content"][-2]["logprob"]
            p = float(np.exp(lp))
            conf = p if res["result"] else 1.0 - p

            if cid.startswith("coin_"):
                coin[cid[5:]] = conf
            elif cid.startswith("wallet_"):
                wallet[cid[7:]] = conf
            elif cid.startswith("timing_"):
                timing[cid[7:]] = conf

    return coin, wallet, timing


def auc_with_weights(X, coin, wallet, timing, wc, ww, wt):
    y, p = [], []

    for _, r in X.iterrows():
        k = f"{r['token_address']}_{r['trader_address']}"
        if k not in coin or k not in wallet or k not in timing:
            continue

        p.append(wc * coin[k] + ww * wallet[k] + wt * timing[k])
        y.append(r["label_cls"])

    if len(set(y)) < 2:
        return None
    return roc_auc_score(y, p)


def build_output(X, coin, wallet, timing, wc, ww, wt):
    y, ret, p = [], [], []

    for _, r in X.iterrows():
        k = f"{r['token_address']}_{r['trader_address']}"
        if k not in coin or k not in wallet or k not in timing:
            continue

        p.append(wc * coin[k] + ww * wallet[k] + wt * timing[k])
        y.append(r["label_cls"])
        ret.append(r["label"])

    return y, ret, p


for model in ["mas_20260114", "mas_zero_shot_20260114"]:
    val_path = PROCESSED_DATA_CS_PATH / "batch_res_val" / f"{model}.jsonl"
    test_path = PROCESSED_DATA_CS_PATH / "batch_res_test" / f"{model}.jsonl"

    coin_v, wallet_v, timing_v = load_agent_results(val_path)
    coin_t, wallet_t, timing_t = load_agent_results(test_path)

    best_auc = -np.inf
    best_w = (1 / 3, 1 / 3, 1 / 3)

    for wc, ww, wt in simplex_grid(0.05):
        auc = auc_with_weights(X_val, coin_v, wallet_v, timing_v, wc, ww, wt)
        if auc is not None and auc > best_auc:
            best_auc = auc
            best_w = (wc, ww, wt)

    wc, ww, wt = best_w

    y_val, ret_val, p_val = build_output(X_val, coin_v, wallet_v, timing_v, wc, ww, wt)
    y_test, ret_test, p_test = build_output(
        X_test, coin_t, wallet_t, timing_t, wc, ww, wt
    )

    res_dict[NAME_MAPPING[model]] = {
        "best_weights": {"coin": wc, "wallet": ww, "timing": wt},
        "val_auc": roc_auc_score(y_val, p_val) if len(set(y_val)) > 1 else None,
        "test_auc": roc_auc_score(y_test, p_test) if len(set(y_test)) > 1 else None,
        "val_proba": p_val,
        "test_proba": p_test,
        "y_val": y_val,
        "y_test": y_test,
        "ret_val": ret_val,
        "ret_test": ret_test,
    }

    print(
        f"[{NAME_MAPPING[model]}] "
        f"w=({wc:.2f},{ww:.2f},{wt:.2f}) "
        f"VAL AUC={res_dict[NAME_MAPPING[model]]['val_auc']:.4f} "
        f"TEST AUC={res_dict[NAME_MAPPING[model]]['test_auc']:.4f}"
    )


with open(PROCESSED_DATA_CS_PATH / "res.json", "w", encoding="utf-8") as f:
    json.dump(res_dict, f, indent=2)
