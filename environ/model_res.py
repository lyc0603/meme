"""Script to merge model results from multiple training runs."""

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from environ.constants import PROCESSED_DATA_CS_PATH
from scripts_cs.ml_preprocess import X_test

with open(PROCESSED_DATA_CS_PATH / "ml_res.json", "r", encoding="utf-8") as f:
    res_dict = json.load(f)

NAME_MAPPING = {
    "mas_20260114": "MAS",
    "mas_zero_shot_20260114": "MAS (Zero Shot)",
}

ma_res_dict = {}
for model in ["mas_20260114", "mas_zero_shot_20260114"]:

    coin_agent_res = {}
    wallet_agent_res = {}

    with open(
        PROCESSED_DATA_CS_PATH / "batch_res" / f"{model}.jsonl",
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            response = json.loads(line)

            custom_idx = response["custom_id"]

            res = json.loads(
                response["response"]["body"]["choices"][0]["message"]["content"]
            )
            toplogprob = response["response"]["body"]["choices"][0]["logprobs"][
                "content"
            ][-2]
            logprob = (
                np.exp(toplogprob["logprob"])
                if res["result"] is True
                else 1 - np.exp(toplogprob["logprob"])
            )

            if custom_idx.startswith("coin_"):
                coin_agent_res[custom_idx[5:]] = {
                    "res": res,
                    "confidence": logprob,
                }
            elif custom_idx.startswith("wallet_"):
                wallet_agent_res[custom_idx[7:]] = {
                    "res": res,
                    "confidence": logprob,
                }

    ma_y_test = []
    ma_proba = []
    for idx, row in X_test.iterrows():
        try:
            token_addr = row["token_address"]
            trader_addr = row["trader_address"]
            coin_confidence = coin_agent_res[f"{token_addr}_{trader_addr}"][
                "confidence"
            ]
            wallet_confidence = wallet_agent_res[f"{token_addr}_{trader_addr}"][
                "confidence"
            ]

            final_prob = 0.5 * coin_confidence + 0.5 * wallet_confidence

            ma_proba.append(final_prob)
            ma_y_test.append(row["label_cls"])

        except KeyError:
            continue

    res_dict[NAME_MAPPING[model]] = {
        "test_auc": roc_auc_score(ma_y_test, ma_proba),
        "proba": ma_proba,
        "y_test": ma_y_test,
    }

# save updated results
with open(PROCESSED_DATA_CS_PATH / "res.json", "w", encoding="utf-8") as f:
    json.dump(res_dict, f, indent=4)
