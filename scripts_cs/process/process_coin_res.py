"""Process the results of a script."""

import json
from pathlib import Path
import os

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH

FREQ_DICT = {
    "1 Min": {"freq": "1min", "before": 1},
    "5 Mins": {"freq": "5min", "before": 5},
    "10 Mins": {"freq": "1min", "before": 10},
    "30 Mins": {"freq": "30min", "before": 30},
    "1 Hour": {"freq": "1h", "before": 1},
    "5 Hours": {"freq": "1h", "before": 5},
    "10 Hours": {"freq": "1h", "before": 10},
}


def load_batch_results(file_path: Path) -> pd.DataFrame:
    """
    Load the agent batch results from a JSONL file and return a DataFrame
    with customized column name for the farming field.
    """
    records = {
        "token_address": [],
        "good_farming": [],
        "reasoning": [],
    }

    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            data = json.loads(line)
            token_address = data["custom_id"]
            content = json.loads(
                data["response"]["body"]["choices"][0]["message"]["content"]
            )
            records["token_address"].append(token_address)
            records["good_farming"].append(content["good_farming"])
            records["reasoning"].append(content["reasoning"])

    return pd.DataFrame(records)


# Load returns
mdd_df = pd.read_csv(Path(PROCESSED_DATA_PATH) / "ret_ma.csv")

res_dict = {}

for agent_name in ["comment", "technical", "transaction", "coin"]:

    batch_coin_df = load_batch_results(
        Path(PROCESSED_DATA_PATH) / "batch" / f"{agent_name}_agent.jsonl",
    )

    merged_df = batch_coin_df.merge(mdd_df, on="token_address", how="left")
    merged_df["prediction"] = merged_df["good_farming"] == True

    os.makedirs(PROCESSED_DATA_PATH / "tab", exist_ok=True)
    res_dict[agent_name] = {
        "freq": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
    }
    # Evaluate metrics for each frequency
    for freq, params in FREQ_DICT.items():
        label_col = f"ret_{freq}"
        merged_df["label"] = merged_df[label_col].apply(
            lambda x: 1 if x > merged_df[label_col].median() else 0
        )

        y_pred = merged_df["prediction"].astype(bool)
        y_true = merged_df["label"].astype(bool)

        res_dict[agent_name]["freq"].append(freq)
        res_dict[agent_name]["accuracy"].append(accuracy_score(y_true, y_pred))
        res_dict[agent_name]["precision"].append(
            precision_score(y_true, y_pred, zero_division=0)
        )
        res_dict[agent_name]["recall"].append(
            recall_score(y_true, y_pred, zero_division=0)
        )
        res_dict[agent_name]["f1_score"].append(
            f1_score(y_true, y_pred, zero_division=0)
        )

with open(Path(PROCESSED_DATA_PATH) / "tab" / "res.json", "w", encoding="utf-8") as f:
    json.dump(res_dict, f, indent=4)
