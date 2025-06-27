"""Process the results of a script."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
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


def load_batch_results(file_path: Path, farming_field: str) -> pd.DataFrame:
    """
    Load the agent batch results from a JSONL file and return a DataFrame
    with customized column name for the farming field.
    """
    records = {
        "token_address": [],
        farming_field: [],
        "reasoning": [],
    }

    with open(file_path, "r") as f:
        for line in tqdm(f):
            data = json.loads(line)
            token_address = data["custom_id"]
            content = json.loads(
                data["response"]["body"]["choices"][0]["message"]["content"]
            )
            records["token_address"].append(token_address)
            records[farming_field].append(content["good_farming"])
            records["reasoning"].append(content["reasoning"])

    return pd.DataFrame(records)


# Load batches with clearly named fields
batch_technical_df = load_batch_results(
    Path(PROCESSED_DATA_PATH) / "batch/technical_agent.jsonl",
    farming_field="good_farming_technical",
)

batch_comment_df = load_batch_results(
    Path(PROCESSED_DATA_PATH) / "batch/comment_agent.jsonl",
    farming_field="good_farming_comment",
)

batch_comment_df = load_batch_results(
    Path(PROCESSED_DATA_PATH) / "batch/transaction_agent.jsonl",
    farming_field="good_farming_transaction",
)

# Load returns
mdd_df = pd.read_csv(Path(PROCESSED_DATA_PATH) / "ret_ma.csv")

# Merge datasets
merged_df = batch_comment_df.merge(
    batch_technical_df, on="token_address", how="left"
).merge(mdd_df, on="token_address", how="left")

# Prediction logic: you can combine or pick one source
# here using only the technical agent’s field:
merged_df["prediction"] = merged_df["good_farming_technical"] == True

# Evaluate metrics for each frequency
for freq, params in FREQ_DICT.items():
    label_col = f"ret_{freq}"
    merged_df["label"] = merged_df[label_col].apply(lambda x: 1 if x > 0 else 0)

    y_pred = merged_df["prediction"].astype(bool)
    y_true = merged_df["label"].astype(bool)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"Metrics for {freq}:")
    print("-" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("-" * 50, "\n")
