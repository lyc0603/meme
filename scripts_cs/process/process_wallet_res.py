"""Script to process the results of the wallet agent."""

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH


def load_batch_results(file_path: Path) -> pd.DataFrame:
    """
    Load the agent batch results from a JSONL file and return a DataFrame
    with customized column name for the farming field.
    """
    records = {
        "token_address": [],
        "good_wallet": [],
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
            records["good_wallet"].append(content["good_wallet"])
            records["reasoning"].append(content["reasoning"])

    return pd.DataFrame(records)


# Load token_information
label_df = pd.read_csv(PROCESSED_DATA_PATH / "wallet_agent.csv")
label_df["id"] = label_df["wallet_address"] + label_df["eval_token"]

wallet_df = load_batch_results(
    Path(PROCESSED_DATA_PATH) / "batch" / "wallet_agent.jsonl"
)
wallet_df = wallet_df.rename(columns={"token_address": "id"})

# Merge the two DataFrames on the 'id' column
merged_df = pd.merge(wallet_df, label_df, on="id", how="outer")
merged_df["good_wallet"] = merged_df["good_wallet"].fillna(False)

res_dict = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": [],
}
# Evaluate metrics for each frequency
merged_df["label"] = merged_df["profit_mean"].apply(lambda x: 1 if x > 0 else 0)
merged_df["prediction"] = merged_df["good_wallet"]

merged_df.to_csv(PROCESSED_DATA_PATH / "tab" / "wallet.csv", index=False)
