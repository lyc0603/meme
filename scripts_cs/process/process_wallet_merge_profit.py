"""Script to merge wallet learn profit data with trader analyzer data."""

import pickle
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from environ.constants import PROCESSED_DATA_PATH


def load_pickle(path: Path):
    """Load a pickle file from the given path."""
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_profit(row):
    """Compute the profit for a given row based on migration block."""
    migrated_profits = sorted(
        [p for p in row["profits"] if p[0] > row["migration_block"]],
        key=lambda x: x[0],
    )
    return migrated_profits[-1][1] if migrated_profits else 0.0


def main():
    """Main function to process and merge wallet data."""
    # load trader analyzer data
    df_raw = pd.DataFrame(load_pickle(PROCESSED_DATA_PATH / "trader_analyzer.pkl"))
    df_raw["profit"] = df_raw.apply(compute_profit, axis=1)

    # keep only tokens after migration index 50
    df_token = (
        df_raw[["token_address", "migration_block"]]
        .drop_duplicates()
        .sort_values("migration_block")
        .reset_index(drop=True)
    )

    filtered_tokens = df_token.loc[df_token.index >= 50, "token_address"]

    # filter raw data for those tokens only
    df_trader = df_raw[df_raw["token_address"].isin(filtered_tokens)]

    # load wallet learn profit data
    df_learn_profit = load_pickle(PROCESSED_DATA_PATH / "wallet_stats.pkl")

    # single merge
    df_merged = pd.merge(
        df_learn_profit,
        df_trader,
        left_on=["wallet_address", "eval_token"],
        right_on=["wallet_address", "token_address"],
        suffixes=("_learn", "_trader"),
        how="inner",
    )

    df_merged.to_csv(PROCESSED_DATA_PATH / "common_wallets.csv", index=False)


if __name__ == "__main__":
    main()
