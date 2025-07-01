"""Script to process wallet statistics with multiprocessing"""

import pandas as pd
import pickle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from environ.trader_analyzer import TraderAnalyzer, Trader, Account
from environ.constants import PROCESSED_DATA_PATH

# Load once
with open(PROCESSED_DATA_PATH / "trader_analyzer.pkl", "rb") as f:
    wallet_stats = pickle.load(f)

df = pd.DataFrame(wallet_stats)

df_token = (
    df[["token_address", "migration_block"]]
    .drop_duplicates()
    .sort_values("migration_block", ascending=True)
    .reset_index(drop=True)
)


# Function to process a single row
def process_token(idx):
    if idx < 50:
        return None  # skip

    row = df_token.iloc[idx]

    token_migration_block = row["migration_block"]
    df_token_50 = df_token.iloc[idx - 50 : idx]["token_address"].unique()

    df_sample = df[
        (df["token_address"].isin(df_token_50))
        & (df["migration_block"] <= token_migration_block)
    ].copy()

    records = []
    for _, row2 in df_sample.iterrows():
        current_trader_info = []
        for block, profit in row2["profits"]:
            if row2["migration_block"] <= block <= token_migration_block:
                current_trader_info.append((block, profit))

        past_profit = (
            sorted(current_trader_info, key=lambda x: x[0], reverse=True)[-1][1]
            if current_trader_info
            else 0.0
        )
        records.append(
            {
                "eval_token": row["token_address"],
                "learn_token": row2["token_address"],
                "wallet_address": row2["wallet_address"],
                "creator": row2["creator"],
                "txn_number": len(current_trader_info),
                "learn_profit": past_profit,
            }
        )
    return records


if __name__ == "__main__":
    with Pool(processes=25) as pool:
        results = list(
            tqdm(
                pool.imap(process_token, range(len(df_token))),
                total=len(df_token),
                desc="Processing",
            )
        )

    # flatten and convert to DataFrame
    flat_results = [
        rec for sublist in results if sublist is not None for rec in sublist
    ]

    df_trader = pd.DataFrame(flat_results)
    df_trader = df_trader.loc[df_trader["learn_profit"] > 0]
    with open(PROCESSED_DATA_PATH / "wallet_stats.pkl", "wb") as f:
        pickle.dump(df_trader, f)
