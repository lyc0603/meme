"""Script to process copy trade data and generate statistics."""

import pickle

import pandas as pd
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH

with open(PROCESSED_DATA_PATH / "trader_analyzer.pkl", "rb") as f:
    wallet_stats = pickle.load(f)

df = pd.DataFrame(wallet_stats)
df_profit = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing traders"):

    migrated_profits = sorted(
        [_ for _ in row["profits"] if _[0] > row["migration_block"]],
        key=lambda x: x[0],
    )

    profit = migrated_profits[-1][1] if migrated_profits else 0.0

    df_profit.append(
        {
            **row.to_dict(),
            "profit": profit,
        }
    )

df = pd.DataFrame(df_profit)

df_token = (
    df[["token_address", "migration_block"]]
    .drop_duplicates()
    .sort_values("migration_block", ascending=True)
    .reset_index(drop=True)
)

with open(PROCESSED_DATA_PATH / "wallet_stats.pkl", "rb") as f:
    df_learn_profit = pickle.load(f)

res_list = []

for idx, row in tqdm(
    df_token.iterrows(), total=len(df_token), desc="Processing tokens"
):
    if idx < 50:
        continue

    migrated_token = row["token_address"]
    df_trader = df.loc[df["token_address"] == migrated_token]

    res_list.append(
        pd.merge(
            df_learn_profit,
            df_trader,
            left_on=["wallet_address", "eval_token"],
            right_on=["wallet_address", "token_address"],
            suffixes=("_learn", "_trader"),
            how="inner",
        )
    )

df_trader = pd.concat(res_list)
df_trader.to_csv(PROCESSED_DATA_PATH / "common_wallets.csv", index=False)
