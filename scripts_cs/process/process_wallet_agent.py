"""Script to process wallet agent data."""

import pandas as pd

from environ.constants import PROCESSED_DATA_PATH

df_trader = pd.read_csv(PROCESSED_DATA_PATH / "common_wallets.csv")

df_res = (
    df_trader.groupby(["wallet_address", "eval_token"])
    .agg(
        learn_profit_sum=pd.NamedAgg(column="learn_profit", aggfunc="sum"),
        learn_profit_std=pd.NamedAgg(column="learn_profit", aggfunc="std"),
        txn_number_mean=pd.NamedAgg(column="txn_number", aggfunc="mean"),
        txn_number_std=pd.NamedAgg(column="txn_number", aggfunc="std"),
        learn_token_count=pd.NamedAgg(column="learn_token", aggfunc="count"),
        profit_mean=pd.NamedAgg(column="profit", aggfunc="mean"),
    )
    .reset_index()
)

df_res = df_res.fillna(0)
# df_res = df_res.loc[df_res["learn_profit_sum"] > 500]
df_res.to_csv(PROCESSED_DATA_PATH / "wallet_agent.csv", index=False)
