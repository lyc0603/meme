"""Script to process profit data."""

import pandas as pd
import statsmodels.api as sm
from pyfixest.estimation import feols
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH, SOL_TOKEN_ADDRESS
from environ.data_class import NewTokenPool
from environ.trader_analyzer import TraderAnalyzer

NUM_OF_OBSERVATIONS = 1_000

creator_profit = []

ret_mdd_tab = pd.read_csv(f"{PROCESSED_DATA_PATH}/ret_mdd.csv")

x_var_list = [
    "creator",
    "txn_number",
]

x_var_creator_interaction = [
    "holding_herf",
    "bundle",
    "max_same_txn",
    "pos_to_number_of_swaps_ratio",
    "unique_replies",
    "reply_interval_herf",
    "unique_repliers",
    "non_swapper_repliers",
]

y_var = "profit"

reg_tab = {
    "token_address": [],
    **{x_var: [] for x_var in x_var_list + x_var_creator_interaction + [y_var]},
    **{f"creator_{x_var}": [] for x_var in x_var_creator_interaction},
}
for idx, token_info in tqdm(ret_mdd_tab.iterrows(), total=len(ret_mdd_tab)):
    token_address = token_info["token_address"]
    meme = TraderAnalyzer(
        NewTokenPool(
            token0=SOL_TOKEN_ADDRESS,
            token1=token_address,
            fee=0,
            pool_add=token_address,
            block_number=0,
            chain=chain,
            base_token=token_address,
            quote_token=SOL_TOKEN_ADDRESS,
            txns={},
        ),
    )

    for trader in meme.traders.values():
        reg_tab["creator"].append(trader.creator)
        reg_tab["txn_number"].append(len(trader.swaps))
        reg_tab["token_address"].append(meme.pool_add)
        reg_tab["profit"].append(trader.profit)
        for x_var in x_var_creator_interaction:
            reg_tab[x_var].append(token_info[x_var])
            reg_tab[f"creator_{x_var}"].append(trader.creator * token_info[x_var])

# Convert to DataFrame
reg_tab = pd.DataFrame(reg_tab)
reg_tab["creator"] = reg_tab["creator"].apply(lambda x: 1 if x else 0)
reg_tab.to_csv(f"{PROCESSED_DATA_PATH}/profit.csv", index=False)
