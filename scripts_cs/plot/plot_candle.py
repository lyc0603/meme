"""Script to plot a candlestick chart using Plotly with multiprocessing."""

import os
from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm

from environ.constants import FIGURE_PATH, SOL_TOKEN_ADDRESS
from environ.data_class import NewTokenPool
from environ.data_loader import DataLoader
from scripts_cs.ml_preprocess import X_test

os.makedirs(FIGURE_PATH / "candle", exist_ok=True)


def process_pool(
    args: tuple[str, str, str, pd.Timestamp],
) -> None:
    """Script to process a single pool and generate a candlestick plot."""
    token_address, trader_address, chain, first_txn_date = args
    meme = DataLoader(
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
    meme.plot_pre_migration_candlestick_plotly(
        trader_addr=trader_address,
        freq="1min",
        before=first_txn_date,
    )
    meme.save_comment(
        trader_addr=trader_address,
        before=first_txn_date,
    )


if __name__ == "__main__":

    for data in ["candle", "comment"]:
        os.makedirs(FIGURE_PATH / data, exist_ok=True)

    X_test["first_txn_date"] = pd.to_datetime(X_test["first_txn_date"])
    pool_trader = X_test[
        ["token_address", "trader_address", "chain", "first_txn_date"]
    ].values.tolist()

    with Pool(processes=cpu_count() - 2) as pool:
        list(
            tqdm(
                pool.imap_unordered(process_pool, pool_trader),
                total=len(pool_trader),
            )
        )
