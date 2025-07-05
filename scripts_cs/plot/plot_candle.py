"""Script to plot a candlestick chart using Plotly with multiprocessing."""

import glob
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from environ.constants import FIGURE_PATH, SOL_TOKEN_ADDRESS
from environ.data_class import NewTokenPool
from environ.sol_fetcher import import_pool
from environ.data_loader import DataLoader
from tenacity import retry, stop_after_attempt, wait_fixed

os.makedirs(FIGURE_PATH / "candle", exist_ok=True)

NUM_OF_OBSERVATIONS = 1000
CHAIN = "raydium"


def process_pool(pool: dict) -> None:
    try:
        meme = DataLoader(
            NewTokenPool(
                token0=SOL_TOKEN_ADDRESS,
                token1=pool["token_address"],
                fee=0,
                pool_add=pool["token_address"],
                block_number=0,
                chain=CHAIN,
                base_token=pool["token_address"],
                quote_token=SOL_TOKEN_ADDRESS,
                txns={},
            ),
        )
        meme.plot_pre_migration_candlestick_plotly(freq="1min")
    except Exception as e:
        print(f"Error processing pool {pool['token_address']}: {e}")


if __name__ == "__main__":
    finished = glob.glob(str(FIGURE_PATH / "candle" / "*.png"))
    pools = import_pool(CHAIN, NUM_OF_OBSERVATIONS)

    with Pool(processes=cpu_count() - 5) as pool:
        list(tqdm(pool.imap_unordered(process_pool, pools), total=len(pools)))
