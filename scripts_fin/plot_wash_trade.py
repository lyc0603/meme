"""Process profit data for Raydium tokens"""

from multiprocessing import Process, Queue, cpu_count

import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from environ.constants import SOL_TOKEN_ADDRESS, FIGURE_PATH
from environ.data_class import NewTokenPool
from environ.meme_analyzer import MemeAnalyzer
from environ.sol_fetcher import import_pool

FONT_SIZE = 18
CHAIN = "raydium"
NUM_OF_OBSERVATIONS = 1000
SOL_TOKEN_ADDRESS = "So11111111111111111111111111111111111111112"


def worker(tq: Queue, rq: Queue):
    """Worker function to process each pool in the task queue."""
    while True:
        pool = tq.get()
        if pool is None:
            break  # Sentinel value received

        token_add = pool["token_address"]
        meme = MemeAnalyzer(
            NewTokenPool(
                token0=SOL_TOKEN_ADDRESS,
                token1=token_add,
                fee=0,
                pool_add=token_add,
                block_number=0,
                chain=CHAIN,
                base_token=token_add,
                quote_token=SOL_TOKEN_ADDRESS,
                txns={},
            )
        )

        rows = []
        for trader_add, trader in {**meme.traders, **meme.bots}.items():
            row = {
                "token_address": token_add,
                "trader_address": trader_add,
                "creator": 1 if trader.creator else 0,
                "profit": trader.profit,
                "wash_trading_score": trader.wash_trading_score,
            }
            rows.append(row)

        rq.put((rows))


if __name__ == "__main__":
    all_pools = import_pool(CHAIN, NUM_OF_OBSERVATIONS)
    task_queue = Queue()
    result_queue = Queue()

    num_workers = min(cpu_count(), len(all_pools)) - 10

    # Start worker processes
    workers = []
    for _ in range(num_workers):
        p = Process(target=worker, args=(task_queue, result_queue))
        p.start()
        workers.append(p)

    # Feed task queue
    for pool in all_pools:
        task_queue.put(pool)

    # Add sentinel values to stop workers
    for _ in range(num_workers):
        task_queue.put(None)

    profit_rows = []
    pfm_rows = []

    with tqdm(total=len(all_pools)) as pbar:
        for _ in range(len(all_pools)):
            profit_row = result_queue.get()
            profit_rows.extend(profit_row)
            pbar.update(1)

    # Join all processes
    for p in workers:
        p.join()

    pft = pd.DataFrame(profit_rows)

    log_data = np.log1p(pft["wash_trading_score"])

    plt.figure(figsize=(8, 5))

    sns.histplot(log_data, bins=50, stat="count", color="steelblue", edgecolor="black")
    plt.yscale("log")  # Only log scale on y-axis

    # draw a vericle line at x = np.log1p(50)
    plt.axvline(
        np.log1p(50),
        color="red",
        linestyle="--",
        label="Threshold When Wash Trading Score > 50",
    )
    plt.legend()

    plt.xlabel(r"$\text{Ln}(1 + {\it Wash\,Trading\,Score}_{i,j})$", fontsize=FONT_SIZE)
    plt.ylabel("Count", fontsize=FONT_SIZE)
    legend = plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        fontsize=FONT_SIZE,
    )

    plt.tick_params(axis="both", labelsize=FONT_SIZE)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH / "wash_trade_score_dis.pdf", bbox_inches="tight")
    plt.show()
