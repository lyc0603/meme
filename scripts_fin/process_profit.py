"""Process profit data for Raydium tokens"""

import pandas as pd
from tqdm import tqdm
import multiprocessing

from environ.constants import PROCESSED_DATA_PATH, SOL_TOKEN_ADDRESS
from environ.data_class import NewTokenPool
from environ.trader_analyzer import TraderAnalyzer

chain = "raydium"


x_var_list = ["creator", "txn_number"]
x_var_creator_interaction = [
    "launch_bundle_transfer",
    "bundle_creator_buy",
    "bundle_launch",
    "bundle_buy",
    "bundle_sell",
    "max_same_txn",
    "pos_to_number_of_swaps_ratio",
    "bot_comment_num",
    "positive_bot_comment_num",
    "negative_bot_comment_num",
]
y_var = "profit"


def producer(ret_mdd_tab, task_queue, num_workers):
    """Producer function to put token information into the task queue."""
    for idx, token_info in ret_mdd_tab.iterrows():
        task_queue.put(token_info)
    # Add sentinels for consumers to exit
    for _ in range(num_workers):
        task_queue.put(None)


def consumer(task_queue, result_queue):
    """Consumer function to process token information and calculate profit."""
    while True:
        token_info = task_queue.get()
        if token_info is None:
            break
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
        rows = []
        for trader in meme.traders.values():
            row = {
                "token_address": token_info["token_address"],
                "creator": 1 if trader.creator else 0,
                "txn_number": len(trader.swaps),
                "profit": trader.profit,
            }
            for x_var in x_var_creator_interaction:
                row[x_var] = token_info[x_var]
                row[f"creator_{x_var}"] = trader.creator * token_info[x_var]
            rows.append(row)
        result_queue.put(rows)


def main():
    """Main function to process profit data."""
    ret_mdd_tab = pd.read_csv(f"{PROCESSED_DATA_PATH}/ret_mdd.csv")

    num_workers = max(1, multiprocessing.cpu_count() - 5)
    task_queue = multiprocessing.Queue(maxsize=num_workers * 2)
    result_queue = multiprocessing.Queue()

    consumers = [
        multiprocessing.Process(target=consumer, args=(task_queue, result_queue))
        for _ in range(num_workers)
    ]
    for c in consumers:
        c.start()

    prod = multiprocessing.Process(
        target=producer, args=(ret_mdd_tab, task_queue, num_workers)
    )
    prod.start()

    reg_rows = []
    total = len(ret_mdd_tab)
    with tqdm(total=total, desc="Processing tokens") as pbar:
        finished = 0
        while finished < total:
            result = result_queue.get()
            reg_rows.extend(result)
            finished += 1
            pbar.update(1)

    prod.join()
    for c in consumers:
        c.join()

    reg_tab = pd.DataFrame(reg_rows)
    reg_tab.to_csv(f"{PROCESSED_DATA_PATH}/profit.csv", index=False)


if __name__ == "__main__":
    main()
