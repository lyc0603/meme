"""Script to process return and maximum drawup (MDU) for token pools."""

import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process, Queue, cpu_count

from environ.constants import PROCESSED_DATA_PATH
from environ.data_class import NewTokenPool
from environ.meme_analyzer import MemeAnalyzer
from environ.sol_fetcher import import_pool

CHAIN = "raydium"
NUM_OF_OBSERVATIONS = 1000
SOL_TOKEN_ADDRESS = "So11111111111111111111111111111111111111112"

FREQ_DICT = {
    "1 Min": {"freq": "1min", "before": 1},
    "5 Mins": {"freq": "1min", "before": 5},
    "10 Mins": {"freq": "1min", "before": 10},
    "30 Mins": {"freq": "1min", "before": 30},
    "1 Hour": {"freq": "1h", "before": 1},
    "5 Hours": {"freq": "1h", "before": 5},
    "10 Hours": {"freq": "1h", "before": 10},
}


def worker(task_queue: Queue, result_queue: Queue):
    """Worker function to process each pool in the task queue."""
    while True:
        pool = task_queue.get()
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

        mdd_df = pd.DataFrame(
            {
                **{
                    f"mdu_{name}": meme.get_mdu(
                        freq=freq["freq"], before=freq["before"]
                    )
                    for name, freq in FREQ_DICT.items()
                },
                **{
                    f"mdd_{name}": meme.get_mdd(
                        freq=freq["freq"], before=freq["before"]
                    )
                    for name, freq in FREQ_DICT.items()
                },
                **{
                    f"ret_{name}": meme.get_ret_before(
                        freq=freq["freq"], before=freq["before"]
                    )
                    for name, freq in FREQ_DICT.items()
                },
                **{
                    "chain": meme.new_token_pool.chain,
                    "token_address": meme.new_token_pool.pool_add,
                    "migration_block": meme.migrate_block,
                    "duration": meme.migration_duration,
                    "#trader": len(meme.swappers),
                    "#txn": len(meme.txn),
                    "holding_herf": meme.get_holdings_herf(),
                    "launch_bundle_transfer": meme.get_bundle_launch_transfer_dummy(),
                    "bundle_creator_buy": meme.get_bundle_creator_buy_dummy(),
                    **dict(
                        zip(
                            ["bundle_launch", "bundle_buy", "bundle_sell"],
                            meme.get_bundle_launch_buy_sell_num(),
                        )
                    ),
                },
            },
            index=[0],
        )

        result_queue.put(mdd_df)


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

    # Collect results
    mdd_list = []
    with tqdm(total=len(all_pools)) as pbar:
        for _ in range(len(all_pools)):
            mdd_list.append(result_queue.get())
            pbar.update(1)

    # Join all processes
    for p in workers:
        p.join()

    # Combine and save
    mdd_df = pd.concat(mdd_list, ignore_index=True)

    mdd_df.to_csv(f"{PROCESSED_DATA_PATH}/ret_ma.csv", index=False)
