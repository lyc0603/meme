"""Process profit data for Raydium tokens"""

from multiprocessing import Process, Queue, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH, SOL_TOKEN_ADDRESS
from environ.data_class import NewTokenPool
from environ.meme_analyzer import MemeAnalyzer
from environ.sol_fetcher import import_pool

NUM_OF_OBSERVATIONS = 1000
SOL_TOKEN_ADDRESS = "So11111111111111111111111111111111111111112"


def worker(tq: Queue, rq: Queue, chain: str):
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
                chain=chain,
                base_token=token_add,
                quote_token=SOL_TOKEN_ADDRESS,
                txns={},
            )
        )

        max_ret, pump_duration = meme.get_max_ret_and_pump_duration()
        bundle_launch, bundle_bot = meme.get_bundle_launch_buy_sell_num()

        launch_bundle = int(bool(bundle_launch))

        pfm_dict = {
            "max_ret": max_ret,
            "pre_migration_duration": meme.get_pre_migration_duration(),
            "pump_duration": pump_duration,
            "dump_duration": meme.get_dump_duration(),
            "number_of_traders": meme.get_number_of_traders(),
        }

        bot_dict = {
            # Launch Bundle Bot
            "launch_bundle": launch_bundle,
            "bundle_bot": bundle_bot,
            # Volume Bot
            "volume_bot": meme.get_volume_bot(),
            # Comment Bot
            "bot_comment_num": meme.get_comment_bot_num(),
            # Sniper Bot
            "sniper_bot": meme.get_sniper_bot(),
        }
        kol_dict = {
            "winner": meme.get_winner(),
            "loser": meme.get_loser(),
        }

        rows = []
        for trader_add, trader in meme.traders.items():
            row = {
                "token_address": token_add,
                "trader_address": trader_add,
                "creator": 1 if trader.creator else 0,
                "winner": 1 if trader.winner else 0,
                "loser": 1 if trader.loser else 0,
                "profit": trader.profit,
                "wash_trading_score": trader.wash_trading_score,
            }
            for x_var, x_var_value in bot_dict.items():
                row[x_var] = x_var_value
            rows.append(row)

        pfm_df = pd.DataFrame(
            {
                "chain": meme.new_token_pool.chain,
                "token_address": meme.new_token_pool.pool_add,
                **pfm_dict,
                **bot_dict,
                **kol_dict,
            },
            index=[0],
        )

        rq.put((pfm_df, rows))


if __name__ == "__main__":
    for chain in ["pre_trump_raydium", "raydium"]:
        all_pools = import_pool(chain, NUM_OF_OBSERVATIONS)
        task_queue = Queue()
        result_queue = Queue()

        num_workers = min(cpu_count(), len(all_pools)) - 5

        # Start worker processes
        workers = []
        for _ in range(num_workers):
            p = Process(target=worker, args=(task_queue, result_queue, chain))
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
                pfm_row, profit_row = result_queue.get()
                profit_rows.extend(profit_row)
                pfm_rows.append(pfm_row)
                pbar.update(1)

        # Join all processes
        for p in workers:
            p.join()

        pfm = pd.concat(pfm_rows, ignore_index=True)

        for var in [
            "bot_comment_num",
        ]:
            pfm[var] = pfm[var].apply(lambda x: 1 if x > 0 else 0)

        for var in [
            "pre_migration_duration",
            "pump_duration",
            "dump_duration",
            "number_of_traders",
        ]:
            pfm[var] = np.log(pfm[var] + 1)

        pfm.to_csv(f"{PROCESSED_DATA_PATH}/pfm_{chain}.csv", index=False)

        pft = pd.DataFrame(profit_rows)
        pft.to_csv(f"{PROCESSED_DATA_PATH}/pft_{chain}.csv", index=False)
