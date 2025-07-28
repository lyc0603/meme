"""Process profit data for Raydium tokens"""

from multiprocessing import Process, Queue, cpu_count

import json
import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH, SOL_TOKEN_ADDRESS
from environ.data_class import NewTokenPool
from environ.meme_analyzer import MemeAnalyzer
from environ.sol_fetcher import import_pool
from environ.utils import handle_first_wash_bot, handle_first_comment_bot

NUM_OF_OBSERVATIONS = 1000
SOL_TOKEN_ADDRESS = "So11111111111111111111111111111111111111112"


wt_set = set()

for task in ["wt", "cm"]:
    if os.path.exists(PROCESSED_DATA_PATH / "wt_cm" / f"delta_set_{task}.pkl"):
        with open(PROCESSED_DATA_PATH / "wt_cm" / f"delta_set_{task}.pkl", "rb") as f:
            wt_set_task = pickle.load(f)
        wt_set.update(wt_set_task)


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

        time_trader_dict = meme.get_time_traders(wt_set) if wt_set else {}

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

        trader_rows = []
        for trader_add, trader in meme.traders.items():
            row = {
                "token_address": token_add,
                "trader_address": trader_add,
                "creator": 1 if trader.creator else 0,
                "winner": 1 if trader.winner else 0,
                "loser": 1 if trader.loser else 0,
                "neutral": 1 if trader.neutral else 0,
                "sniper": 1 if trader.sniper else 0,
                "profit": trader.profit,
                "wash_trading_score": trader.wash_trading_score,
            }
            for x_var, x_var_value in bot_dict.items():
                row[x_var] = x_var_value
            trader_rows.append(row)

        # Handle first wash-trading bot
        wash_trading_bot_rows = handle_first_wash_bot(meme, token_add, meme.launch_time)

        # Handle first comment of the first bot
        comment_bot_rows = handle_first_comment_bot(meme, token_add, meme.launch_time)

        pfm_df = pd.DataFrame(
            {
                "chain": meme.new_token_pool.chain,
                "token_address": meme.new_token_pool.pool_add,
                **pfm_dict,
                **bot_dict,
            },
            index=[0],
        )

        rq.put(
            (
                pfm_df,
                trader_rows,
                wash_trading_bot_rows,
                comment_bot_rows,
                time_trader_dict,
            )
        )


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

        pfm_rows = []
        profit_rows = []
        wash_rows = []
        comment_rows = []
        time_trader_rows = {}

        with tqdm(total=len(all_pools)) as pbar:
            for _ in range(len(all_pools)):
                pfm_row, profit_row, wash_row, comment_row, time_trader_row = (
                    result_queue.get()
                )
                pfm_rows.append(pfm_row)
                profit_rows.extend(profit_row)
                wash_rows.extend(wash_row)
                comment_rows.extend(comment_row)
                time_trader_rows[pfm_row["token_address"].values[0]] = time_trader_row
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

        # Save wash trading bots
        os.makedirs(PROCESSED_DATA_PATH / "wt_cm", exist_ok=True)
        bot_df = pd.DataFrame(wash_rows)
        bot_df.to_csv(
            f"{PROCESSED_DATA_PATH}/wt_cm/wash_trading_{chain}.csv", index=False
        )

        # Save comment bot data (only first comment of first bot)
        comment_df = pd.DataFrame(comment_rows)
        comment_df.to_csv(
            f"{PROCESSED_DATA_PATH}/wt_cm/comment_bots_{chain}.csv", index=False
        )

        # Save time traders data
        with open(
            PROCESSED_DATA_PATH / "wt_cm" / f"time_traders_{chain}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(time_trader_rows, f)
