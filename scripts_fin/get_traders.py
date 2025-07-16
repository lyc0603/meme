"""Get sample for traders."""

import random
import os
import json

from tqdm import tqdm

from environ.constants import SOL_TOKEN_ADDRESS, PROCESSED_DATA_PATH
from environ.data_class import NewTokenPool
from environ.meme_analyzer import MemeAnalyzer
from environ.sol_fetcher import import_pool


os.makedirs(f"{PROCESSED_DATA_PATH}/trader", exist_ok=True)

global_set = set()

for chain, num in [
    ("pre_trump_raydium", 1000),
    ("raydium", 1000),
    ("pre_trump_pumpfun", 3000),
    ("pumpfun", 3000),
]:
    traders = {}
    counter = 0

    for pool in tqdm(
        import_pool(
            chain,
            num,
        )
    ):
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
            ),
        )

        if chain in ["pre_trump_pumpfun", "pumpfun"]:
            if meme.check_migrate() | (meme.check_max_purchase_pct() < 0.2):
                continue

        non_bot_creator_transfer_traders_list = list(
            meme.non_bot_creator_transfer_traders
        )

        if len(non_bot_creator_transfer_traders_list) != 0:
            sample_trader = random.choice(non_bot_creator_transfer_traders_list)
        else:
            sample_trader = meme.creator

        while sample_trader in global_set:
            if meme.non_bot_creator_transfer_traders.issubset(global_set):
                sample_trader = meme.creator
                break
            else:
                sample_trader = random.choice(non_bot_creator_transfer_traders_list)

        traders[meme.new_token_pool.pool_add] = sample_trader
        global_set.add(sample_trader)
        counter += 1
        if counter >= 1_000:
            break

    with open(
        f"{PROCESSED_DATA_PATH}/trader/{chain}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(traders, f, indent=4)
