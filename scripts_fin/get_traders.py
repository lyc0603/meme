"""Get sample for traders."""

import random
import os
import json

from tqdm import tqdm

from environ.constants import SOL_TOKEN_ADDRESS, PROCESSED_DATA_PATH
from environ.data_class import NewTokenPool
from environ.meme_analyzer import MemeAnalyzer
from environ.sol_fetcher import import_pool

NUM_OF_OBSERVATIONS = 1000

os.makedirs(f"{PROCESSED_DATA_PATH}/trader", exist_ok=True)

for chain in [
    # "pre_trump_pumpfun",
    "pre_trump_raydium",
    # "pumpfun",
    "raydium",
]:
    traders = {}

    for pool in tqdm(
        import_pool(
            chain,
            NUM_OF_OBSERVATIONS,
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
        sample_trader = random.choice(list(meme.non_bot_creator_transfer_traders))

        while sample_trader in traders.values():
            sample_trader = random.choice(list(meme.non_bot_creator_transfer_traders))

        traders[meme.new_token_pool.pool_add] = sample_trader

    with open(
        f"{PROCESSED_DATA_PATH}/trader/{chain}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(traders, f, indent=4)
