"""
Script to process the traders
"""

import pickle

import pandas as pd
from tqdm import tqdm

from environ.constants import (
    PROCESSED_DATA_PATH,
    TRUMP_BLOCK,
)
from environ.db import fetch_native_pool_since_block
from collections import defaultdict
from environ.data_class import Swap

for chain in ["ethereum"]:

    trader_meme = defaultdict(int)
    pools = [
        pool["args"]["pool"]
        for pool in fetch_native_pool_since_block(
            chain, TRUMP_BLOCK[chain], pool_number=2000
        )
    ]

    for pool in tqdm(pools, total=len(pools), desc=f"Processing {chain} data"):
        with open(f"{PROCESSED_DATA_PATH}/txn/{chain}/{pool}.pkl", "rb") as f:
            pool_data = pickle.load(f)

        trader_set = set()

        for txn in pool_data:
            if len({k: v for k, v in txn.acts.items() if isinstance(v, Swap)}) > 0:
                trader_set.add(txn.maker)

        for trader in trader_set:
            trader_meme[trader] += 1
