"""
Script to process historical meme data
"""

import json
import os
from glob import glob

from tqdm import tqdm

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH, UNISWAP_V3_FACTORY_DICT

os.makedirs(PROCESSED_DATA_PATH / "plot" / "historical_meme", exist_ok=True)


for chain, _ in UNISWAP_V3_FACTORY_DICT.items():
    file_list = glob(f"{DATA_PATH}/{chain}/pool/*.jsonl")
    pool_data = []

    for file in file_list:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                pool_data.append(json.loads(line))

    pool_data = sorted(pool_data, key=lambda x: x["blockNumber"])

    with open(f"{DATA_PATH}/{chain}/time.jsonl", "r", encoding="utf-8") as f:
        time_data = sorted([json.loads(line) for line in f], key=lambda x: x["date"])

    data_list = []
    token_set = set()
    yesterday_token_set = set()
    yesterday_pool_set = set()

    for date_info in tqdm(time_data, desc=f"Processing {chain} data"):

        pool_before = [
            pool
            for pool in pool_data
            if pool["blockNumber"] <= date_info["block_after"]
        ]

        pool_num = len(pool_before)
        for token_type in ["token0", "token1"]:
            token_set.update([pool["args"][token_type] for pool in pool_before])

        if pool_num == 0:
            continue

        data_list.append(
            {
                "date": date_info["date"],
                "block_before": date_info["block_before"],
                "block_after": date_info["block_after"],
                "pool_num": pool_num,
                "token_num": len(token_set),
                "new_token": list(token_set - yesterday_token_set),
                "new_pool": [_ for _ in pool_before if _ not in yesterday_pool_set],
            }
        )

        yesterday_token_set, yesterday_pool_set = token_set, pool_before

    with open(
        f"{PROCESSED_DATA_PATH}/plot/historical_meme/{chain}.jsonl",
        "w",
        encoding="utf-8",
    ) as f:
        for data in data_list:
            f.write(json.dumps(data) + "\n")
