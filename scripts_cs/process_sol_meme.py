"""Script to process the Solana pool data"""

import json
import os
from glob import glob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from tqdm import tqdm

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH

os.makedirs(PROCESSED_DATA_PATH / "plot" / "historical_meme", exist_ok=True)

file_list = glob(f"{DATA_PATH}/solana/pool/*.json")
pool_list = []

for file in tqdm(file_list):
    with open(file, "r", encoding="utf-8") as f:
        pool_data = json.load(f)
        pool_list.extend(pool_data["data"]["data"])

# sort the pool data by open time
pool_list_with_info = sorted(
    # filter the pool data from 2020-01-01 to 2025-03-17
    [i for i in pool_list if 1577836800 <= int(i["openTime"]) <= 1742220657],
    key=lambda x: int(x["openTime"]),
)

# generate a list from 2023-01-23 to 2025-03-01
date_list = []

start_date = datetime(2021, 2, 6)
end_date = datetime(2025, 3, 1)

while start_date <= end_date:
    date_list.append(start_date)
    start_date += timedelta(days=1)

# extra info
len_pool_list_without_info = len(pool_list) - len(pool_list_with_info)
avg_daily_pool_num = len_pool_list_without_info // len(date_list)

token_set_all = set()
for pool in pool_list:
    token_set_all.add(pool["mintA"]["address"])
    token_set_all.add(pool["mintB"]["address"])

token_set_with_info = set()
for pool in pool_list_with_info:
    token_set_with_info.add(pool["mintA"]["address"])
    token_set_with_info.add(pool["mintB"]["address"])

len_token_list_without_info = len(token_set_all) - len(token_set_with_info)
avg_daily_token_num = len_token_list_without_info // len(date_list)

res_list = []

for _, date in tqdm(enumerate(date_list), total=len(date_list)):

    for idx, pool in enumerate(pool_list_with_info):
        if int(pool["openTime"]) > date.timestamp():
            pool_before = pool_list_with_info[: idx + 1]
            break

    pool_num = len(pool_before)
    token_set = set()
    for pool in pool_before:
        token_set.add(pool["mintA"]["address"])
        token_set.add(pool["mintB"]["address"])

    if pool_num == 0:
        continue

    res_list.append(
        {
            "date": date.strftime("%Y-%m-%d"),
            "pool_num": pool_num + avg_daily_pool_num * (_ + 1),
            "token_num": len(token_set) + avg_daily_token_num * (_ + 1),
        }
    )


with open(
    PROCESSED_DATA_PATH / "plot" / "historical_meme" / "solana.jsonl",
    "a",
    encoding="utf-8",
) as f:
    for i in res_list:
        f.write(json.dumps(i) + "\n")
