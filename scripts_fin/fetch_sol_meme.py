"""Script to fetch solana pool data"""

import json
import os
import time

import requests

from environ.constants import DATA_PATH

os.makedirs(f"{DATA_PATH}/solana/pool", exist_ok=True)

page = 1

while True:
    time.sleep(1)
    while True:
        try:
            response = requests.get(
                f"https://api-v3.raydium.io/pools/info/list?\
poolType=all&poolSortField=apr30d&sortType=asc&pageSize=1000&page={page}",
                timeout=10,
            )
            data = response.json()
            if data:
                break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)
    with open(f"{DATA_PATH}/solana/pool/{page}.json", "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"Page {page} done")

    if data["data"]["hasNextPage"] is True:
        page += 1
    else:
        break
