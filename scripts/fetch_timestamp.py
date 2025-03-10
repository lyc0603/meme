"""
Script to fetch the timestamp block number mapping from DeFiLlama
"""

import datetime
import json

import requests
from tqdm import tqdm

from environ.constants import DATA_PATH, UNISWAP_V3_FACTORY_DICT

DEFILLAMA_TIMESTAMP_ENDPOINT = "https://coins.llama.fi/block"


for chain, chain_info in UNISWAP_V3_FACTORY_DICT.items():

    start_date = datetime.datetime.strptime(
        UNISWAP_V3_FACTORY_DICT[chain]["timestamp"], "%Y-%m-%d"
    ).replace(tzinfo=datetime.timezone.utc)
    current_datetime = datetime.datetime.now(datetime.timezone.utc)

    dates = []
    while start_date < current_datetime:
        dates.append(start_date)
        start_date += datetime.timedelta(days=1)

    with open(f"{DATA_PATH}/{chain}/timestamp.jsonl", "a", encoding="utf-8") as f:
        for date in tqdm(dates, desc=f"Fetching {chain} timestamp"):
            timestamp = int(date.timestamp())
            response = requests.get(
                f"{DEFILLAMA_TIMESTAMP_ENDPOINT}/{chain_info["defillama"]}/{timestamp}",
                timeout=60,
            )
            data = response.json()
            data["date"] = date.strftime("%Y-%m-%d")
            f.write(json.dumps(data) + "\n")
