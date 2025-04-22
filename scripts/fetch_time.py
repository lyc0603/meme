"""
Script to fetch the block of a specific timestamp
"""

import argparse
import datetime
import json
import multiprocessing
import os
import time
from functools import partial
from typing import List

from dotenv import load_dotenv
from tqdm import tqdm
from web3 import Web3
from web3.providers import HTTPProvider
from web3.middleware import ExtraDataToPOAMiddleware

from environ.constants import DATA_PATH, INFURA_API_BASE_DICT, UNISWAP_V3_FACTORY_DICT

load_dotenv()


def get_timestamp_range(chain: str) -> List:
    """Get the timestamp range for a specific chain"""

    done = []

    start_date = datetime.datetime.strptime(
        UNISWAP_V3_FACTORY_DICT[chain]["timestamp"], "%Y-%m-%d"
    ).replace(tzinfo=datetime.timezone.utc)
    current_datetime = datetime.datetime.now(datetime.timezone.utc)

    if os.path.exists(f"{DATA_PATH}/{chain}/time.jsonl"):
        with open(f"{DATA_PATH}/{chain}/time.jsonl", "r", encoding="utf-8") as f:
            done = [json.loads(line)["date"] for line in f]

    todo = []
    while start_date < current_datetime:
        if start_date.strftime("%Y-%m-%d") not in done:
            todo.append(start_date)
        start_date += datetime.timedelta(days=1)

    return todo


def find_blocks_around_timestamp(
    w3: Web3,
    date: datetime.datetime,
) -> dict:
    """Find the blocks before and after a specific timestamp"""

    timestamp = int(date.timestamp())
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    latest_block = w3.eth.block_number
    first_block = 0

    # Binary search to find the block before the timestamp
    low, high = first_block, latest_block
    while low < high:
        time.sleep(0.2)
        mid = (low + high) // 2
        block = w3.eth.get_block(mid)
        if block.timestamp < timestamp:
            low = mid + 1
        else:
            high = mid

    block_before = low - 1 if low > first_block else None
    block_after = low if low <= latest_block else None

    return {
        "date": date.strftime("%Y-%m-%d"),
        "block_before": block_before,
        "block_after": block_after,
    }


def find_blocks_around_timestamp_concurrently(
    api_queue: multiprocessing.Queue,
    res_queue: multiprocessing.Queue,
    date: datetime.datetime,
) -> None:
    """Find the blocks before and after a specific timestamp"""

    http = api_queue.get()
    try:
        res_queue.put(find_blocks_around_timestamp(Web3(HTTPProvider(http)), date))
    except Exception as e:
        print(f"Error fetching block: {e}")
        time.sleep(10)
    api_queue.put(http)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Pool Fetcher CLI")
    parser.add_argument(
        "--chain",
        type=str,
        required=True,
        choices=INFURA_API_BASE_DICT.keys(),
        help="The chain to fetch data from (e.g., polygon).",
    )
    return parser.parse_args()


def main():
    """
    CLI entrypoint
    """

    args = parse_args()
    INFURA_API_KEYS = str(os.getenv("INFURA_API_KEYS")).split(",")

    with multiprocessing.Manager() as manager:
        api_queue = manager.Queue()
        res_queue = manager.Queue()

        for api_key in INFURA_API_KEYS:
            api_queue.put(INFURA_API_BASE_DICT[args.chain] + api_key)

        dates_list = get_timestamp_range(args.chain)
        num_processes = min(len(INFURA_API_KEYS), os.cpu_count())

        with multiprocessing.Pool(processes=num_processes) as pool:
            partial_process = partial(
                find_blocks_around_timestamp, api_queue, res_queue
            )
            for _ in tqdm(
                pool.imap_unordered(partial_process, dates_list),
                total=len(dates_list),
                desc=f"Fetching {args.chain} timestamp",
            ):
                pass

        with open(f"{DATA_PATH}/{args.chain}/time.jsonl", "a", encoding="utf-8") as f:
            while not res_queue.empty():
                data = res_queue.get()
                f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    chain = "avalanche"
    INFURA_API_KEYS = str(os.getenv("INFURA_API_KEYS")).split(",")[-1]
    w3 = Web3(HTTPProvider(INFURA_API_BASE_DICT[chain] + INFURA_API_KEYS))

    _ = find_blocks_around_timestamp(
        w3,
        datetime.datetime.strptime("2025-01-17 14:01:48", "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=datetime.timezone.utc
        ),
    )
