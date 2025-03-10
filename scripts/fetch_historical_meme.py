"""
Script to fetch new pool data
"""

import argparse
import json
import logging
import multiprocessing
import os
import time
from functools import partial

from dotenv import load_dotenv
from tqdm import tqdm
from web3 import Web3
from web3.providers import HTTPProvider

from environ.constants import (
    ABI_PATH,
    DATA_PATH,
    INFURA_API_BASE_DICT,
    UNISWAP_V3_FACTORY_DICT,
)
from environ.utils import _fetch_events_for_all_contracts, fetch_current_block, to_dict

load_dotenv()

os.makedirs(f"{DATA_PATH}/log", exist_ok=True)
logging.basicConfig(
    filename=f"{DATA_PATH}/log/error.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.ERROR,
)


def fetch_new_pools(
    chain: str, from_block: int, to_block: int, queue: multiprocessing.Queue
) -> None:
    """Fetch new pools using a specific API key and block range"""

    time.sleep(1)
    http = queue.get()

    try:
        w3 = Web3(HTTPProvider(http))

        # Fetch pool creation events
        pool_created_event = w3.eth.contract(
            abi=json.load(open(f"{ABI_PATH}/v3factory.json", encoding="utf-8"))
        ).events.PoolCreated

        events = _fetch_events_for_all_contracts(
            w3,
            pool_created_event,
            {"address": UNISWAP_V3_FACTORY_DICT[chain]["address"]},
            from_block,
            to_block,
        )

        events = to_dict(events)

        with open(
            f"{DATA_PATH}/{chain}/pool/{from_block}_{to_block}.jsonl",
            "a",
            encoding="utf-8",
        ) as f:
            for event in events:
                f.write(json.dumps(event) + "\n")
    except Exception as e:
        print(
            f"Fetching Pools: Block not found for block range {from_block} - {to_block}, {e}"
        )

    queue.put(http)


def fetch_new_pools_wrap(
    chain: str, queue: multiprocessing.Queue, block_tuple: tuple
) -> None:
    """Unpack block tuple and trigger pool fetching"""
    from_block, to_block = block_tuple
    fetch_new_pools(chain, from_block, to_block, queue)


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


def split_blocks(start_block: int, end_block: int, step: int, chain: str) -> list:
    """
    Split the blocks into step ranges
    """

    min_block = (start_block // step) * step
    max_block = (end_block // step) * step

    blocks = []

    for i in range(min_block, max_block, step):
        # check if the file already exists
        if not os.path.exists(f"{DATA_PATH}/{chain}/pool/{i}_{i + step - 1}.json"):
            blocks.append((i, i + step - 1))
        else:
            continue

    print(f"TODOS: {len(blocks)}")
    return blocks


def main() -> None:
    """
    CLI entrypoint
    """

    args = parse_args()
    if not os.path.exists(f"{DATA_PATH}/{args.chain}/pool"):
        os.makedirs(f"{DATA_PATH}/{args.chain}/pool")

    INFURA_API_KEYS = str(os.getenv("INFURA_API_KEYS")).split(",")

    with multiprocessing.Manager() as manager:
        api_queue = manager.Queue()

        for api_key in INFURA_API_KEYS:
            api_queue.put(INFURA_API_BASE_DICT[args.chain] + api_key)

        blocks = split_blocks(
            UNISWAP_V3_FACTORY_DICT[args.chain]["block"],
            fetch_current_block(INFURA_API_BASE_DICT[args.chain] + INFURA_API_KEYS[0]),
            UNISWAP_V3_FACTORY_DICT[args.chain]["step"],
            args.chain,
        )

        num_processes = min(len(INFURA_API_KEYS), os.cpu_count())
        with multiprocessing.Pool(processes=num_processes) as pool:
            partial_process = partial(fetch_new_pools_wrap, args.chain, api_queue)
            for _ in tqdm(
                pool.imap_unordered(partial_process, blocks),
                total=len(blocks),
            ):
                pass


if __name__ == "__main__":
    main()
