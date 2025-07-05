"""Script to process pool data in evalution"""

import argparse
import glob
import multiprocessing
import os
from functools import partial

from dotenv import load_dotenv
from tqdm import tqdm

from environ.constants import NATIVE_ADDRESS_DICT, PROCESSED_DATA_PATH, TRUMP_BLOCK
from environ.data_class import NewTokenPool
from environ.db import fetch_native_pool_since_block
from environ.txn_monitor import TxnMonitor

load_dotenv()


def get_todo_list(
    chain: str,
) -> list:
    """
    Get the todo list for the pool
    """

    pools = fetch_native_pool_since_block(chain, TRUMP_BLOCK[chain])
    finished_pools = set(
        [
            os.path.basename(path).split(".")[0]
            for path in glob.glob(f"{PROCESSED_DATA_PATH}/txn/{chain}/*.pkl")
        ]
    )
    return [pool for pool in pools if pool["args"]["pool"] not in finished_pools]


def fetch_txn_concurrently(
    pool: dict, http_queue: multiprocessing.Queue, chain: str
) -> None:
    """
    Fetch the transaction data concurrently
    """
    args = pool["args"]
    http = http_queue.get()
    try:
        txn = TxnMonitor(
            http,
            NewTokenPool(
                token0=args["token0"],
                token1=args["token1"],
                fee=args["fee"],
                pool_add=args["pool"],
                block_number=pool["blockNumber"],
                chain=chain,
                base_token=(
                    args["token0"]
                    if args["token0"] != NATIVE_ADDRESS_DICT[chain]
                    else args["token1"]
                ),
                quote_token=(
                    args["token0"]
                    if args["token0"] == NATIVE_ADDRESS_DICT[chain]
                    else args["token1"]
                ),
                txns={},
            ),
        )
        txn.aggregate()
    except Exception as e:
        print(f"Error fetching transaction data for pool {args['pool']}: {e}")
    finally:
        http_queue.put(http)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description="Pool Fetcher CLI")
    parser.add_argument(
        "--chain",
        type=str,
        default="bnb",
        help="The chain to fetch data from (e.g., polygon).",
    )
    return parser.parse_args()


if __name__ == "__main__":

    arg_parser = parse_args()
    chain_name = arg_parser.chain
    pools = get_todo_list(chain_name)
    INFURA_API_KEYS = str(os.getenv("INFURA_API_KEYS")).split(",")

    with multiprocessing.Manager() as manager:
        api_queue = manager.Queue()
        for api_key in INFURA_API_KEYS:
            api_queue.put(api_key)
        partial_process = partial(
            fetch_txn_concurrently,
            http_queue=api_queue,
            chain=chain_name,
        )
        with multiprocessing.Pool(
            processes=min(len(INFURA_API_KEYS), os.cpu_count())
        ) as con_pool:
            for _ in tqdm(
                con_pool.imap_unordered(
                    partial_process,
                    pools,
                ),
                total=len(pools),
            ):
                pass
