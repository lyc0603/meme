"""
Script to fetch smart contract
"""

import argparse
import glob
import os
from tqdm import tqdm

from dotenv import load_dotenv

from environ.constants import PROCESSED_DATA_PATH, TRUMP_BLOCK, NATIVE_ADDRESS_DICT
from environ.contract_fetcher import TokenContractFecther
from environ.db import fetch_native_pool_since_block
from environ.data_class import NewTokenPool

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
            for path in glob.glob(f"{PROCESSED_DATA_PATH}/smart_contract/{chain}/*.pkl")
        ]
    )
    return [pool for pool in pools if pool["args"]["pool"] not in finished_pools]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description="Pool Fetcher CLI")
    parser.add_argument(
        "--chain",
        type=str,
        default="base",
        help="The chain to fetch data from (e.g., polygon).",
    )
    return parser.parse_args()


if __name__ == "__main__":

    arg_parser = parse_args()
    chain_name = arg_parser.chain
    os.makedirs(f"{PROCESSED_DATA_PATH}/smart_contract/{chain_name}", exist_ok=True)
    pools = get_todo_list(chain_name)

    for pool in tqdm(pools):
        try:
            args = pool["args"]
            contract = TokenContractFecther(
                NewTokenPool(
                    token0=args["token0"],
                    token1=args["token1"],
                    fee=args["fee"],
                    pool_add=args["pool"],
                    block_number=pool["blockNumber"],
                    chain=chain_name,
                    base_token=(
                        args["token0"]
                        if args["token0"] != NATIVE_ADDRESS_DICT[chain_name]
                        else args["token1"]
                    ),
                    quote_token=(
                        args["token1"]
                        if args["token1"] != NATIVE_ADDRESS_DICT[chain_name]
                        else args["token0"]
                    ),
                    txns={},
                ),
            )
            contract.save_contract()
        except Exception as e:
            print(f"Error fetching contract data for pool {args['pool']}: {e}")
