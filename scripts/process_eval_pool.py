"""Script to process pool data in evalution"""

import os
import argparse

from tqdm import tqdm

from environ.constants import NATIVE_ADDRESS_DICT
from environ.data_class import NewTokenPool
from environ.db import client
from environ.txn_monitor import TxnMonitor

TRUMP_BLOCK = {"base": 25166580}


def fetch_native_pool_since_block(
    chain: str,
    from_block: int,
    pool_number: int = 1000,
):
    """
    Fetch the new token pool data since a block
    """
    db = client[chain]
    collection = db["pool"]

    previous_tokens = set()
    token0_before = collection.distinct(
        "args.token0", {"blockNumber": {"$lt": from_block}}
    )
    token1_before = collection.distinct(
        "args.token1", {"blockNumber": {"$lt": from_block}}
    )
    previous_tokens |= set(token0_before)
    previous_tokens |= set(token1_before)

    cursor = collection.find(
        {
            "blockNumber": {"$gte": from_block},
            "$or": [
                {
                    "args.token0": NATIVE_ADDRESS_DICT[chain],
                },
                {
                    "args.token1": NATIVE_ADDRESS_DICT[chain],
                },
            ],
        },
    ).sort([("blockNumber", 1)])

    result = []
    for doc in cursor:
        t0 = doc["args"]["token0"]
        t1 = doc["args"]["token1"]

        if (t0 not in previous_tokens) or (t1 not in previous_tokens):
            result.append(doc)
            previous_tokens.add(t0)
            previous_tokens.add(t1)

        if len(result) >= pool_number:
            break

    return result


if __name__ == "__main__":

    pools = fetch_native_pool_since_block("base", TRUMP_BLOCK["base"])

    for pool in tqdm(pools):
        args = pool["args"]
        chain = "base"
        txn = TxnMonitor(
            str(os.getenv("INFURA_API_KEYS")).rsplit(",", maxsplit=1)[-1],
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
                    args["token1"]
                    if args["token1"] != NATIVE_ADDRESS_DICT[chain]
                    else args["token0"]
                ),
                txns={},
            ),
        )

        txn.aggregate()
        txn.save_txns()
