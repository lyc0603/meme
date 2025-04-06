"""
Function to convert the data
"""

from multiprocessing import Pool
from typing import Iterable, List
from functools import partial
from environ.constants import NATIVE_ADDRESS_DICT


import pymongo
from tqdm import tqdm

N_WORKERS = 36
client = pymongo.MongoClient("mongodb://localhost:27017/")


def insert_func(doc, db_name: str, collection_name: str):
    """
    Inserts a single document into the specified collection
    """
    local_client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = local_client[db_name]
    collection = db[collection_name]
    return collection.insert_one(doc)


def insert_db(
    data: List,
    db_name: str,
    collection_name: str,
    index: List,
) -> None:
    """
    Function to insert pd.DataFrame into the mongodb database
    """
    db = client[db_name]
    collection = db[collection_name]
    collection.create_index(
        [(index_name, pymongo.ASCENDING) for index_name in index], unique=True
    )

    worker_fn = partial(insert_func, db_name=db_name, collection_name=collection_name)
    with Pool(N_WORKERS) as p:
        list(tqdm(p.imap(worker_fn, data), total=len(data)))


def load_db(
    db_name: str,
    collection_name: str,
) -> Iterable:
    """
    Function to load the data from the mongodb database
    """
    db = client[db_name]
    collection = db[collection_name]

    for document in tqdm(collection.find()):
        yield document


def fetch_native_pool_since_block(
    chain: str,
    from_block: int,
    pool_number: int = 2000,
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
    import json
    from glob import glob

    from environ.constants import DATA_PATH, UNISWAP_V3_FACTORY_DICT

    for chain, _ in UNISWAP_V3_FACTORY_DICT.items():
        file_list = glob(f"{DATA_PATH}/{chain}/pool/*.jsonl")
        pool_data = []

        for file in file_list:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    pool_data.append(json.loads(line))

        insert_db(pool_data, chain, "pool", ["args.pool"])

        with open(f"{DATA_PATH}/{chain}/time.jsonl", "r", encoding="utf-8") as f:
            time_data = sorted(
                [json.loads(line) for line in f], key=lambda x: x["date"]
            )

        insert_db(time_data, chain, "timestamp", ["date"])
