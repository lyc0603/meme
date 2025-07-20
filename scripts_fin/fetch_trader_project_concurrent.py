"""Fetch trader project data concurrently from Snowflake and save to JSONL files."""

import json
import os
import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Literal

import pandas as pd
import requests
from dotenv import load_dotenv
from snowflake.connector import DictCursor, connect
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from environ.constants import DATA_PATH, HEADERS, PROCESSED_DATA_PATH
from environ.data_class import Swap, Transfer, Txn
from environ.query import UNCON_SWAP_QUERY, UNCON_TRANSFER_QUERY, LAUNCH_QUERY_TEMPLATE
from multiprocessing import Pool, cpu_count


load_dotenv()
default_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)


# ----------------------------------------
# CONFIGURATION
# ----------------------------------------

OUTPUT_DIR = Path(f"{PROCESSED_DATA_PATH}/kol_non_kol")

SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    "role": os.getenv("SNOWFLAKE_ROLE"),
}

MAX_WORKERS = 30

# ----------------------------------------
# UTILITY FUNCTIONS
# ----------------------------------------


def snowflake_connection():
    """Create a Snowflake connection."""
    return connect(**SNOWFLAKE_CONFIG).cursor(DictCursor)


def lower_case_key(d: dict) -> dict:
    """Convert all keys in a dictionary to lowercase."""
    return {k.lower(): v for k, v in d.items()}


def write_jsonl(rows: Iterable[dict], filepath: Path, typ: str):
    """Write rows to a JSONL file with proper formatting."""

    with open(filepath, "w", encoding="utf-8") as f:
        for row in rows:
            row = lower_case_key(row)

            if typ == "launch":
                keys = ["launch_time"]
            else:
                keys = ["block_timestamp", "inserted_timestamp", "modified_timestamp"]

            for k in keys:
                if k in row and row[k]:
                    row[k] = (
                        row[k]
                        .astimezone(timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%S.000Z")
                    )

            if typ == "txn":
                keys = ["swapper", "swap_from_mint", "swap_to_mint"]
            elif typ == "transfer":
                keys = ["tx_from", "tx_to", "mint"]
            else:
                keys = ["token_address", "token_creator", "pumpfun_pool_address"]
            for k in keys:
                if k in row and isinstance(row[k], str):
                    row[k] = row[k].strip('"')
            f.write(json.dumps(row) + "\n")


def chunk_list(lst, n):
    """Split list `lst` into `n` approximately equal chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def worker_chunk(chunk: list[str], query: str, typ: str, out_dir: Path):
    """Worker that logs in once and processes a chunk of token addresses."""
    os.makedirs(out_dir, exist_ok=True)
    try:
        with snowflake_connection() as cursor:
            for token_address in chunk:
                filepath = out_dir / f"{token_address}.jsonl"
                if filepath.exists():
                    continue
                try:
                    cursor.execute(query.format(token_address=token_address))
                    rows = cursor.fetchall()
                    write_jsonl(rows, filepath, typ)
                except Exception as e:
                    print(f"[ERROR] {token_address}: {e}")
    except Exception as e:
        print(f"[LOGIN ERROR] Failed to connect for chunk: {e}")


def fetch_all_in_chunks(token_list: list[str], query: str, out_dir: Path, typ: str):
    """Fetch data for all tokens in the list concurrently, splitting into chunks."""
    chunks = chunk_list(token_list, MAX_WORKERS)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for chunk in chunks:
            futures.append(executor.submit(worker_chunk, chunk, query, typ, out_dir))
            time.sleep(5)  # Wait 5 seconds between task submissions

        for _ in tqdm(
            as_completed(futures), total=len(futures), desc=f"Fetching {typ}"
        ):
            pass


def fetch_all_launches_in_chunks(token_list: list[str]):
    """Fetch launch data for all tokens in the list concurrently, splitting into chunks."""
    chunks = chunk_list(token_list, MAX_WORKERS)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        for i, chunk in enumerate(chunks):
            futures.append(executor.submit(fetch_launch_chunk, chunk, i))
            time.sleep(5)  # Wait 5 seconds between task submissions

        for _ in tqdm(
            as_completed(futures), total=len(futures), desc="Fetching Launches"
        ):
            pass


# def fetch_and_save(token_address: str, query: str, typ: str, out_dir: Path):
#     """Fetch data for a single token and save to a file."""
#     filepath = out_dir / f"{token_address}.jsonl"
#     if filepath.exists():
#         return

#     try:
#         with snowflake_connection() as cursor:
#             cursor.execute(query.format(token_address=token_address))
#             rows = cursor.fetchall()
#             write_jsonl(rows, filepath, typ)
#     except Exception as e:
#         print(f"[ERROR] {token_address}: {e}")


# def fetch_all(token_list: list[dict], query: str, out_dir: Path, typ: str):
#     """Fetch data for all tokens in the list concurrently."""
#     os.makedirs(out_dir, exist_ok=True)
#     futures = []
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         for t in token_list:
#             futures.append(executor.submit(fetch_and_save, t, query, typ, out_dir))
#             time.sleep(5)  # wait 5 seconds after each submission

#         for _ in tqdm(
#             as_completed(futures), total=len(futures), desc=f"Fetching {typ}"
#         ):
#             pass


def fetch_launch_chunk(chunk: list[str], chunk_id: int):
    """Fetch launch data for a chunk and save it to a single file named by chunk ID."""
    out_dir = OUTPUT_DIR / "creation"
    os.makedirs(out_dir, exist_ok=True)

    address_list_str = ",".join(f"'{addr}'" for addr in chunk)
    query = LAUNCH_QUERY_TEMPLATE.format(address_list=address_list_str)

    try:
        with snowflake_connection() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

        write_jsonl(rows, out_dir / f"{chunk_id}.jsonl", "launch")

    except Exception as e:
        print(f"[CHUNK ERROR] Chunk {chunk_id}: {e}")


def fetch_replies(token_list: list[dict], out_dir: Path):
    """Fetch replies for a single token and save to a file."""
    os.makedirs(out_dir, exist_ok=True)
    for token_address in tqdm(token_list, desc="Fetching Replies"):
        data_lst = []
        offset = 0
        has_more = True

        while has_more:
            try:
                time.sleep(random.uniform(3, 4))
                data = fetch_reply(token_address, 1000, offset)
                data_lst.extend(data["replies"])
                offset += len(data["replies"])
                if data["hasMore"] is False:
                    has_more = False
            except Exception as e:
                print(f"Error fetching replies for {token_address}: {e}")
                break

        with open(
            out_dir / f"{token_address}.jsonl",
            "w",
            encoding="utf-8",
        ) as f:
            for row in data_lst:
                f.write(json.dumps(row) + "\n")


@default_retry
def fetch_reply(
    token_address: str | int | Any,
    limit: int,
    offset: int = 0,
) -> dict[str, Any]:
    """Fetch replies from the pump.fun for a given token address."""

    url = (
        f"https://frontend-api-v3.pump.fun/replies/{token_address}"
        f"?limit={limit}&offset={offset}"
        "&user=B3vZuHWgsQqLctSvkexF1TzfBmo8EKAqGJpqvZwbeREH"
    )
    return requests.get(
        url,
        headers=HEADERS,
        timeout=60,
    ).json()


def process_creation() -> None:
    """Process the transactions data in the pool"""

    os.makedirs(
        PROCESSED_DATA_PATH / "learning" / "creation",
        exist_ok=True,
    )

    load_path = glob(str(PROCESSED_DATA_PATH / "kol_non_kol" / "creation" / "*.jsonl"))
    for path in tqdm(load_path, desc="Processing Creation Data"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                pool_info = json.loads(line)
                token_add = pool_info["token_address"]
                migrate_ts = (
                    int(
                        datetime.strptime(
                            str(pool_info["block_timestamp"]), "%Y-%m-%dT%H:%M:%S.%fZ"
                        ).timestamp()
                    )
                    if "block_timestamp" in pool_info
                    else None
                )
                launch_ts = int(
                    datetime.strptime(
                        str(pool_info["launch_time"]), "%Y-%m-%dT%H:%M:%S.%fZ"
                    ).timestamp()
                )
                # processed the creation data
                with open(
                    PROCESSED_DATA_PATH / "learning" / "creation" / f"{token_add}.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        {
                            "migrate_time": migrate_ts,
                            "migrate_block": (
                                int(pool_info["block_id"])
                                if "block_id" in pool_info
                                else None
                            ),
                            "launch_time": launch_ts,
                            "launch_block": int(pool_info["launch_block_id"]),
                            "token_creator": pool_info["token_creator"],
                            "pumpfun_pool_address": pool_info["pumpfun_pool_address"],
                            "launch_tx_id": pool_info["launch_tx_id"],
                        },
                        f,
                        indent=4,
                    )


def process_single_txn(token_add: str) -> None:
    """Process and save one token's transaction data."""
    txn_lst = []

    for possible_path in [
        PROCESSED_DATA_PATH / "kol_non_kol",
        DATA_PATH / "solana" / "pre_trump_pumpfun",
        DATA_PATH / "solana" / "pre_trump_raydium",
        DATA_PATH / "solana" / "pumpfun",
        DATA_PATH / "solana" / "raydium",
    ]:
        txn_path = possible_path / "txn" / f"{token_add}.jsonl"
        if os.path.exists(txn_path):
            break
    else:
        # If no path exists, skip this token
        print(f"[SKIP] {token_add}: No transaction data found.")

    try:
        with open(txn_path, "r", encoding="utf-8") as f:
            for line in f:
                txn = json.loads(line)

                if txn["swap_from_amount"] * txn["swap_to_amount"] == 0:
                    continue
                if (txn["swap_from_symbol"] != "SOL") & (
                    txn["swap_to_symbol"] != "SOL"
                ):
                    continue

                txn_lst.append(
                    Txn(
                        date=datetime.strptime(
                            txn["block_timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ"
                        ),
                        block=txn["block_id"],
                        txn_hash=txn["tx_id"],
                        maker=txn["swapper"],
                        acts={
                            0: Swap(
                                block=txn["block_id"],
                                txn_hash=txn["tx_id"],
                                log_index=0,
                                typ=(
                                    "Buy"
                                    if txn["swap_from_symbol"] == "SOL"
                                    else "Sell"
                                ),
                                usd=txn["swap_from_amount_usd"]
                                or txn["swap_to_amount_usd"],
                                base=(
                                    txn["swap_to_amount"]
                                    if txn["swap_from_symbol"] == "SOL"
                                    else txn["swap_from_amount"]
                                ),
                                quote=(
                                    txn["swap_from_amount"]
                                    if txn["swap_from_symbol"] == "SOL"
                                    else txn["swap_to_amount"]
                                ),
                                price=(
                                    txn["swap_from_amount_usd"] / txn["swap_to_amount"]
                                    if txn["swap_from_symbol"] == "SOL"
                                    else txn["swap_to_amount_usd"]
                                    / txn["swap_from_amount"]
                                ),
                                dex=txn["swap_program"],
                            )
                        },
                    )
                )
    except Exception as e:
        print(f"[ERROR] {token_add}: {e}")
        return

    os.makedirs(PROCESSED_DATA_PATH / "learning" / "txn", exist_ok=True)
    with open(PROCESSED_DATA_PATH / "learning" / "txn" / f"{token_add}.pkl", "wb") as f:
        pickle.dump(txn_lst, f)


# ----------------------------------------
# MAIN
# ----------------------------------------

if __name__ == "__main__":

    trader_t = pd.read_csv(PROCESSED_DATA_PATH / "trader_t_stats.csv")
    finished = set(trader_t["token_address"].unique())

    if os.path.exists(OUTPUT_DIR / "txn"):
        fetch_finished_list = glob(str(OUTPUT_DIR / "txn" / "*.jsonl"))
        finished.update(
            [Path(f).stem for f in fetch_finished_list if Path(f).is_file()]
        )

    with open(
        PROCESSED_DATA_PATH / "kol_non_kol_traded_tokens.json", "r", encoding="utf-8"
    ) as f:
        token_list = json.load(f)

    all_token_list = set([_ for k, v in token_list.items() for _ in v])
    token_list = set(all_token_list)

    # fetch_all_launches_in_chunks(list(token_list))
    # fetch_all_in_chunks(
    #     list(token_list), UNCON_SWAP_QUERY, OUTPUT_DIR / "txn", typ="txn"
    # )
    # fetch_all(token_list, UNCON_SWAP_QUERY, OUTPUT_DIR / "txn", typ="txn")
    # fetch_all(token_list, UNCON_TRANSFER_QUERY, OUTPUT_DIR / "transfer", typ="transfer")
    # fetch_replies(
    #     token_list,
    #     OUTPUT_DIR / "replies",
    # )
    # with Pool(cpu_count() - 10) as pool:
    #     list(
    #         tqdm(
    #             pool.imap_unordered(process_single_txn, all_token_list),
    #             total=len(all_token_list),
    #         )
    #     )
    # process_creation()
