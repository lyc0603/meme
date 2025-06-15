"""Script to generate a list of bundles"""

import time
import json
from glob import glob
import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Generator

import dotenv
from flipside import Flipside
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH, SOL_TOKEN_ADDRESS
from environ.data_class import NewTokenPool
from environ.meme_analyzer import MemeAnalyzer
from environ.sol_fetcher import import_pool

# Config
NUM_OF_OBSERVATIONS = 1_000
dotenv.load_dotenv()
FLIPSIDE_API_KEY = "4db69d1d-d55a-497c-82e4-93933c22e5f1"
FLIPSIDE_BASE_URL = "https://api-v2.flipsidecrypto.xyz"

CHAIN = "raydium"
QUERY_INTERVAL = timedelta(days=1)

default_retry = retry(
    reraise=True,
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)

bundle_dict = {}
# for pool in tqdm(
#     import_pool(
#         CHAIN,
#         NUM_OF_OBSERVATIONS,
#     )
# ):
#     pool_add = pool["token_address"]
#     meme = MemeAnalyzer(
#         NewTokenPool(
#             token0=SOL_TOKEN_ADDRESS,
#             token1=pool_add,
#             fee=0,
#             pool_add=pool_add,
#             block_number=0,
#             chain=CHAIN,
#             base_token=pool_add,
#             quote_token=SOL_TOKEN_ADDRESS,
#             txns={},
#         ),
#     )
#     txn_same_block = meme.get_txn_same_block()
#     bundle_dict[pool_add] = {
#         "maker": meme.creator,
#         "launch_time": meme.launch_time,
#         "migration_time": meme.block_created_time,
#         "bundle": txn_same_block,
#     }

# with open(PROCESSED_DATA_PATH / "bundle.pkl", "wb") as f:
#     pickle.dump(bundle_dict, f)


def bundle_query(wallet_list: list[str], starting_time: str, ending_time: str) -> str:
    """Generate a query to fetch transfers involving a list of wallets."""
    all_wallets = wallet_list

    wallet_values = ""
    for i, wallet in enumerate(all_wallets):
        if i == len(all_wallets) - 1:
            wallet_values += f"('{wallet}')"
        else:
            wallet_values += f"('{wallet}'),\n    "

    query = f"""
WITH addr_set(address) AS (
  SELECT column1 FROM (VALUES
    {wallet_values}
  ) AS t(column1)
)

SELECT ft.*
FROM solana.core.fact_transfers ft
LEFT JOIN addr_set af ON ft.tx_from = af.address
LEFT JOIN addr_set at ON ft.tx_to = at.address
WHERE (af.address IS NOT NULL OR at.address IS NOT NULL)
  AND ft.block_timestamp BETWEEN TIMESTAMP '{starting_time}' AND TIMESTAMP '{ending_time}'
ORDER BY ft.block_timestamp ASC;
"""
    return query


def datetime_range(
    start: datetime, end: datetime, step: timedelta
) -> Generator[tuple[datetime, datetime], None, None]:
    """Generate a range of datetime intervals from start to end with a given step."""
    current = start
    while current < end:
        interval_end = min(end, current + step)
        yield current, interval_end
        current = interval_end + timedelta(seconds=1)  # Skip overlap


@default_retry
def fetch_data(
    query: str,
    page_number: int,
    **query_params: str | int | Any,
) -> Any:
    """Fetch transactions before 12 hours since the migration."""
    return Flipside(str(FLIPSIDE_API_KEY), FLIPSIDE_BASE_URL).query(
        query.format(**query_params),
        page_number=page_number,
    )


def save_bundle_data(data_lst: list[dict], token_add: str):
    """Save the bundle data to a JSONL file."""
    out_path = PROCESSED_DATA_PATH / "bundle" / f"{token_add}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in data_lst:
            f.write(json.dumps(row) + "\n")


with open(PROCESSED_DATA_PATH / "bundle.pkl", "rb") as f:
    bundle_dict = pickle.load(f)

os.makedirs(PROCESSED_DATA_PATH / "bundle", exist_ok=True)
done_tasks = [
    os.path.splitext(os.path.basename(file))[0]
    for file in glob(str(PROCESSED_DATA_PATH / "bundle" / "*.jsonl"))
]


for token_address, bundle_info in bundle_dict.items():

    if token_address in done_tasks:
        continue

    wallet_set = set([bundle_info["maker"]])
    for block, bundle in bundle_info["bundle"].items():
        wallet_set.update([_["maker"] for _ in bundle])

    launch_time = bundle_info["launch_time"] - timedelta(hours=1)
    migration_time = bundle_info["migration_time"]

    full_data = []
    time_ranges = list(datetime_range(launch_time, migration_time, QUERY_INTERVAL))

    try:
        for start_time, end_time in tqdm(
            time_ranges, desc=f"Processing {token_address}", leave=False
        ):

            query = bundle_query(
                wallet_list=list(wallet_set),
                starting_time=datetime.strftime(start_time, "%Y-%m-%d %H:%M:%S"),
                ending_time=datetime.strftime(end_time, "%Y-%m-%d %H:%M:%S"),
            )
            current_page = 0
            total_pages = 1
            while current_page < total_pages:
                data = fetch_data(query=query, page_number=current_page)
                if data.records:
                    full_data.extend(data.records)
                    total_pages = data.page.totalPages
                    current_page += 1
                else:
                    break
        save_bundle_data(full_data, token_address)
    except Exception as e:
        print(f"Error processing {token_address}: {e}")
        continue
