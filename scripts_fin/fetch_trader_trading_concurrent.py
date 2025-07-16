"""Fetch trading data for traders concurrently using Snowflake."""

import os
from dotenv import load_dotenv
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable
from tqdm import tqdm
from datetime import timezone
from snowflake.connector import connect, DictCursor
from environ.constants import PROCESSED_DATA_PATH

# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

SWAPPER_QUERY = """SELECT *
FROM defi.ez_dex_swaps
WHERE swapper = '{swapper}'
ORDER BY block_timestamp ASC;"""

load_dotenv()

SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    "role": os.getenv("SNOWFLAKE_ROLE"),
}

CATEGORIES = ["pre_trump_raydium", "raydium", "pumpfun", "pre_trump_pumpfun"]
BASE_DIR = Path(f"{PROCESSED_DATA_PATH}/trader")
MAX_WORKERS = 20

# -----------------------------------------
# CORE FUNCTIONS
# -----------------------------------------


def snowflake_connection():
    """Create a Snowflake connection."""
    return connect(**SNOWFLAKE_CONFIG).cursor(DictCursor)


def lower_case_key(d: dict) -> dict:
    """Convert all keys in a dictionary to lowercase."""
    return {k.lower(): v for k, v in d.items()}


def write_rows_to_file(rows: Iterable[dict], final_path: Path) -> None:
    """Write rows to a JSONL file safely."""
    temp_path = final_path.with_suffix(".jsonl.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        for row in rows:
            row = lower_case_key(row)
            for k in ["block_timestamp", "inserted_timestamp", "modified_timestamp"]:
                if k in row and row[k]:
                    row[k] = (
                        row[k]
                        .astimezone(timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%S.000Z")
                    )
            for k in ["swapper", "swap_from_mint", "swap_to_mint"]:
                if k in row and isinstance(row[k], str):
                    row[k] = row[k].strip('"')
            f.write(json.dumps(row) + "\n")
    os.rename(temp_path, final_path)


def fetch_single_trader(swapper: str, output_dir: Path) -> None:
    """Fetch trading data for a single trader and write to a file."""
    try:
        with snowflake_connection() as cursor:
            filepath = output_dir / f"{swapper}.jsonl"
            cursor.execute(SWAPPER_QUERY.format(swapper=swapper))
            rows = cursor.fetchall()
            write_rows_to_file(rows, filepath)
    except Exception as e:
        print(f"[ERROR] Swapper {swapper}: {e}")


def fetch_trader_trading_concurrent(category: str, max_workers: int = 10):
    """Fetch trading data for all traders in a category concurrently."""
    trader_json_path = BASE_DIR / f"{category}.json"
    output_dir = BASE_DIR / category
    os.makedirs(output_dir, exist_ok=True)

    finished = {p.stem for p in output_dir.glob("*.jsonl")}

    with open(trader_json_path, "r", encoding="utf-8") as f:
        traders = json.load(f)

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _, swapper in traders.items():
            if swapper not in finished:
                futures.append(
                    executor.submit(fetch_single_trader, swapper, output_dir)
                )

        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"Fetching {category}"
        ):
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] Thread failure: {e}")


# -----------------------------------------
# MAIN LOOP
# -----------------------------------------

if __name__ == "__main__":
    for category in CATEGORIES:
        fetch_trader_trading_concurrent(category, MAX_WORKERS)
