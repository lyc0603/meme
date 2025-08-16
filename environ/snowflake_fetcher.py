"""Class to fetch Solana data from the Solana API."""

import time
import random
import argparse
import json
import os
import pickle
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Literal, Optional
from snowflake.connector import connect, DictCursor

import dotenv
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from environ.constants import DATA_PATH, HEADERS, PROCESSED_DATA_PATH

dotenv.load_dotenv()
default_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)

os.makedirs(DATA_PATH / "solana", exist_ok=True)

SOLANA_PATH_DICT = {
    "pumpfun": DATA_PATH / "solana" / "pumpfun.jsonl",
    "raydium": DATA_PATH / "solana" / "raydium.jsonl",
    "pre_trump_raydium": DATA_PATH / "solana" / "pre_trump_raydium.jsonl",
    "pre_trump_pumpfun": DATA_PATH / "solana" / "pre_trump_pumpfun.jsonl",
}

DEX_DICT = {
    "raydium": "Raydium Liquidity Pool V4",
    "pumpfun": "pump.fun",
}


def lower_case_key(d: dict) -> dict:
    """Convert all keys in a dictionary to lowercase."""
    new_d = {}
    for k, v in d.items():
        new_d[k.lower()] = v

    return new_d


def import_pool(
    category: Literal["pumpfun", "raydium", "pre_trump_raydium", "pre_trump_pumpfun"],
    num: Optional[int] = None,
) -> list[tuple[str, str | int | Any]]:
    """Utility function to fetch the pool list."""

    pool = []
    with open(
        SOLANA_PATH_DICT[category],
        "r",
        encoding="utf-8",
    ) as f:
        for idx, line in enumerate(f, 1):
            if num:
                if idx > num:
                    break
            pool.append(json.loads(line))

    return pool


class SolanaFetcher:
    """Class to fetch Solana data from the FLIPSIDE."""

    def __init__(
        self,
        category: Literal[
            "pumpfun", "raydium", "pre_trump_raydium", "pre_trump_pumpfun"
        ],
        num: int,
        timestamp: str,
        task_query: str,
    ) -> None:

        self.cs = self.snowflake()
        self.category = category
        self.num = num
        self.timestamp = timestamp
        self.task_query = task_query

        if os.path.exists(SOLANA_PATH_DICT[category]):
            self.pool = (
                [
                    (_["token_address"], _["block_timestamp"])
                    for _ in import_pool(
                        category,
                        num,
                    )
                ]
                if self.category in ["raydium", "pre_trump_raydium"]
                else [
                    (_["token_address"], _["launch_time"])
                    for _ in import_pool(
                        category,
                        num,
                    )
                ]
            )
        else:
            self.fetch_task(
                self.task_query,
                self.timestamp,
                self.num,
                SOLANA_PATH_DICT[category],
            )
        self.cache = None

    def snowflake(self) -> Any:
        """Return the snowflake connection."""
        conn_params = {
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA"),
            "role": os.getenv("SNOWFLAKE_ROLE"),
        }
        ctx = connect(**conn_params)
        return ctx.cursor(DictCursor)

    @default_retry
    def fetch_task(
        self,
        query: str,
        timestamp: str,
        num: int,
        save_path: Path,
    ) -> None:
        """Fetch the list of meme coins"""
        self.cs.execute(
            query.format(
                timestamp=timestamp,
                num=num,
            ),
        )
        rows = self.cs.fetchall()

        self.pool = rows
        with open(
            save_path,
            "w",
            encoding="utf-8",
        ) as f:
            if isinstance(rows, Iterable):
                for row in rows:
                    row = lower_case_key(row)
                    for k in ["launch_time"]:
                        row[k] = (
                            row[k]
                            .astimezone(timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%S.000Z")
                        )
                    row["token_creator"] = row["token_creator"].strip('"')
                    f.write(json.dumps(row) + "\n")
            else:
                raise ValueError(
                    f"Invalid response type: {type(rows)}. Expected Iterable."
                )

    def parse_task(
        self,
        save_path: Path,
    ) -> list[tuple[str | int | Any, str | int | Any]]:
        """Task to be run."""

        os.makedirs(save_path, exist_ok=True)
        finished = [
            _.split("/")[-1].split(".")[0] for _ in glob(str(save_path / "*.jsonl"))
        ]

        if isinstance(self.pool, list):
            return [
                (token_add, block_ts)
                for token_add, block_ts in self.pool
                if token_add not in finished
            ]
        else:
            raise ValueError(f"Invalid pool type: {type(self.pool)}. Expected list.")

    @default_retry
    def fetch_data(
        self,
        query: str,
        **query_params: str | int | Any,
    ) -> Any:
        """Fetch transactions before 12 hours since the migration."""

        self.cs.execute(query.format(**query_params))
        return self.cs.fetchall()

    def fetch_trader_trading(self, query: str, save_path: Path) -> dict[str, Any]:
        """Fetch trader trading data for a given token address."""
        os.makedirs(PROCESSED_DATA_PATH / "trader" / self.category, exist_ok=True)

        finished = [
            _.split("/")[-1].split(".")[0] for _ in glob(str(save_path / "*.jsonl"))
        ]

        with open(
            PROCESSED_DATA_PATH / "trader" / f"{self.category}.json",
            "r",
            encoding="utf-8",
        ) as f:
            traders = json.load(f)

        for _, swapper in tqdm(traders.items(), desc="Fetching Trader Trading"):
            if swapper in finished:
                continue
            rows = self.fetch_data(
                query=query,
                swapper=swapper,
            )
            with open(
                save_path / f"{swapper}.jsonl",
                "w",
                encoding="utf-8",
            ) as f:
                if isinstance(rows, Iterable):
                    for row in rows:
                        row = lower_case_key(row)
                        for k in [
                            "block_timestamp",
                            "inserted_timestamp",
                            "modified_timestamp",
                        ]:
                            row[k] = (
                                row[k]
                                .astimezone(timezone.utc)
                                .strftime("%Y-%m-%dT%H:%M:%S.000Z")
                            )

                        tsk_list = ["swapper", "swap_from_mint", "swap_to_mint"]

                        for k in tsk_list:
                            row[k] = row[k].strip('"')
                        f.write(json.dumps(row) + "\n")
                else:
                    raise ValueError(
                        f"Invalid response type: {type(rows)}. Expected Iterable."
                    )

    @default_retry
    def fetch_reply(
        self,
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

    def fetch_replies(
        self,
        save_path: Path = DATA_PATH / "solana" / "raydium" / "reply",
    ) -> None:
        """Fetch replies from the pump.fun for all tokens in the pool."""

        for token_address, _ in tqdm(
            self.parse_task(save_path), desc="Fetching Replies"
        ):
            data_lst = []
            offset = 0
            has_more = True

            while has_more:
                try:
                    time.sleep(random.uniform(4, 5))
                    data = self.fetch_reply(token_address, 1000, offset)
                    data_lst.extend(data["replies"])
                    offset += len(data["replies"])
                    if data["hasMore"] is False:
                        has_more = False
                except Exception as e:
                    print(f"Error fetching replies for {token_address}: {e}")
                    break

            with open(
                save_path / f"{token_address}.jsonl",
                "w",
                encoding="utf-8",
            ) as f:
                for row in data_lst:
                    f.write(json.dumps(row) + "\n")

    def fetch(self, query: str, save_path: Path, typ: str = "txn") -> Any:
        """Fetch transactions before 12 hours since the migration."""

        for token_address, block_timestamp in tqdm(
            self.parse_task(save_path),
        ):
            rows = self.fetch_data(
                query=query,
                token_address=token_address,
                migration_timestamp=datetime.strptime(
                    str(block_timestamp), "%Y-%m-%dT%H:%M:%S.%fZ"
                ).strftime("%Y-%m-%d %H:%M:%S"),
            )
            with open(
                save_path / f"{token_address}.jsonl",
                "w",
                encoding="utf-8",
            ) as f:
                if isinstance(rows, Iterable):
                    for row in rows:
                        row = lower_case_key(row)
                        for k in [
                            "block_timestamp",
                            "inserted_timestamp",
                            "modified_timestamp",
                        ]:
                            row[k] = (
                                row[k]
                                .astimezone(timezone.utc)
                                .strftime("%Y-%m-%dT%H:%M:%S.000Z")
                            )

                        if typ == "txn":
                            tsk_list = ["swapper", "swap_from_mint", "swap_to_mint"]
                        else:
                            tsk_list = ["tx_from", "tx_to", "mint"]

                        for k in tsk_list:
                            row[k] = row[k].strip('"')
                        f.write(json.dumps(row) + "\n")
                else:
                    raise ValueError(
                        f"Invalid response type: {type(rows)}. Expected Iterable."
                    )
