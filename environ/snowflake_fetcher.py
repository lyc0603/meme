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
import numpy as np
from environ.query import (
    SWAPPER_QUERY,
    UNCON_SWAP_QUERY,
    UNCON_TRANSFER_QUERY,
    SWAP_QUERY,
    TRANSFER_QUERY,
    LAUNCH_QUERY,
    MIGRATION_QUERY,
)

import dotenv
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from environ.constants import DATA_PATH, HEADERS, PROCESSED_DATA_PATH
from environ.data_class import Swap, Transfer, Txn

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


def process_txn(
    category: Literal["pumpfun", "raydium", "pre_trump_raydium", "pre_trump_pumpfun"],
    num: int = 1000,
) -> None:
    """Process the transactions data in the pool"""

    for save_path in ["txn", "creation", "transfer"]:
        os.makedirs(
            PROCESSED_DATA_PATH / save_path / category,
            exist_ok=True,
        )

    for pool_info in tqdm(import_pool(category, num=num)):
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
            PROCESSED_DATA_PATH / "creation" / category / f"{token_add}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "migrate_time": migrate_ts,
                    "migrate_block": (
                        int(pool_info["block_id"]) if "block_id" in pool_info else None
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

        # processed transfer data
        with open(
            DATA_PATH / "solana" / category / "transfer" / f"{token_add}.jsonl",
            "r",
            encoding="utf-8",
        ) as f:
            transfer_lst = []
            for line in f:
                transfer = json.loads(line)
                transfer_lst.append(
                    Transfer(
                        date=datetime.strptime(
                            transfer["block_timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ"
                        ),
                        block=transfer["block_id"],
                        txn_hash=transfer["tx_id"],
                        log_index=transfer["index"],
                        from_=transfer["tx_from"],
                        to=transfer["tx_to"],
                        value=transfer["amount"],
                    )
                )

        with open(
            PROCESSED_DATA_PATH / "transfer" / category / f"{token_add}.pkl",
            "wb",
        ) as f:
            pickle.dump(transfer_lst, f)

        # processed transaction data
        with open(
            DATA_PATH / "solana" / category / "txn" / f"{token_add}.jsonl",
            "r",
            encoding="utf-8",
        ) as f:
            txn_lst = []
            for line in f:
                txn = json.loads(line)

                # special case for the swap
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
                                usd=(
                                    txn["swap_from_amount_usd"]
                                    if txn["swap_from_amount_usd"]
                                    else txn["swap_to_amount_usd"]
                                ),
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

        with open(
            PROCESSED_DATA_PATH / "txn" / category / f"{token_add}.pkl",
            "wb",
        ) as f:
            pickle.dump(txn_lst, f)


# def parse_args() -> argparse.Namespace:
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(description="Solana Fetcher")
#     parser.add_argument(
#         "--num",
#         type=int,
#         default=100,
#         help="Number of tokens to fetch",
#     )
#     parser.add_argument(
#         "--timestamp",
#         type=str,
#         default="2025-01-17 14:01:48",
#         help="Timestamp to start fetching from",
#     )
#     return parser.parse_args()


# def main() -> None:
#     """Main function to run the Solana fetcher."""

#     args = parse_args()

#     # Fetch the pumpfun meme coins
#     pumpfun_fetcher = SolanaFetcher(
#         category="pumpfun",
#         num=args.num,
#         timestamp=args.timestamp,
#         task_query=LAUNCH_QUERY,
#     )
#     pumpfun_fetcher.fetch(SWAP_QUERY, DATA_PATH / "solana" / "pumpfun" / "txn")
#     pumpfun_fetcher.fetch(TRANSFER_QUERY, DATA_PATH / "solana" / "pumpfun" / "transfer")

#     # Fetch the raydium migration meme coins
#     raydium_fetcher = SolanaFetcher(
#         category="raydium",
#         num=args.num,
#         timestamp=args.timestamp,
#         task_query=MIGRATION_QUERY,
#     )
#     raydium_fetcher.fetch(SWAP_QUERY, DATA_PATH / "solana" / "raydium" / "txn")
#     raydium_fetcher.fetch(TRANSFER_QUERY, DATA_PATH / "solana" / "raydium" / "transfer")

#     # Process the transactions data
#     for category in ["pumpfun", "raydium"]:
#         process_txn(category)


if __name__ == "__main__":

    # pre-trump raydium fetcher
    solana_fetcher = SolanaFetcher(
        category="pre_trump_raydium",
        num=1000,
        timestamp="2024-10-17 14:01:48",
        task_query=MIGRATION_QUERY,
    )
    # solana_fetcher.fetch_task(
    #     MIGRATION_QUERY,
    #     "2024-10-17 14:01:48",
    #     1000,
    #     DATA_PATH / "solana" / "pre_trump_raydium.jsonl",
    # )
    # solana_fetcher.fetch(SWAP_QUERY, DATA_PATH / "solana" / "pre_trump_raydium" / "txn")
    # solana_fetcher.fetch(
    #     TRANSFER_QUERY, DATA_PATH / "solana" / "pre_trump_raydium" / "transfer"
    # )
    # solana_fetcher.fetch_replies(
    #     save_path=DATA_PATH / "solana" / "pre_trump_raydium" / "reply",
    # )
    solana_fetcher.fetch_trader_trading(
        query=SWAPPER_QUERY,
        save_path=PROCESSED_DATA_PATH / "trader" / "pre_trump_raydium",
    )

    # pre-trump pumpfun fetcher
    solana_fetcher = SolanaFetcher(
        category="pre_trump_pumpfun",
        num=3000,
        timestamp="2024-10-17 14:01:48",
        task_query=LAUNCH_QUERY,
    )
    # solana_fetcher.fetch_task(
    #     LAUNCH_QUERY,
    #     "2024-10-17 14:01:48",
    #     3000,
    #     DATA_PATH / "solana" / "pre_trump_pumpfun.jsonl",
    # )
    # solana_fetcher.fetch(
    #     UNCON_SWAP_QUERY, DATA_PATH / "solana" / "pre_trump_pumpfun" / "txn", typ="txn"
    # )
    # solana_fetcher.fetch(
    #     UNCON_TRANSFER_QUERY,
    #     DATA_PATH / "solana" / "pre_trump_pumpfun" / "transfer",
    #     typ="transfer",
    # )
    # solana_fetcher.fetch_replies(
    #     save_path=DATA_PATH / "solana" / "pre_trump_pumpfun" / "reply",
    # )
    solana_fetcher.fetch_trader_trading(
        query=SWAPPER_QUERY,
        save_path=PROCESSED_DATA_PATH / "trader" / "pre_trump_pumpfun",
    )

    # post-trump pumpfun fetcher
    solana_fetcher = SolanaFetcher(
        category="pumpfun",
        num=3000,
        timestamp="2025-01-17 14:01:48",
        task_query=LAUNCH_QUERY,
    )
    # solana_fetcher.fetch_task(
    #     LAUNCH_QUERY,
    #     "2025-01-17 14:01:48",
    #     3000,
    #     DATA_PATH / "solana" / "pumpfun.jsonl",
    # )
    # solana_fetcher.fetch(UNCON_SWAP_QUERY, DATA_PATH / "solana" / "pumpfun" / "txn", typ="txn")
    # solana_fetcher.fetch(
    #     UNCON_TRANSFER_QUERY,
    #     DATA_PATH / "solana" / "pumpfun" / "transfer",
    #     typ="transfer",
    # )
    # solana_fetcher.fetch_replies(
    #     save_path=DATA_PATH / "solana" / "pumpfun" / "reply",
    # )
    solana_fetcher.fetch_trader_trading(
        query=SWAPPER_QUERY,
        save_path=PROCESSED_DATA_PATH / "trader" / "pumpfun",
    )

    solana_fetcher = SolanaFetcher(
        category="raydium",
        num=1000,
        timestamp="2025-01-17 14:01:48",
        task_query=MIGRATION_QUERY,
    )
    # solana_fetcher.fetch_task(
    #     MIGRATION_QUERY,
    #     "2025-01-17 14:01:48",
    #     1000,
    #     DATA_PATH / "solana" / "raydium.jsonl",
    # )
    # solana_fetcher.fetch(SWAP_QUERY, DATA_PATH / "solana" / "raydium" / "txn")
    # solana_fetcher.fetch(TRANSFER_QUERY, DATA_PATH / "solana" / "raydium" / "transfer")
    # solana_fetcher.fetch_replies(
    #     save_path=DATA_PATH / "solana" / "raydium" / "reply",
    # )
    solana_fetcher.fetch_trader_trading(
        query=SWAPPER_QUERY,
        save_path=PROCESSED_DATA_PATH / "trader" / "raydium",
    )

    # for cate, n in [("pumpfun", 3000)]:
    #     process_txn(cate, n)
