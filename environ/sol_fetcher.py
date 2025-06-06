"""Class to fetch Solana data from the Solana API."""

import time
import random
import argparse
import json
import os
import pickle
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import dotenv
import requests
from flipside import Flipside
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from environ.constants import DATA_PATH, HEADERS, PROCESSED_DATA_PATH
from environ.data_class import Swap, Transfer, Txn

dotenv.load_dotenv()
FLIPSIDE_API_KEY = os.getenv("FLIPSIDE_API")
FLIPSIDE_BASE_URL = "https://api-v2.flipsidecrypto.xyz"
default_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)

os.makedirs(DATA_PATH / "solana", exist_ok=True)

SOLANA_PATH_DICT = {
    "pumpfun": DATA_PATH / "solana" / "pumpfun.jsonl",
    "raydium": DATA_PATH / "solana" / "raydium.jsonl",
}

DEX_DICT = {
    "raydium": "Raydium Liquidity Pool V4",
    "pumpfun": "pump.fun",
}


def import_pool(
    category: Literal["pumpfun", "raydium"], num: Optional[int] = None
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
        category: Literal["pumpfun", "raydium"],
        num: int,
        timestamp: str,
        task_query: str,
    ) -> None:

        self.flipside = Flipside(str(FLIPSIDE_API_KEY), FLIPSIDE_BASE_URL)
        self.category = category
        self.num = num
        self.timestamp = timestamp
        self.task_query = task_query

        if os.path.exists(SOLANA_PATH_DICT[category]):
            self.pool = [
                (_["token_address"], _["block_timestamp"])
                for _ in import_pool(
                    category,
                    num,
                )
            ]
        else:
            self.fetch_task(
                self.task_query,
                self.timestamp,
                self.num,
                SOLANA_PATH_DICT[category],
            )
        self.cache = None

    @default_retry
    def fetch_task(
        self,
        query: str,
        timestamp: str,
        num: int,
        save_path: Path,
    ) -> None:
        """Fetch the list of meme coins"""
        res = self.flipside.query(
            query.format(
                timestamp=timestamp,
                num=num,
            ),
        )

        self.pool = res.records
        with open(
            save_path,
            "w",
            encoding="utf-8",
        ) as f:
            if isinstance(res.records, Iterable):
                for row in res.records:
                    f.write(json.dumps(row) + "\n")
            else:
                raise ValueError(
                    f"Invalid response type: {type(res.records)}. Expected Iterable."
                )

    def fetch_personal_transfer(self) -> None:
        """Fetch personal transfer data."""

        save_path = DATA_PATH / "solana" / self.category / "transfer"
        os.makedirs(save_path, exist_ok=True)

        for token_address, _ in tqdm(
            self.parse_task(save_path), desc="Fetching Personal Transfers"
        ):
            try:
                time.sleep(random.uniform(3, 4))
                data = self.flipside.query(
                    f"""
                    SELECT *
                    FROM solana.core.fact_transfers
                    WHERE mint = '{token_address}'
                    ORDER BY block_timestamp ASC;
                    """
                )

                with open(
                    save_path / f"{token_address}.jsonl",
                    "w",
                    encoding="utf-8",
                ) as f:
                    for row in data.records:
                        f.write(json.dumps(row) + "\n")
            except Exception as e:
                print(f"Error fetching personal transfers for {token_address}: {e}")

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
        page_number: int,
        **query_params: str | int | Any,
    ) -> Any:
        """Fetch transactions before 12 hours since the migration."""
        return self.flipside.query(
            query.format(**query_params),
            page_number=page_number,
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
                    time.sleep(random.uniform(3, 4))
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

    def fetch(self, query: str, save_path: Path) -> Any:
        """Fetch transactions before 12 hours since the migration."""

        for token_address, block_timestamp in tqdm(
            self.parse_task(save_path),
        ):

            data_lst = []

            current_page_number = 0
            total_pages = 1

            try:
                while current_page_number < total_pages:
                    current_page_number += 1
                    self.cache = self.fetch_data(
                        query,
                        current_page_number,
                        token_address,
                        datetime.strptime(
                            str(block_timestamp), "%Y-%m-%dT%H:%M:%S.%fZ"
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    data_lst.extend(self.cache.records)
                    total_pages = self.cache.page.totalPages

                    if self.cache.status != "QUERY_STATE_SUCCESS":
                        raise Exception(
                            f"Query failed with status: {self.cache.status}"
                        )

                # save as jsonl
                with open(
                    save_path / f"{token_address}.jsonl",
                    "w",
                    encoding="utf-8",
                ) as f:
                    for row in data_lst:
                        f.write(json.dumps(row) + "\n")
            except Exception as e:
                print(f"Error fetching data for {token_address}: {e}")


SWAP_QUERY = """SELECT *
FROM solana.defi.ez_dex_swaps
WHERE (
         swap_from_mint = '{token_address}'
      OR swap_to_mint = '{token_address}'
      )
  AND block_timestamp < TIMESTAMP '{migration_timestamp}' + INTERVAL '12 hours'
ORDER BY block_timestamp ASC;"""

TRANSFER_QUERY = """SELECT *
FROM solana.core.fact_transfers
WHERE (
         mint = '{token_address}'
      )
  AND block_timestamp < TIMESTAMP '{migration_timestamp}' + INTERVAL '12 hours'
ORDER BY block_timestamp ASC;"""

LAUNCH_QUERY = """SELECT
  block_timestamp,
  tx_id,
  decoded_instruction:accounts[0]:pubkey::string AS token_address,
  signers[0] AS token_creator,
  decoded_instruction:accounts[2]:pubkey::string AS pumpfun_pool_address
FROM
  solana.core.ez_events_decoded
WHERE
  program_id = '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P'
  AND event_type = 'create'
  AND block_timestamp > TIMESTAMP '{timestamp}'
ORDER BY block_timestamp
LIMIT {num};"""

MIGRATION_QUERY = """SELECT
  mig.block_timestamp AS block_timestamp,
  mig.tx_id AS tx_id,
  mig.token_address,
  mig.sol_lamports,
  mig.meme_amount,
  lau.block_timestamp AS launch_time,
  lau.tx_id AS launch_tx_id,
  lau.token_creator,
  lau.pumpfun_pool_address
FROM (
  SELECT
    block_timestamp,
    tx_id,
    decoded_instruction:accounts[9]:pubkey::string AS token_address,
    decoded_instruction:args:initCoinAmount::number AS sol_lamports,
    decoded_instruction:args:initPcAmount::number AS meme_amount
  FROM
    solana.core.ez_events_decoded
  WHERE
    signers[0] = '39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg'
    AND program_id = '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8'
    AND event_type = 'initialize2'
    AND succeeded
    AND decoded_instruction:accounts[8]:pubkey::string = 'So11111111111111111111111111111111111111112'
    AND block_timestamp > TIMESTAMP '{timestamp}'
  ORDER BY block_timestamp
  LIMIT {num}
) mig
LEFT JOIN (
  SELECT
    block_timestamp,
    tx_id,
    decoded_instruction:accounts[0]:pubkey::string AS token_address,
    signers[0] AS token_creator,
    decoded_instruction:accounts[2]:pubkey::string AS pumpfun_pool_address
  FROM
    solana.core.ez_events_decoded
  WHERE
    program_id = '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P'
    AND event_type = 'create'
) lau
ON mig.token_address = lau.token_address;
ORDER BY block_timestamp"""


def process_txn(category: Literal["pumpfun", "raydium"]) -> None:
    """Process the transactions data in the pool"""

    for save_path in ["txn", "creation", "transfer"]:
        os.makedirs(
            PROCESSED_DATA_PATH / save_path / category,
            exist_ok=True,
        )

    for pool_info in tqdm(import_pool(category)):
        token_add = pool_info["token_address"]
        block_ts = int(
            datetime.strptime(
                str(pool_info["block_timestamp"]), "%Y-%m-%dT%H:%M:%S.%fZ"
            ).timestamp()
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
                    "created_time": block_ts,
                    "launch_time": launch_ts,
                    "token_creator": pool_info["token_creator"],
                    "pumpfun_pool_address": pool_info["pumpfun_pool_address"],
                    "launch_tx_id": pool_info["launch_tx_id"],
                },
                f,
                indent=4,
            )

        # # processed transfer data
        # with open(
        #     DATA_PATH / "solana" / category / "transfer" / f"{token_add}.jsonl",
        #     "r",
        #     encoding="utf-8",
        # ) as f:
        #     transfer_lst = []
        #     for line in f:
        #         transfer = json.loads(line)
        #         transfer_lst.append(
        #             Transfer(
        #                 date=datetime.strptime(
        #                     transfer["block_timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ"
        #                 ),
        #                 block=transfer["block_id"],
        #                 txn_hash=transfer["tx_id"],
        #                 log_index=transfer["index"],
        #                 from_=transfer["tx_from"],
        #                 to=transfer["tx_to"],
        #                 value=transfer["amount"],
        #             )
        #         )

        # with open(
        #     PROCESSED_DATA_PATH / "transfer" / category / f"{token_add}.pkl",
        #     "wb",
        # ) as f:
        #     pickle.dump(transfer_lst, f)

        # # processed transaction data
        # with open(
        #     DATA_PATH / "solana" / category / "txn" / f"{token_add}.jsonl",
        #     "r",
        #     encoding="utf-8",
        # ) as f:
        #     txn_lst = []
        #     for line in f:
        #         txn = json.loads(line)

        #         # special case for the swap
        #         if txn["swap_from_amount"] * txn["swap_to_amount"] == 0:
        #             continue

        #         if (txn["swap_from_symbol"] != "SOL") & (
        #             txn["swap_to_symbol"] != "SOL"
        #         ):
        #             continue

        #         txn_lst.append(
        #             Txn(
        #                 date=datetime.strptime(
        #                     txn["block_timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ"
        #                 ),
        #                 block=txn["block_id"],
        #                 txn_hash=txn["tx_id"],
        #                 maker=txn["swapper"],
        #                 acts={
        #                     0: Swap(
        #                         block=txn["block_id"],
        #                         txn_hash=txn["tx_id"],
        #                         log_index=0,
        #                         typ=(
        #                             "Buy"
        #                             if txn["swap_from_symbol"] == "SOL"
        #                             else "Sell"
        #                         ),
        #                         usd=(
        #                             txn["swap_from_amount_usd"]
        #                             if txn["swap_from_amount_usd"]
        #                             else txn["swap_to_amount_usd"]
        #                         ),
        #                         base=(
        #                             txn["swap_to_amount"]
        #                             if txn["swap_from_symbol"] == "SOL"
        #                             else txn["swap_from_amount"]
        #                         ),
        #                         quote=(
        #                             txn["swap_from_amount"]
        #                             if txn["swap_from_symbol"] == "SOL"
        #                             else txn["swap_to_amount"]
        #                         ),
        #                         price=(
        #                             txn["swap_from_amount_usd"] / txn["swap_to_amount"]
        #                             if txn["swap_from_symbol"] == "SOL"
        #                             else txn["swap_to_amount_usd"]
        #                             / txn["swap_from_amount"]
        #                         ),
        #                     )
        #                 },
        #             )
        #         )

        # with open(
        #     PROCESSED_DATA_PATH / "txn" / category / f"{token_add}.pkl",
        #     "wb",
        # ) as f:
        #     pickle.dump(txn_lst, f)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Solana Fetcher")
    parser.add_argument(
        "--num",
        type=int,
        default=100,
        help="Number of tokens to fetch",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default="2025-01-17 14:01:48",
        help="Timestamp to start fetching from",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to run the Solana fetcher."""

    args = parse_args()

    # Fetch the pumpfun meme coins
    pumpfun_fetcher = SolanaFetcher(
        category="pumpfun",
        num=args.num,
        timestamp=args.timestamp,
        task_query=LAUNCH_QUERY,
    )
    pumpfun_fetcher.fetch(SWAP_QUERY, DATA_PATH / "solana" / "pumpfun" / "txn")
    pumpfun_fetcher.fetch(TRANSFER_QUERY, DATA_PATH / "solana" / "pumpfun" / "transfer")

    # Fetch the raydium migration meme coins
    raydium_fetcher = SolanaFetcher(
        category="raydium",
        num=args.num,
        timestamp=args.timestamp,
        task_query=MIGRATION_QUERY,
    )
    raydium_fetcher.fetch(SWAP_QUERY, DATA_PATH / "solana" / "raydium" / "txn")
    raydium_fetcher.fetch(TRANSFER_QUERY, DATA_PATH / "solana" / "raydium" / "transfer")

    # Process the transactions data
    for category in ["pumpfun", "raydium"]:
        process_txn(category)


if __name__ == "__main__":
    # solana_fetcher = SolanaFetcher(
    #     category="pumpfun",
    #     num=1000,
    #     timestamp="2025-01-17 14:01:48",
    #     task_query=LAUNCH_QUERY,
    # )
    # # solana_fetcher.fetch_task(
    # #     LAUNCH_QUERY,
    # #     "2025-01-17 14:01:48",
    # #     1000,
    # #     DATA_PATH / "solana" / "raydium.jsonl",
    # # )
    # solana_fetcher.fetch(SWAP_QUERY, DATA_PATH / "solana" / "pumpfun" / "txn")
    # solana_fetcher.fetch(TRANSFER_QUERY, DATA_PATH / "solana" / "pumpfun" / "transfer")

    # solana_fetcher = SolanaFetcher(
    #     category="raydium",
    #     num=1000,
    #     timestamp="2025-01-17 14:01:48",
    #     task_query=MIGRATION_QUERY,
    # )
    # # solana_fetcher.fetch_task(
    # #     MIGRATION_QUERY,
    # #     "2025-01-17 14:01:48",
    # #     1000,
    # #     DATA_PATH / "solana" / "raydium.jsonl",
    # # )
    # # solana_fetcher.fetch(SWAP_QUERY, DATA_PATH / "solana" / "raydium" / "txn")
    # # solana_fetcher.fetch(TRANSFER_QUERY, DATA_PATH / "solana" / "raydium" / "transfer")
    # solana_fetcher.fetch_replies(
    #     save_path=DATA_PATH / "solana" / "raydium" / "reply",
    # )
    for category in ["raydium"]:
        process_txn(category)
