"""Class to fetch Solana data from the Solana API."""

import pickle
import argparse
import json
import os
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any

import dotenv
from flipside import Flipside
from tqdm import tqdm

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH
from environ.data_class import Txn, Swap, Transfer

dotenv.load_dotenv()
FLIPSIDE_API_KEY = os.getenv("FLIPSIDE_API")
FLIPSIDE_BASE_URL = "https://api-v2.flipsidecrypto.xyz"

os.makedirs(DATA_PATH / "solana", exist_ok=True)


class SolanaFetcher:
    """Class to fetch Solana data from the FLIPSIDE."""

    def __init__(
        self,
        pool_path: Path = DATA_PATH / "solana" / "pumpfun.jsonl",
    ) -> None:

        self.flipside = Flipside(str(FLIPSIDE_API_KEY), FLIPSIDE_BASE_URL)
        self.pool = []
        if os.path.exists(pool_path):
            with open(
                pool_path,
                "r",
                encoding="utf-8",
            ) as f:
                for line in f:
                    self.pool.append(json.loads(line))
        self.cache = None

    def fetch_task(
        self,
        query: str,
        timestamp: str,
        num: int,
        save_path: Path,
    ) -> Any:
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
            for row in res.records:
                f.write(json.dumps(row) + "\n")

    def parse_task(
        self,
        num: int,
        save_path: Path,
    ) -> list[tuple[str, str]]:
        """Task to be run."""

        os.makedirs(save_path, exist_ok=True)
        finished = [
            _.split("/")[-1].split(".")[0] for _ in glob(str(save_path / "*.jsonl"))
        ]
        return [
            (token_add, block_ts)
            for idx, (token_add, block_ts) in enumerate(
                [(_["token_address"], _["block_timestamp"]) for _ in self.pool], 1
            )
            if (token_add not in finished) & (idx <= num)
        ]

    def fetch_data(
        self, query: str, token_address: str, migration_timestamp: str, page_number: int
    ) -> Any:
        """Fetch transactions before 12 hours since the migration."""
        return self.flipside.query(
            query.format(
                token_address=token_address, migration_timestamp=migration_timestamp
            ),
            page_number=page_number,
        )

    def fetch(self, query: str, num: int, save_path: Path) -> Any:
        """Fetch transactions before 12 hours since the migration."""

        for token_address, block_timestamp in tqdm(
            self.parse_task(num, save_path),
        ):

            data_lst = []

            current_page_number = 0
            total_pages = 1

            while current_page_number < total_pages:
                current_page_number += 1
                self.cache = self.fetch_data(
                    query,
                    token_address,
                    datetime.strptime(
                        block_timestamp, "%Y-%m-%dT%H:%M:%S.%fZ"
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    current_page_number,
                )
                data_lst.extend(self.cache.records)
                total_pages = self.cache.page.totalPages

                if self.cache.status != "QUERY_STATE_SUCCESS":
                    raise Exception(f"Query failed with status: {self.cache.status}")

            # save as jsonl
            with open(
                save_path / f"{token_address}.jsonl",
                "w",
                encoding="utf-8",
            ) as f:
                for row in data_lst:
                    f.write(json.dumps(row) + "\n")


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

LAUNCH_QUERY = """select
  block_timestamp,
  tx_id,
  decoded_instruction:accounts[0]:pubkey::string as token_address
from
  solana.core.ez_events_decoded
where
  program_id = '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P'
  and event_type = 'create'
  and block_timestamp > timestamp '{timestamp}'
order by block_timestamp
limit {num};"""

MIGRATION_QUERY = """select
  block_timestamp,
  tx_id,
  decoded_instruction:accounts[9]:pubkey::string as token_address,
  decoded_instruction:args:initCoinAmount::number as sol_lamports,
  decoded_instruction:args:initPcAmount::number as meme_amount
from 
  solana.core.ez_events_decoded
where 
  signers[0] = '39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg'
  and program_id = '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8'
  and event_type = 'initialize2'
  and succeeded
  and decoded_instruction:accounts[8]:pubkey::string = 'So11111111111111111111111111111111111111112'
  and block_timestamp > timestamp '{timestamp}'
order by block_timestamp
limit {num};"""


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Solana Fetcher")
    parser.add_argument(
        "--pool_path",
        type=str,
        default=DATA_PATH / "solana" / "pumpfun.jsonl",
        help="Path to the pool file",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=100,
        help="Number of tokens to fetch",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to run the Solana fetcher."""

    args = parse_args()

    # Fetch the pumpfun meme coins
    pumpfun_fetcher = SolanaFetcher(pool_path=args.pool_path)
    pumpfun_fetcher.fetch_task(
        LAUNCH_QUERY,
        "2025-01-17 14:01:48",
        args.num,
        DATA_PATH / "solana" / "pumpfun.jsonl",
    )
    pumpfun_fetcher.fetch(
        SWAP_QUERY, args.num, DATA_PATH / "solana" / "pumpfun" / "txn"
    )
    pumpfun_fetcher.fetch(
        TRANSFER_QUERY, args.num, DATA_PATH / "solana" / "pumpfun" / "transfer"
    )

    # Fetch the raydium migration meme coins
    raydium_fetcher = SolanaFetcher(pool_path=args.pool_path)
    raydium_fetcher.fetch_task(
        MIGRATION_QUERY,
        "2025-01-17 14:01:48",
        args.num,
        DATA_PATH / "solana" / "raydium.jsonl",
    )
    raydium_fetcher.fetch(
        SWAP_QUERY, args.num, DATA_PATH / "solana" / "raydium" / "txn"
    )
    raydium_fetcher.fetch(
        TRANSFER_QUERY, args.num, DATA_PATH / "solana" / "raydium" / "transfer"
    )


if __name__ == "__main__":
    # solana_fetcher = SolanaFetcher(pool_path=DATA_PATH / "solana" / "raydium.jsonl")
    # solana_fetcher.fetch_task(
    #     MIGRATION_QUERY,
    #     "2025-01-17 14:01:48",
    #     1000,
    #     DATA_PATH / "solana" / "raydium.jsonl",
    # )
    # # solana_fetcher.fetch(SWAP_QUERY, 100, DATA_PATH / "solana" / "pumpfun" / "txn")
    # # solana_fetcher.fetch(
    # #     TRANSFER_QUERY, 100, DATA_PATH / "solana" / "pumpfun" / "transfer"
    # # )

    save_path = PROCESSED_DATA_PATH / "txn" / "solana_pumpfun"
    os.makedirs(save_path, exist_ok=True)
    for txn_path in tqdm(
        glob(str(DATA_PATH / "solana" / "pumpfun" / "txn" / "*.jsonl"))
    ):
        token_add = txn_path.split("/")[-1].split(".")[0]
        with open(txn_path, "r", encoding="utf-8") as f:
            txn_lst = []
            for line in f:
                txn = json.loads(line)

                # special case for the swap
                if txn["swap_from_amount"] * txn["swap_to_amount"] == 0:
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
                            0: [
                                Swap(
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
                                        txn["swap_from_amount_usd"]
                                        / txn["swap_to_amount"]
                                        if txn["swap_from_symbol"] == "SOL"
                                        else txn["swap_to_amount_usd"]
                                        / txn["swap_from_amount"]
                                    ),
                                )
                            ]
                        },
                    )
                )

            with open(
                save_path / f"{token_add}.pkl",
                "wb",
            ) as f:
                pickle.dump(txn_lst, f)
