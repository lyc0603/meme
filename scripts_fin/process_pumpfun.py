"""Process raw Solana data fetched from Snowflake (CLI with safe defaults)."""

import argparse
import os
import json
import pickle
from datetime import datetime
from typing import Literal, Iterable
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH, DATA_PATH
from environ.snowflake_fetcher import import_pool
from environ.data_class import Txn, Swap, Transfer


def process_txn(
    category: Literal["pumpfun", "raydium", "pre_trump_raydium", "pre_trump_pumpfun"],
    num: int = 1000,
) -> None:
    """Process the transactions data in the pool."""

    for save_path in ["txn", "creation", "transfer"]:
        os.makedirs(PROCESSED_DATA_PATH / save_path / category, exist_ok=True)

    for pool_info in tqdm(import_pool(category, num=num), desc=f"Process {category}"):
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

        # creation metadata
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

        # transfers
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

        # swaps/txns
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


# -------------------- CLI wrapper (argparse) --------------------

DEFAULTS = {
    "pumpfun": 3000,
    "pre_trump_pumpfun": 3000,
    "raydium": 1000,
    "pre_trump_raydium": 1000,
}
ALL_CATEGORIES = list(DEFAULTS.keys())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Process raw Solana data fetched from Snowflake into pickled objects."
    )
    p.add_argument(
        "--categories",
        nargs="+",
        choices=ALL_CATEGORIES + ["all"],
        default=["all"],  # no args -> process all with defaults
        help="One or more categories to process. Default: all",
    )
    p.add_argument(
        "--num",
        type=int,
        default=None,
        help="Override sample size for ALL selected categories. "
        "If omitted, per-category defaults are used.",
    )
    return p.parse_args()


def ensure_list(categories: Iterable[str]) -> list[str]:
    cats = list(categories)
    if "all" in cats:
        return ALL_CATEGORIES
    return cats


def run_category(category: str, num_override: int | None) -> None:
    n = num_override if num_override is not None else DEFAULTS[category]
    process_txn(category, n)


def main() -> None:
    args = parse_args()
    categories = ensure_list(args.categories)
    for cat in categories:
        run_category(cat, args.num)


if __name__ == "__main__":
    main()
