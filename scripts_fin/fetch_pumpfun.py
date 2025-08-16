"""Fetch Solana Data from Snowflake (Argparse version without skip options)"""

import argparse
from pathlib import Path
from environ.sol_fetcher import SolanaFetcher
from environ.query import (
    SWAPPER_QUERY,
    UNCON_SWAP_QUERY,
    UNCON_TRANSFER_QUERY,
    SWAP_QUERY,
    TRANSFER_QUERY,
    LAUNCH_QUERY,
    MIGRATION_QUERY,
)
from environ.constants import PROCESSED_DATA_PATH, DATA_PATH

QUERY_MAP = {
    "pumpfun": LAUNCH_QUERY,
    "raydium": MIGRATION_QUERY,
    "pre_trump_raydium": MIGRATION_QUERY,
    "pre_trump_pumpfun": LAUNCH_QUERY,
}

FETCH_QUERY_MAP = {
    "pumpfun": (UNCON_SWAP_QUERY, UNCON_TRANSFER_QUERY),
    "pre_trump_pumpfun": (UNCON_SWAP_QUERY, UNCON_TRANSFER_QUERY),
    "raydium": (SWAP_QUERY, TRANSFER_QUERY),
    "pre_trump_raydium": (SWAP_QUERY, TRANSFER_QUERY),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch Solana data from Snowflake")
    parser.add_argument(
        "--category",
        type=str,
        choices=["pumpfun", "raydium", "pre_trump_raydium", "pre_trump_pumpfun"],
        required=True,
        help="Token category to fetch",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=1000,
        help="Number of tokens to fetch",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True,
        help="Timestamp to start fetching from (YYYY-MM-DD HH:MM:SS)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    solana_fetcher = SolanaFetcher(
        category=args.category,
        num=args.num,
        timestamp=args.timestamp,
        task_query=QUERY_MAP[args.category],
    )

    # Fetch token pool
    solana_fetcher.fetch_task(
        QUERY_MAP[args.category],
        args.timestamp,
        args.num,
        DATA_PATH / "solana" / f"{args.category}.jsonl",
    )

    # Fetch swaps and transfers
    swap_q, transfer_q = FETCH_QUERY_MAP[args.category]
    solana_fetcher.fetch(
        swap_q, DATA_PATH / "solana" / args.category / "txn", typ="txn"
    )
    solana_fetcher.fetch(
        transfer_q, DATA_PATH / "solana" / args.category / "transfer", typ="transfer"
    )

    # Always fetch replies
    solana_fetcher.fetch_replies(
        save_path=DATA_PATH / "solana" / args.category / "reply"
    )

    # Always fetch trader data
    solana_fetcher.fetch_trader_trading(
        query=SWAPPER_QUERY,
        save_path=PROCESSED_DATA_PATH / "trader" / args.category,
    )


if __name__ == "__main__":
    main()
