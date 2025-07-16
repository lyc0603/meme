"""This script processes KOL (Key Opinion Leader) traders to extract tokens involved in their trades."""

import json
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH

SAMPLE_SIZE = 100
TRADER_PATH = PROCESSED_DATA_PATH / "trader"


def load_kol_traders() -> pd.DataFrame:
    """Load KOL traders with t-statistics greater than 2.576."""
    trader_t = pd.read_csv(PROCESSED_DATA_PATH / "trader_t_stats.csv")
    kol = trader_t[trader_t["t_stat"] > 2.576].sample(SAMPLE_SIZE)
    return kol


def process_trader(row_dict: dict) -> tuple[str, list[str]]:
    """Process a single trader's transactions to extract tokens."""
    cate = row_dict["category"]
    trader_add = row_dict["trader_address"]
    file_path = TRADER_PATH / cate / f"{trader_add}.jsonl"

    token_set = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                txn = json.loads(line)
                token_set.add(txn["swap_from_mint"])
                token_set.add(txn["swap_to_mint"])
    except FileNotFoundError:
        return cate, []

    pump_tokens = [tok for tok in token_set if tok.endswith("pump")]
    return trader_add, pump_tokens


def main():
    """Main function to process KOL traders and extract their tokens."""
    kol = load_kol_traders()

    with Pool(processes=cpu_count() - 5) as pool:
        results = list(
            tqdm(
                pool.imap(process_trader, kol.to_dict("records")),
                total=len(kol),
                desc="Processing KOL trades (parallel)",
            )
        )

    kol_meme = defaultdict(list)
    for trader_address, token_list in results:
        kol_meme[trader_address].extend(token_list)

    # Optional: deduplicate token lists
    kol_meme = {k: list(set(v)) for k, v in kol_meme.items()}

    return kol_meme


if __name__ == "__main__":
    kol_meme = main()
