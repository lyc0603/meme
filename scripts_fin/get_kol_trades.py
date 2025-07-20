"""This script processes KOL (Key Opinion Leader) traders to extract tokens involved in their trades."""

import json
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH

TRADER_PATH = PROCESSED_DATA_PATH / "trader"


def load_kol_traders() -> pd.DataFrame:
    """Load and shuffle KOL and non-KOL traders with balanced sampling."""

    trader_t = pd.read_csv(PROCESSED_DATA_PATH / "trader_t_stats.csv")
    trader_t = trader_t.loc[trader_t["meme_num"] <= 1000].dropna(subset=["t_stat"])

    kol = trader_t.loc[trader_t["t_stat"] > 2.576].sample(50)
    non_kol = trader_t.loc[trader_t["t_stat"] <= 2.576].sample(50)

    combined = pd.concat([kol, non_kol], ignore_index=True)
    return combined.sample(frac=1, random_state=42).reset_index(drop=True)


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
    # trader_t = pd.read_csv(PROCESSED_DATA_PATH / "trader_t_stats.csv")
    kol_meme = main()
    with open(
        PROCESSED_DATA_PATH / "kol_non_kol_traded_tokens.json", "w", encoding="utf-8"
    ) as f:
        json.dump(kol_meme, f, indent=4)
