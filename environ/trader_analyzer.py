"""This module analyzes trader transactions to calculate t-statistics for meme token profits."""

import json
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
import numpy as np

from scipy import stats
from environ.data_class import Multiswap
from environ.constants import PROCESSED_DATA_PATH


class Account:
    """Base class for accounts, representing a trader's address."""

    def __init__(self, address: str):
        self.address = address


class Trader(Account):
    """Class to store trader's transactions and profits."""

    def __init__(self, address: str, category: str):
        super().__init__(address)
        self.category: str = category
        self.swaps: defaultdict[str, list[Multiswap]] = defaultdict(list)
        self.balances: defaultdict[str, float] = defaultdict(float)
        self.profits: defaultdict[str, float] = defaultdict(float)
        self.costs: defaultdict[str, float] = defaultdict(float)
        self.meme_rets: list[float] = []

    def swap(self, swap: Multiswap) -> None:
        """Process a swap transaction and update balances and profits."""
        if swap.typ == "Buy":
            self.buy(swap)
        elif swap.typ == "Sell":
            self.sell(swap)

    def buy(self, swap) -> None:
        """Process a buy transaction."""
        self.balances[swap.meme] += swap.base if swap.base else 0.0
        self.profits[swap.meme] -= swap.usd if swap.usd else 0.0
        self.costs[swap.meme] += swap.usd if swap.usd else 0.0

    def sell(self, swap) -> None:
        """Process a sell transaction."""
        self.balances[swap.meme] -= swap.base if swap.base else 0.0
        self.profits[swap.meme] += swap.usd if swap.usd else 0.0

    def build_meme_profits(self) -> dict[str, float]:
        """Return profits for meme tokens."""

        for k, v in self.profits.items():
            if k.endswith("pump"):
                cost = self.costs[k]
                if (self.balances[k] >= 0) & (cost > 0):
                    self.meme_rets.append(v / cost)

    def profit_t_stats(self) -> float:
        """Calculate t-statistic for profits."""
        if len(self.meme_rets) < 2:
            return float("nan")

        if np.std(self.meme_rets) <= 1e-8:
            return float("nan")

        return stats.ttest_1samp(self.meme_rets, popmean=0)[0]


def process_trader_task(args):
    """Process a trader's transactions and calculate t-statistic for profits."""
    cate, token_add, trader_add = args
    trader = Trader(trader_add, cate)

    file_path = PROCESSED_DATA_PATH / "trader" / cate / f"{trader_add}.jsonl"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                txn = json.loads(line)
                if txn["swap_from_amount"] is None or txn["swap_to_amount"] is None:
                    continue
                if txn["swap_from_amount"] * txn["swap_to_amount"] == 0:
                    continue
                if (txn["swap_from_symbol"] != "SOL") & (
                    txn["swap_to_symbol"] != "SOL"
                ):
                    continue

                trader.swap(
                    Multiswap(
                        block=txn["block_id"],
                        txn_hash=txn["tx_id"],
                        log_index=0,
                        typ=("Buy" if txn["swap_from_symbol"] == "SOL" else "Sell"),
                        usd=txn["swap_from_amount_usd"] or txn["swap_to_amount_usd"],
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
                            else txn["swap_to_amount_usd"] / txn["swap_from_amount"]
                        ),
                        dex=txn["swap_program"],
                        meme=(
                            txn["swap_to_mint"]
                            if txn["swap_from_symbol"] == "SOL"
                            else txn["swap_from_mint"]
                        ),
                    )
                )
            trader.build_meme_profits()

    except FileNotFoundError:
        return None  # skip missing file

    return {
        "category": cate,
        "trader_address": trader.address,
        "token_address": token_add,
        "meme_num": len(trader.meme_rets),
        "t_stat": trader.profit_t_stats(),
    }


if __name__ == "__main__":
    tasks = []
    for cate in ["pre_trump_raydium", "raydium", "pumpfun", "pre_trump_pumpfun"]:
        with open(
            PROCESSED_DATA_PATH / "trader" / f"{cate}.json", "r", encoding="utf-8"
        ) as f:
            traders_data = json.load(f)
        for token_add, trader_add in traders_data.items():
            tasks.append((cate, token_add, trader_add))

    results = []
    with Pool(cpu_count() - 5) as pool:
        for result in tqdm(
            pool.imap_unordered(process_trader_task, tasks), total=len(tasks)
        ):
            if result is not None:
                results.append(result)

    t_stats_df = pd.DataFrame(results)
    t_stats_df.to_csv(PROCESSED_DATA_PATH / "trader_t_stats.csv", index=False)
