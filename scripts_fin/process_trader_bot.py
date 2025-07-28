"""Class for Meme Environment."""

import os
import datetime
import json
import pickle
from collections import defaultdict
from datetime import UTC
from datetime import timezone
from pathlib import Path
from typing import Any, Optional
from multiprocessing import Pool, cpu_count, Manager


import numpy as np
import pandas as pd

from environ.constants import PROCESSED_DATA_PATH
from environ.data_class import NewTokenPool, Swap
from tqdm import tqdm

MIGATOR = "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg"
SOL_TOKEN_ADDRESS = "So11111111111111111111111111111111111111112"


class MemeBase:
    """Base class for meme token analysis"""

    def __init__(self, new_token_pool: NewTokenPool):
        self.new_token_pool = new_token_pool
        self.txn = self._load_pickle("txn")
        self.comment = self._load_jsonl("comment")
        (
            self.migrate_block,
            self.launch_block,
            self.migrate_time,
            self.launch_time,
            self.creator,
            self.pool_add,
            self.launch_txn_id,
        ) = self._load_creation()
        self.swappers = self._build_transfer_swap()

    def _load_creation(
        self,
    ) -> tuple[int, int, datetime.datetime, datetime.datetime, str, str, str]:
        """Method to load the creation information of the meme token"""
        path = (
            f"{PROCESSED_DATA_PATH}/learning/creation/"
            f"{self.new_token_pool.pool_add}.json"
        )
        with open(path, "r", encoding="utf-8") as f:
            file = json.load(f)

        return (
            file["migrate_block"] if file["migrate_block"] else None,
            file["launch_block"],
            (
                datetime.datetime.fromtimestamp(file["migrate_time"], UTC)
                if file["migrate_time"]
                else None
            ),
            (datetime.datetime.fromtimestamp(file["launch_time"], UTC)),
            file["token_creator"],
            file["pumpfun_pool_address"],
            file["launch_tx_id"],
        )

    def _load_pickle(self, attr: str):
        """Method to load the pickle file of the meme token"""
        path = (
            f"{PROCESSED_DATA_PATH}/learning/{attr}/"
            f"{self.new_token_pool.pool_add}.pkl"
        )
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_jsonl(self, attr: str):
        """Method to load the jsonl file of the meme token"""
        path = (
            f"{PROCESSED_DATA_PATH}/{attr}/kol_non_kol/"
            f"{self.new_token_pool.pool_add}.jsonl"
        )
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                file = f.readlines()
                return [json.loads(line) for line in file]
        else:
            return []

    def _build_transfer_swap(self) -> tuple[list[str], dict[str, list]]:
        """Method to get the unique non-swap transfers of the meme token"""

        swappers = defaultdict(list)
        for swap in self.get_acts(Swap):
            swappers[swap["maker"]].append(swap)

        return swappers

    def get_acts(self, act: type) -> list:
        """Method to get the swap transaction of the meme token"""
        acts_list = []
        for txn in self.txn:
            acts_dict = {
                k: v for k, v in sorted(txn.acts.items()) if isinstance(v, act)
            }
            if not acts_dict:
                continue
            acts_list.append(
                {
                    "date": txn.date,
                    "acts": acts_dict,
                    "maker": txn.maker,
                    "block": txn.block,
                    "txn_hash": txn.txn_hash,
                }
            )
        acts_list.sort(key=lambda x: x["date"])
        return acts_list


INITIAL_PRICE = 2.8e-8


class Account:
    """Class to store the account data"""

    def __init__(self, address: tuple[str, ...]):
        self.address = address


class Trader(Account):
    """Class to store the trader data"""

    def __init__(self, address: tuple[str, ...], pool_add: str):
        super().__init__(address)
        self.token = pool_add
        self.creator: bool = False
        self.balance: float = 0.0
        self.profit: float = 0.0
        self.swaps: list[Swap] = []
        self.wash_trading_score: Optional[float] = None

    def swap(self, swap: Swap) -> None:
        """Method to handle swap transactions"""
        if swap.typ == "Buy":
            self.buy(swap)
        elif swap.typ == "Sell":
            self.sell(swap)
        self.swaps.append(swap)

    def buy(self, swap) -> None:
        """Method to handle buy transactions"""
        self.balance += swap.base if swap.base else 0.0
        self.profit -= swap.usd if swap.usd else 0.0

    def sell(self, swap) -> None:
        """Method to handle sell transactions"""
        self.balance -= swap.base if swap.base else 0.0
        self.profit += swap.usd if swap.usd else 0.0

    def wash_trading(self) -> None:
        """Method to check the volume of the trader"""

        txn_amount_list = [swap.base for swap in self.swaps]
        if not txn_amount_list:
            return 0.0

        flip_count = sum(
            1
            for i in range(1, len(self.swaps))
            if (self.swaps[i].typ != self.swaps[i - 1].typ)
            & (self.swaps[i].base == self.swaps[i - 1].base)
        )

        self.wash_trading_score = flip_count / (np.abs(self.balance) + 1)

        return self.wash_trading_score


class MemeAnalyzer(MemeBase):
    """Class to analyze meme token"""

    def __init__(
        self,
        new_token_pool: NewTokenPool,
    ):
        super().__init__(new_token_pool)
        self.bundle = self._build_bundle()

        # analyze traders
        self.traders = self._load_swaps()
        self.volume_bot = False
        self.traders, self.bots = self._wash_trade()

    # Build-in Methods
    def _build_bundle(self) -> dict[str, Any]:
        """Method to build the bundle data"""

        bundle = defaultdict(list)

        for _, acts in enumerate(self.get_acts(Swap)):
            if self.migrate_block:
                if acts["block"] <= self.migrate_block:
                    bundle[acts["block"]].append(acts)
            else:
                bundle[acts["block"]].append(acts)

        return {block: acts for block, acts in bundle.items() if len(acts) > 1}

    # Metrics for Sniper Bot
    def get_sniper_bot(self) -> int:
        """Method to check if the meme token is a sniper bot"""
        return int(
            len(
                [
                    _
                    for _ in self.get_acts(Swap)
                    if (_["block"] - self.launch_block < 5)
                    & (_["block"] != self.launch_block)
                    & (_["acts"][0]["typ"] == "Buy")
                ]
            )
            > 0
        )

    # Metrics for Bundle Bot
    def get_bundle_launch_buy_sell_num(self) -> tuple[int]:
        """Method to get the number of bundle buys"""
        bundle_launch = 0

        for block, bundle_info in self.bundle.items():
            if block == self.launch_block:
                if (
                    len(
                        [
                            row["maker"]
                            for row in bundle_info
                            if row["maker"] != self.creator
                        ]
                    )
                    > 0
                ):
                    bundle_launch += 1

        return bundle_launch

    # Metrics for Comment Bot
    def get_comment_bot(self) -> int:
        """Method to get the number of positive comments"""

        for reply in self.comment:
            if reply["bot"]:
                return 1
        return 0

    # Metrics for Volume Bot
    def get_volume_bot(self) -> bool:
        """Method to check if the meme token is a volume bot"""
        return int(self.volume_bot)

    # trader methods
    def _load_swaps(self):
        """Method to load the profit of the traders"""
        traders = {}
        for swapper, swaps_list in self.swappers.items():
            trader = Trader((swapper,), self.pool_add)
            trader.creator = swapper == self.creator
            sorted_swaps = sorted(swaps_list, key=lambda x: x["block"])
            for swap in sorted_swaps:
                trader.swap(swap["acts"][0])
            traders[(swapper,)] = trader
        return traders

    def _wash_trade(self) -> float:
        """Method to get the wash trading volume of the meme token"""
        traders = {}
        bots = {}
        for trader_add, trader in self.traders.items():
            if trader.wash_trading() > 50:
                self.volume_bot = True
                bots[trader_add] = trader
            else:
                traders[trader_add] = trader
        return traders, bots

    def search_trader(self, address: str) -> tuple[tuple[str, ...], Trader]:
        """Method to search for a trader by address"""
        for k, v in self.traders.items():
            if address in k:
                return k, v

        self.traders[(address,)] = Trader((address,), self.pool_add)
        return (address,), self.traders[(address,)]


def analyze_token(token_add: str) -> Optional[tuple[str, dict]]:
    """Return token_address and its bot classification dictionary."""
    try:
        meme = MemeAnalyzer(
            NewTokenPool(
                token0=SOL_TOKEN_ADDRESS,
                token1=token_add,
                fee=0,
                pool_add=token_add,
                block_number=0,
                chain="trader_bot",
                base_token=token_add,
                quote_token=SOL_TOKEN_ADDRESS,
                txns={},
            )
        )

        bot_dict = {
            "launch_bundle": int(bool(meme.get_bundle_launch_buy_sell_num())),
            "volume_bot": meme.get_volume_bot(),
            "sniper_bot": meme.get_sniper_bot(),
            "comment_bot": meme.get_comment_bot(),
        }

        return token_add, bot_dict

    except Exception as e:
        print(f"[ERROR] Token {token_add} failed: {e}")
        return None


def init_worker(q):
    """Initialize worker with a global task queue."""
    global task_queue
    task_queue = q


def worker_from_queue(_):
    """Worker that fetches a token from the global task_queue."""
    try:
        token = task_queue.get_nowait()
    except Exception:
        return None
    return analyze_token(token)


if __name__ == "__main__":
    # Load token list
    with open(
        PROCESSED_DATA_PATH / "kol_non_kol_traded_tokens.json", "r", encoding="utf-8"
    ) as f:
        token_dict = json.load(f)
    token_list = list({t for v in token_dict.values() for t in v})

    # Setup multiprocessing queue
    manager = Manager()
    q = manager.Queue()
    for token in token_list:
        q.put(token)

    # Pool setup
    num_workers = max(cpu_count() - 5, 1)
    with Pool(processes=num_workers, initializer=init_worker, initargs=(q,)) as pool:
        results = list(
            filter(
                None,
                tqdm(
                    pool.imap_unordered(worker_from_queue, range(len(token_list))),
                    total=len(token_list),
                    desc="Analyzing tokens",
                ),
            )
        )

    # Save results
    token_bot_map = dict(results)
    with open(PROCESSED_DATA_PATH / "bot_token_map.json", "w", encoding="utf-8") as f:
        json.dump(token_bot_map, f, indent=2)

    print(f"Processed {len(token_bot_map)} tokens.")
