"""Base Class for Meme Environment."""

import os
import datetime
import json
import pickle
from collections import defaultdict
from datetime import UTC, timezone

from environ.constants import PROCESSED_DATA_PATH
from environ.data_class import NewTokenPool, Swap

MIGATOR = "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg"


class MemeBase:
    """Base class for meme token analysis"""

    def __init__(self, new_token_pool: NewTokenPool):
        self.new_token_pool = new_token_pool
        self.txn = self._load_pickle("txn")
        self.transfer = self._load_pickle("transfer")
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
        self.non_swap_transfers, self.swappers, self.time_traders = (
            self._build_transfer_swap()
        )

    def _load_pickle(self, attr: str):
        """Method to load the pickle file of the meme token"""
        path = (
            f"{PROCESSED_DATA_PATH}/{attr}/"
            f"{self.new_token_pool.chain}/{self.new_token_pool.pool_add}.pkl"
        )
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_jsonl(self, attr: str):
        """Method to load the jsonl file of the meme token"""
        path = (
            f"{PROCESSED_DATA_PATH}/{attr}/"
            f"{self.new_token_pool.chain}/{self.new_token_pool.pool_add}.jsonl"
        )
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                file = f.readlines()
                return [json.loads(line) for line in file]
        else:
            return []

    def _load_creation(
        self,
    ) -> tuple[int, int, datetime.datetime, datetime.datetime, str, str, str]:
        """Method to load the creation information of the meme token"""
        path = (
            f"{PROCESSED_DATA_PATH}/creation/{self.new_token_pool.chain}/"
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

    def _build_transfer_swap(self) -> tuple[list[str], dict[str, list]]:
        """Method to get the unique non-swap transfers of the meme token"""
        non_swap_transfers_hash = set()
        for transfer in self.transfer:
            if self.migrate_time:
                if (transfer.date.replace(tzinfo=timezone.utc) < self.migrate_time) & (
                    MIGATOR not in [transfer.from_, transfer.to]
                ):
                    non_swap_transfers_hash.add(transfer.txn_hash)
            else:
                if MIGATOR not in [transfer.from_, transfer.to]:
                    non_swap_transfers_hash.add(transfer.txn_hash)

        swappers = defaultdict(list)
        time_traders = {}
        unique_traders = set()
        for i, swap in enumerate(self.get_acts(Swap)):
            swappers[swap["maker"]].append(swap)
            non_swap_transfers_hash.discard(swap["txn_hash"])

            delta_int = int(
                (
                    swap["date"].replace(tzinfo=timezone.utc) - self.launch_time
                ).total_seconds()
                / 60
            )
            if i == 0:
                last_delta = delta_int

            if delta_int > last_delta:
                for delta in range(last_delta, delta_int):
                    time_traders[delta] = len(unique_traders)
                last_delta = delta_int

            unique_traders.add(swap["maker"])

        non_swap_transfers = []
        for transfer in self.transfer:
            if transfer.txn_hash in non_swap_transfers_hash:
                non_swap_transfers.append(transfer)

        return non_swap_transfers, swappers, time_traders

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

    def get_time_traders(self, subset: set) -> dict[int, int]:
        """Method to get the time traders of the meme token"""
        return {k: v for k, v in self.time_traders.items() if k in subset}
