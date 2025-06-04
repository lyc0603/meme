"""Base Class for Meme Environment."""

import datetime
import json
import pickle
from collections import defaultdict
from datetime import UTC, timezone

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH
from environ.data_class import NewTokenPool, Swap

MIGATOR = "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg"


class MemeBase:
    """Base class for meme token analysis"""

    def __init__(self, new_token_pool: NewTokenPool):
        self.new_token_pool = new_token_pool
        self.txn = self._load_pickle("txn")
        self.transfer = self._load_pickle("transfer")
        self.reply = self._load_reply()
        (
            self.block_created_time,
            self.launch_time,
            self.creator,
            self.pool_add,
            self.launch_txn_id,
        ) = self._load_creation()
        self.dev_buy = 0
        self.dev_sell = 0
        self.dev_transfer = 0
        self.dev_transfer_amount = 0
        self.non_swap_transfers, self.swappers = self._build_transfer_swap()

    def _load_pickle(self, attr: str):
        """Method to load the pickle file of the meme token"""
        path = (
            f"{PROCESSED_DATA_PATH}/{attr}/"
            f"{self.new_token_pool.chain}/{self.new_token_pool.pool_add}.pkl"
        )
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_creation(
        self,
    ) -> tuple[datetime.datetime, datetime.datetime, str, str, str]:
        """Method to load the creation information of the meme token"""
        path = (
            f"{PROCESSED_DATA_PATH}/creation/{self.new_token_pool.chain}/"
            f"{self.new_token_pool.pool_add}.json"
        )
        with open(path, "r", encoding="utf-8") as f:
            file = json.load(f)

        return (
            datetime.datetime.fromtimestamp(file["created_time"], UTC),
            file["launch_time"],
            file["token_creator"],
            file["pumpfun_pool_address"],
            file["launch_tx_id"],
        )

    def _load_reply(self) -> list:
        """Method to load the reply of the meme token"""
        path = (
            f"{DATA_PATH}/solana/{self.new_token_pool.chain}/reply/"
            f"{self.new_token_pool.pool_add}.jsonl"
        )
        with open(path, "r", encoding="utf-8") as f:
            file = f.readlines()
            return [json.loads(line) for line in file]

    def _build_transfer_swap(self) -> tuple[list[str], dict[str, list]]:
        """Method to get the unique non-swap transfers of the meme token"""
        non_swap_transfers_hash = set()
        for transfer in self.transfer:
            if (
                transfer.date.replace(tzinfo=timezone.utc) < self.block_created_time
            ) & (MIGATOR not in [transfer.from_, transfer.to]):
                non_swap_transfers_hash.add(transfer.txn_hash)

        swappers = defaultdict(list)
        for swap in self.get_acts(Swap):
            swappers[swap["maker"]].append(swap)
            non_swap_transfers_hash.discard(swap["txn_hash"])

            if self.creator == swap["maker"]:
                match swap["acts"][0].typ:
                    case "Buy":
                        self.dev_buy = 1
                    case "Sell":
                        self.dev_sell = 1

        non_swap_transfers = []
        for transfer in self.transfer:
            if transfer.txn_hash in non_swap_transfers_hash:
                non_swap_transfers.append(transfer)
                if self.creator in [
                    transfer.from_,
                    transfer.to,
                ]:
                    self.dev_transfer = 1

                    if transfer.to == self.creator:
                        self.dev_transfer_amount += transfer.value

        return non_swap_transfers, swappers

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
