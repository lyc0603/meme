"""Class to monitor the transaction"""

import datetime
import json
import logging
import os
import pickle
import time
from collections import defaultdict
from datetime import UTC, timedelta
from typing import Any, Callable, Iterable

from web3 import Web3

from environ.constants import (
    ABI_PATH,
    INFURA_API_BASE_DICT,
    NATIVE_ADDRESS_DICT,
    PROCESSED_DATA_PATH,
)
from environ.data_class import Burn, Collect, Mint, NewTokenPool, Swap, Txn
from environ.eth_fetcher import (
    estimate_block_freq,
    fetch_block_timestamp,
    fetch_events_for_all_contracts,
    fetch_token_decimal,
    fetch_transaction,
    fetch_weth_price,
)

AVG_ITER = 3

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="logs/txn_monitor.log",
    filemode="a",
)


class TxnMonitor:
    """Class to monitor the transaction of a token"""

    def __init__(
        self,
        api: str,
        new_token_pool: NewTokenPool,
        duration: timedelta = timedelta(hours=12),
    ) -> None:
        """Initialize the class

        Args:
            api (str): The Infura API key
            new_token_pool (NewTokenPool): The new token pool instance
            duration (timedelta): The time duration to monitor
        """

        self.acts = {}
        self.txns = []
        self.new_token_pool = new_token_pool
        self.chain = new_token_pool.chain
        self.duration = duration
        self.w3 = Web3(Web3.HTTPProvider(INFURA_API_BASE_DICT[self.chain] + api))

        # Fetch the corrent block
        self.current_block = self.w3.eth.block_number
        self.last_block = self.current_block - 1
        self.est_block_duration = (
            self.duration.total_seconds() / estimate_block_freq(self.w3) / AVG_ITER
        )

        self.finished = False

        # Fetch the token decimals
        self.token0_decimal = fetch_token_decimal(self.w3, new_token_pool.token0)
        self.token1_decimal = fetch_token_decimal(self.w3, new_token_pool.token1)
        self.created_time = fetch_block_timestamp(
            self.new_token_pool.block_number, self.w3
        )
        os.makedirs(
            f"{PROCESSED_DATA_PATH}/pool/{self.new_token_pool.chain}/", exist_ok=True
        )

    def fetch_act(
        self,
        act: str,
        from_block: int,
        to_block: int,
    ) -> Iterable:
        """Fetch the act

        Args:
            act (str): The action to fetch
            from_block (int): The from block
            to_block (int): The to block
        """
        time.sleep(0.2)
        return fetch_events_for_all_contracts(
            self.w3,
            getattr(
                self.w3.eth.contract(
                    abi=json.load(open(ABI_PATH / "v3pool.json", encoding="utf-8")),
                ).events,
                act,
            ),
            {"address": Web3.to_checksum_address(self.new_token_pool.pool_add)},
            from_block,
            to_block,
        )

    @staticmethod
    def _get_common_fields(event: dict) -> dict:
        """Extract common fields from the event

        Args:
            event (dict): The event to extract

        Returns:
            Dict: The common fields: block, txn_hash, log_index
        """
        return {
            "block": event["blockNumber"],
            "txn_hash": Web3.to_hex(event["transactionHash"]),
            "log_index": event["logIndex"],
        }

    def _get_base_quote_amounts(
        self, amount0: int, amount1: int
    ) -> tuple[float, float]:
        """Determine base and quote amounts based on token positions."""
        if self.new_token_pool.base_token == self.new_token_pool.token1:
            return (
                abs(amount1) / 10**self.token1_decimal,
                abs(amount0) / 10**self.token0_decimal,
            )
        return (
            abs(amount0) / 10**self.token0_decimal,
            abs(amount1) / 10**self.token1_decimal,
        )

    def parse_swap(self, swap: dict) -> Swap:
        """Parse the swap events

        Args:
            events (Iterable): The events to parse
        """

        common = self._get_common_fields(swap)
        args = swap["args"]
        base_amount, quote_amount = self._get_base_quote_amounts(
            args["amount0"], args["amount1"]
        )

        # Determine price and transaction type
        if self.new_token_pool.base_token == self.new_token_pool.token1:
            price = -args["amount0"] / args["amount1"]
            typ = "Sell" if args["amount1"] > 0 else "Buy"
        else:
            price = -args["amount1"] / args["amount0"]
            typ = "Sell" if args["amount0"] > 0 else "Buy"

        # Calculate USD value
        if self.new_token_pool.quote_token == NATIVE_ADDRESS_DICT[self.chain]:
            weth_price = fetch_weth_price(
                w3=self.w3, block=swap["blockNumber"], chain=self.chain
            )
            usd_value = quote_amount * weth_price
            price *= weth_price
        else:
            usd_value = quote_amount
            price = price

        return Swap(
            **common,
            typ=typ,
            usd=usd_value,
            price=price,
            base=base_amount,
            quote=quote_amount,
        )

    def _parse_generic_event(self, event: dict, event_class: type) -> Any:
        """Generic parser for mint/burn/collect events."""

        common = self._get_common_fields(event)
        args = event["args"]
        base_amount, quote_amount = self._get_base_quote_amounts(
            args["amount0"], args["amount1"]
        )
        return event_class(**common, base=base_amount, quote=quote_amount)

    def parse_mint(self, mint: dict) -> Mint:
        """Parse a mint event into a structured format.
        Args:
            mint (dict): The Mint instance
        """
        return self._parse_generic_event(mint, Mint)

    def parse_burn(self, burn: dict) -> Burn:
        """Parse a burn event into a structured format.
        Args:
            burn (dict): The Burn instance
        """
        return self._parse_generic_event(burn, Burn)

    def parse_collect(self, collect: dict) -> Collect:
        """Parse a collect event into a structured format.
        Args:
            collect (dict): The Collect instance
        """
        return self._parse_generic_event(collect, Collect)

    def _fetch_and_parse_events(
        self, action: str, parser: Callable, from_block: int, to_block: int
    ) -> list:
        """Fetch and parse the events

        Args:
            action (str): The action to fetch
            parser (Callable): The parser function"""
        events = self.fetch_act(act=action, from_block=from_block, to_block=to_block)
        return [parser(event) for event in events]

    def aggregate_act(self, from_block: int, to_block: int) -> None:
        """Aggregate the actions"""

        if self.finished:
            return

        self.acts = {
            "Swap": [],
            "Mint": [],
            "Burn": [],
            "Collect": [],
        }

        for act, _ in self.acts.items():
            parser = getattr(self, f"parse_{act.lower()}")
            self.acts[act].extend(
                self._fetch_and_parse_events(act, parser, from_block, to_block)
            )

        txn_dict = defaultdict(dict)

        for _, acts in self.acts.items():
            for act in acts:
                txn_dict[act.txn_hash][act.log_index] = act

        txn_block_list = [(k, list(v.values())[0].block) for k, v in txn_dict.items()]
        txn_block_list.sort(key=lambda x: x[1])

        # get date for each txn
        for txn_hash, block in txn_block_list:
            txns = txn_dict[txn_hash]
            date = datetime.datetime.fromtimestamp(
                fetch_block_timestamp(block, self.w3), UTC
            )
            if (
                date
                > datetime.datetime.fromtimestamp(self.created_time, UTC)
                + self.duration
            ):
                self.finished = True
                break
            maker = fetch_transaction(txn_hash, self.w3)["from"]
            self.txns.append(Txn(date, block, txn_hash, txns, maker))

    def aggregate(self) -> None:
        """Aggregate the actions"""
        starting_block = self.new_token_pool.block_number
        while not self.finished:
            self.aggregate_act(
                int(starting_block), int(starting_block + self.est_block_duration - 1)
            )
            starting_block += self.est_block_duration

    def save_txns(self) -> None:
        """Save the txns to a file

        Args:
            path (str): The path to save the txns
        """

        with open(
            f"{PROCESSED_DATA_PATH}/pool/{self.new_token_pool.chain}/{self.new_token_pool.pool_add}.pkl",
            "wb",
        ) as f:
            pickle.dump(self.txns, f)


if __name__ == "__main__":

    # Test TxnMonitor
    txn = TxnMonitor(
        str(os.getenv("INFURA_API_KEYS")).rsplit(",", maxsplit=1)[-1],
        NewTokenPool(
            token0="0x4200000000000000000000000000000000000006",
            token1="0xBe35071605277d8Be5a52c84A66AB1bc855A758D",
            fee=10000,
            pool_add="0x946066d23919982116C2Ce07A5bc841c65A60b63",
            block_number=24385454,
            chain="base",
            base_token="0xBe35071605277d8Be5a52c84A66AB1bc855A758D",
            quote_token="0x4200000000000000000000000000000000000006",
            txns={},
        ),
    )

    txn.aggregate()
    txn.save_txns()
