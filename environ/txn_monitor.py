"""Class to monitor the transaction"""

import datetime
import json
import logging
import os
import pickle
from collections import defaultdict
from datetime import UTC, timedelta
from typing import Any, Callable, Iterable

from web3 import Web3
from web3.exceptions import Web3RPCError

from environ.constants import (
    ABI_PATH,
    INFURA_API_BASE_DICT,
    NATIVE_ADDRESS_DICT,
    PROCESSED_DATA_PATH,
)
from environ.data_class import Burn, Collect, Mint, NewTokenPool, Swap, Transfer, Txn
from environ.eth_fetcher import (
    estimate_block_freq,
    fetch_block_timestamp,
    fetch_current_block,
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
        self.transfers = []
        self.new_token_pool = new_token_pool
        self.chain = new_token_pool.chain
        self.duration = duration
        self.w3 = Web3(Web3.HTTPProvider(INFURA_API_BASE_DICT[self.chain] + api))

        # Fetch the corrent block
        self.current_block = fetch_current_block(self.w3)
        self.est_block_duration = (
            self.duration.total_seconds() / estimate_block_freq(self.w3) / AVG_ITER
        )
        self.finished = False

        # Last block of the transaction
        with open(
            f"{PROCESSED_DATA_PATH}/txn/{self.chain}/{self.new_token_pool.pool_add}.pkl",
            "rb",
        ) as f:
            transactions = pickle.load(f)
            if len(transactions) > 0:
                self.last_block = transactions[-1].block
            else:
                self.last_block = self.new_token_pool.block_number

        # Fetch the token decimals
        self.token0_decimal = fetch_token_decimal(self.w3, self.new_token_pool.token0)
        self.token1_decimal = fetch_token_decimal(self.w3, self.new_token_pool.token1)
        self.created_time = fetch_block_timestamp(
            self.new_token_pool.block_number, self.w3
        )
        for path in [
            f"{PROCESSED_DATA_PATH}/txn/{self.new_token_pool.chain}/",
            f"{PROCESSED_DATA_PATH}/transfer/{self.new_token_pool.chain}/",
        ]:
            os.makedirs(path, exist_ok=True)

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

    def fetch_transfer(
        self,
        from_block: int,
        to_block: int,
    ) -> None:
        """Fetch the transfer events

        Args:
            from_block (int): The from block
            to_block (int): The to block
        """

        try:
            fetched_events = fetch_events_for_all_contracts(
                self.w3,
                self.w3.eth.contract(
                    abi=json.load(open(ABI_PATH / "erc20.json", encoding="utf-8")),
                ).events.Transfer,
                {"address": Web3.to_checksum_address(self.new_token_pool.base_token)},
                from_block,
                to_block,
            )
            self.transfers.extend(
                [self.parse_transfer(fetched_event) for fetched_event in fetched_events]
            )

        except Web3RPCError as e:
            try:
                error_msg = json.loads(e.args[0].replace("'", '"'))
                if error_msg["code"] == -32005:
                    mid_block = (from_block + to_block) // 2

                    # Recursive call, pass accumulator
                    self.fetch_transfer(from_block, mid_block)
                    self.fetch_transfer(mid_block + 1, to_block)

            except json.JSONDecodeError:
                print(
                    f"Fetching Swaps: JSON decode error fetching swap events for block range {e}"
                )
            except Exception:
                print(
                    f"Fetching Swaps: Unknown error fetching swap events for block range {e}"
                )
        except Exception as e:
            print(
                f"Fetching Swaps: Unknown error fetching swap events for block range {e}"
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
            price = -args["amount0"] / args["amount1"] if args["amount1"] != 0 else 0
            typ = "Sell" if args["amount1"] > 0 else "Buy"
        else:
            price = -args["amount1"] / args["amount0"] if args["amount0"] != 0 else 0
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

    def parse_transfer(self, transfer: dict) -> Transfer:
        """Parse the transfer events

        Args:
            events (Iterable): The events to parse
        """
        common = self._get_common_fields(transfer)
        args = transfer["args"]
        return Transfer(
            **common,
            from_=args["from"],
            to=args["to"],
            value=(
                args["value"] / 10**self.token0_decimal
                if self.new_token_pool.base_token == self.new_token_pool.token0
                else args["value"] / 10**self.token1_decimal
            ),
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

        # If the txn_block_list is empty, we can check the date of the first txn
        if not txn_block_list:
            date = datetime.datetime.fromtimestamp(
                fetch_block_timestamp(from_block, self.w3), UTC
            )
            if (
                date
                > datetime.datetime.fromtimestamp(self.created_time, UTC)
                + self.duration
            ):
                self.finished = True
                return

        # get date for each txn
        for txn_hash, block in txn_block_list:
            txns = txn_dict[txn_hash]
            date = datetime.datetime.fromtimestamp(
                fetch_block_timestamp(block, self.w3), UTC
            )
            # record the block of last txn
            self.last_block = block
            if (
                date
                > datetime.datetime.fromtimestamp(self.created_time, UTC)
                + self.duration
            ):
                self.finished = True
                break
            maker = fetch_transaction(txn_hash, self.w3)["from"]
            self.txns.append(Txn(date, block, txn_hash, txns, maker))

    def aggregate_transfers(self) -> None:
        """Aggregate the transfers"""
        from_block = self.new_token_pool.block_number
        to_block = self.last_block

        self.fetch_transfer(
            from_block=from_block,
            to_block=to_block,
        )

    def aggregate_txns(self) -> None:
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
            f"{PROCESSED_DATA_PATH}/txn/{self.new_token_pool.chain}/{self.new_token_pool.pool_add}.pkl",
            "wb",
        ) as f:
            pickle.dump(self.txns, f)

    def save_transfers(self) -> None:
        """Save the transfers to a file

        Args:
            path (str): The path to save the txns
        """

        with open(
            f"{PROCESSED_DATA_PATH}/transfer/{self.new_token_pool.chain}/{self.new_token_pool.pool_add}.pkl",
            "wb",
        ) as f:
            pickle.dump(self.transfers, f)


if __name__ == "__main__":

    # Test TxnMonitor
    txn = TxnMonitor(
        str(os.getenv("INFURA_API_KEYS")).rsplit(",", maxsplit=1)[-1],
        NewTokenPool(
            token0="0x4200000000000000000000000000000000000006",
            token1="0x9A487b50c0E98BF7c4c63E8E09A5A21A34B1E579",
            fee=10000,
            pool_add="0x40Fae4EE4d8C5A1629571A6aA363C99Ae11D28e5",
            block_number=25168417,
            chain="base",
            base_token="0x9A487b50c0E98BF7c4c63E8E09A5A21A34B1E579",
            quote_token="0x4200000000000000000000000000000000000006",
            txns={},
        ),
    )

    # txn.aggregate_txns()
    txn.aggregate_transfers()
    # txn.save_transfers()
    # txn.save_txns()
