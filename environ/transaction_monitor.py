"""
Class to monitor the transaction
"""

import datetime
import json
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List

from tqdm import tqdm
from web3 import HTTPProvider, Web3

from environ.eth_fetcher import (
    fetch_current_block,
    fetch_events_for_all_contracts,
    fetch_token_decimal,
    get_block_timestamp_concurrent,
    get_transaction_concurrent,
    get_weth_price_concurrent,
)
from environ.constants import ABI_PATH, WETH_ADDRESS
from environ.data_class import Burn, Collect, Mint, NewTokenPool, Swap, Trader, Txn


class WalletMonitor:
    """Class to monitor the wallet"""

    def __init__(self, txns: Iterable, token: str) -> None:
        self.txns = txns
        self.token = token
        self.traders = {}

        # Get the traders
        self.get_traders()

    def get_traders(self) -> None:
        """Get the traders"""

        for txn in tqdm(self.txns, desc="Txns"):
            if txn.maker not in self.traders:
                self.traders[txn.maker] = Trader(
                    address=txn.maker,
                    token=self.token,
                    txns={txn.txn_hash: txn},
                )
            else:
                self.traders[txn.maker].txns[txn.txn_hash] = txn


# Token Monitor Class
class TxnMonitor:
    """Class to monitor the transaction of a token"""

    def __init__(
        self,
        w3: Web3,
        new_token_pool: NewTokenPool,
        from_block: int,
        to_block: int,
        chain: str,
    ) -> None:
        self.w3 = w3
        self.new_token_pool = new_token_pool
        self.from_block = from_block
        self.to_block = to_block
        self.chain = chain

        # Fetch the token decimals
        self.token0_decimal = fetch_token_decimal(w3, new_token_pool.token0)
        self.token1_decimal = fetch_token_decimal(w3, new_token_pool.token1)

        # Initialize transaction data
        self.acts = {
            "swap": self._fetch_and_parse_events("Swap", self.parse_swap),
            "mint": self._fetch_and_parse_events("Mint", self.parse_mint),
            "burn": self._fetch_and_parse_events("Burn", self.parse_burn),
            "collect": self._fetch_and_parse_events("Collect", self.parse_collect),
        }
        self.txns = []

    def _fetch_and_parse_events(self, action: str, parser: Callable) -> List:
        """Fetch and parse the events"""
        events = self.fetch_act(action)
        return parser(events)

    def fetch_act(self, act: str = "Swap") -> Iterable:
        """Fetch the swap/mint/burn events"""
        return fetch_events_for_all_contracts(
            self.w3,
            getattr(
                self.w3.eth.contract(
                    abi=json.load(open(ABI_PATH / "v3pool.json", encoding="utf-8")),
                ).events,
                act,
            ),
            {"address": Web3.to_checksum_address(self.new_token_pool.pool_add)},
            self.from_block,
            self.to_block,
        )

    def _get_common_fields(self, event: dict) -> Dict:
        """Extract common fields from the event"""
        return {
            "block": event["blockNumber"],
            "txn_hash": Web3.to_hex(event["transactionHash"]),
            "log_index": event["logIndex"],
        }

    def _get_meme_pair_amounts(self, amount0: int, amount1: int) -> tuple[float, float]:
        """Determine meme and pair amounts based on token positions."""
        if self.new_token_pool.meme_token == self.new_token_pool.token1:
            return (
                abs(amount1) / 10**self.token1_decimal,
                abs(amount0) / 10**self.token0_decimal,
            )
        return (
            abs(amount0) / 10**self.token0_decimal,
            abs(amount1) / 10**self.token1_decimal,
        )

    def parse_swap(self, swaps: list[dict]) -> list[Swap]:
        """Parse a swap event into a structured format."""

        res_list = []

        blocks = [swap["blockNumber"] for swap in swaps]
        weth_prices_dict = get_weth_price_concurrent(self.chain, blocks)

        for swap in swaps:
            common = self._get_common_fields(swap)
            args = swap["args"]
            meme_amount, pair_amount = self._get_meme_pair_amounts(
                args["amount0"], args["amount1"]
            )

            # Determine price and transaction type
            if self.new_token_pool.meme_token == self.new_token_pool.token1:
                price = -args["amount0"] / args["amount1"]
                typ = "Sell" if args["amount1"] > 0 else "Buy"
            else:
                price = -args["amount1"] / args["amount0"]
                typ = "Sell" if args["amount0"] > 0 else "Buy"

            # Calculate USD value
            if self.new_token_pool.pair_token == WETH_ADDRESS:
                weth_price = weth_prices_dict[swap["blockNumber"]]
                usd_value = pair_amount * weth_price
                price *= weth_price
            else:
                usd_value = pair_amount
                price = price

            res_list.append(
                Swap(
                    **common,
                    typ=typ,
                    usd=usd_value,
                    price=price,
                    meme=meme_amount,
                    pair=pair_amount,
                )
            )
        return res_list

    def _parse_generic_event(self, events: list[dict], event_class: type) -> Any:
        """Generic parser for mint/burn/collect events."""

        res_list = []

        for event in events:
            common = self._get_common_fields(event)
            args = event["args"]
            meme_amount, pair_amount = self._get_meme_pair_amounts(
                args["amount0"], args["amount1"]
            )
            res_list.append(event_class(**common, meme=meme_amount, pair=pair_amount))
        return res_list

    def parse_mint(self, mints: list[dict]) -> Mint:
        """Parse a mint event into a structured format."""
        return self._parse_generic_event(mints, Mint)

    def parse_burn(self, burns: list[dict]) -> Burn:
        """Parse a burn event into a structured format."""
        return self._parse_generic_event(burns, Burn)

    def parse_collect(self, collects: list[dict]) -> Collect:
        """Parse a collect event into a structured format."""
        return self._parse_generic_event(collects, Collect)

    def aggregate_act(self, act: str = "Swap") -> None:
        """Aggregate the swap/mint/burn events"""

        txn_dict = defaultdict(dict)

        for _, acts in self.acts.items():
            for act in acts:
                txn_dict[act.txn_hash][act.log_index] = act

        dates_dict = get_block_timestamp_concurrent(
            self.chain,
            [act.block for acts in txn_dict.values() for act in acts.values()],
        )
        txns_dict = get_transaction_concurrent(self.chain, txn_dict.keys())

        # get date for each txn
        for txn_hash, txns in tqdm(txn_dict.items(), desc="Txns Date"):
            block = txns[list(txns.keys())[0]].block
            date = datetime.datetime.utcfromtimestamp(dates_dict[block])
            maker = txns_dict[txn_hash]["from"]
            self.txns.append(Txn(date, block, txn_hash, txns, maker))


if __name__ == "__main__":

    import os

    # Initialize Web3
    INFURA_API_KEY = str(os.getenv("INFURA_API_KEYS")).split(",")[-1]
    w3 = Web3(HTTPProvider(f"https://mainnet.infura.io/v3/{INFURA_API_KEY}"))
    current_block = fetch_current_block(w3)

    # Test TxnMonitor
    txn = TxnMonitor(
        w3,
        NewTokenPool(
            token0="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            token1="0xD1De603884e6424241cAf53EfA846e7C6163755c",
            fee=3000,
            pool_add="0x7315cd518E2eD43b6025c02fA178488F03D25Cdc",
            block_number=21723886,
            meme_token="0xD1De603884e6424241cAf53EfA846e7C6163755c",
            pair_token="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            txns={},
        ),
        21723886,
        current_block,
        "ethereum",
    )

    txn.aggregate_act()
