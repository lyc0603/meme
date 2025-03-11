"""
Class to monitor the transaction
"""

import json
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List

from tqdm import tqdm
from web3 import HTTPProvider, Web3

from environ.block_fetcher import (
    call_function,
    fetch_current_block,
    fetch_events_for_all_contracts,
    get_block_timestamp,
    get_block_timestamp_concurrent,
    get_token_decimals,
    get_transaction,
    get_transaction_concurrent,
)
from environ.constants import ABI_PATH, USDC_WETH_500_POOL, WETH_ADDRESS
from environ.data_class import Burn, Collect, Mint, NewTokenPool, Swap, Trader, Txn
import multiprocessing

decimal_set = {}


def fetch_token_decimal(w3: Web3, token: str) -> int:
    """Fetch the token decimal"""
    if token in decimal_set:
        return decimal_set[token]

    decimal = get_token_decimals(w3, token)
    decimal_set[token] = decimal
    return decimal


def fetch_slot(w3: Web3, pool: str, block: int | str) -> Any:
    """Fetch the slot from the pool"""
    return call_function(
        w3,
        pool,
        json.load(open(ABI_PATH / "v3pool.json", encoding="utf-8")),
        "slot0",
        block,
    )


def fetch_sqrtpricex96(w3: Web3, pool: str, block: int | str) -> float:
    """Fetch the sqrtPriceX96 from the pool"""
    return fetch_slot(w3, pool, block)[0]


def fetch_price(
    w3: Web3, pool: str, block: int | str, token0_decimal: int, token1_decimal: int
) -> float:
    """Fetch the y / x price from the pool"""
    return ((fetch_sqrtpricex96(w3, pool, block) / 2**96) ** 2) * (
        10 ** (token0_decimal - token1_decimal)
    )


def fetch_weth_price(w3: Web3, block: int | str = "latest") -> float:
    """Fetch the WETH price from Uniswap V3"""
    return 1 / fetch_price(
        w3, USDC_WETH_500_POOL, block, token0_decimal=6, token1_decimal=18
    )


# def fetch_weth_price_concurrent(
#     chain: str,
#     blocks: Iterable[int],
# ) -> Dict[int, float]:
#     """Fetch the WETH price from Uniswap V3 in batch using multiprocessing"""

#     with multiprocessing.Manager() as manager:
#         w3_queue = manager.Queue()
#         res_queue = manager.Queue()
#         for api_key in INFURA_API_KEYS:
#             w3_queue.put(HTTPProvider(INFURA_API_BASE_DICT[chain] + api_key))

#         num_processes = min(len(INFURA_API_KEYS), os.cpu_count())
#         with multiprocessing.Pool(num_processes) as pool:
#             partial_process = partial(
#                 fetch_weth_price, w3_queue=w3_queue, res_queue=res_queue
#             )
#             for _ in tqdm(
#                 pool.imap_unordered(
#                     partial_process,
#                     blocks,
#                 ),
#             ):
#                 pass

#         block_timestamp = {}

#         while not res_queue.empty():
#             block_timestamp.update(res_queue.get())

#     return block_timestamp


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
        self, w3: Web3, new_token_pool: NewTokenPool, from_block: int, to_block: int
    ) -> None:
        self.w3 = w3
        self.new_token_pool = new_token_pool
        self.from_block = from_block
        self.to_block = to_block

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
        return [parser(event) for event in tqdm(events, desc=f"{action}s")]

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

    def parse_swap(self, swap: dict) -> Swap:
        """Parse a swap event into a structured format."""
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
            weth_price = fetch_weth_price(self.w3, common["block"])
            usd_value = pair_amount * weth_price
            price *= weth_price
        else:
            usd_value = pair_amount
            price = price

        return Swap(
            **common,
            typ=typ,
            usd=usd_value,
            price=price,
            meme=meme_amount,
            pair=pair_amount,
        )

    def _parse_generic_event(self, event: dict, event_class: type) -> Any:
        """Generic parser for mint/burn/collect events."""
        common = self._get_common_fields(event)
        args = event["args"]
        meme_amount, pair_amount = self._get_meme_pair_amounts(
            args["amount0"], args["amount1"]
        )
        return event_class(**common, meme=meme_amount, pair=pair_amount)

    def parse_mint(self, mint: dict) -> Mint:
        """Parse a mint event into a structured format."""
        return self._parse_generic_event(mint, Mint)

    def parse_burn(self, burn: dict) -> Burn:
        """Parse a burn event into a structured format."""
        return self._parse_generic_event(burn, Burn)

    def parse_collect(self, collect: dict) -> Collect:
        """Parse a collect event into a structured format."""
        return self._parse_generic_event(collect, Collect)

    def aggregate_act(self, act: str = "Swap") -> None:
        """Aggregate the swap/mint/burn events"""

        txn_dict = defaultdict(dict)

        for _, acts in self.acts.items():
            for act in acts:
                txn_dict[act.txn_hash][act.log_index] = act

        txns = get_transaction_concurrent("ethereum", txn_dict.keys())
        print(txns)

        # get date for each txn
        for txn_hash, txns in tqdm(txn_dict.items(), desc="Txns Date"):
            block = txns[list(txns.keys())[0]].block
            date = get_block_timestamp_concurrent("ethereum", block)
            maker = get_transaction(self.w3, txn_hash)["from"]
            # self.txns.append(Txn(date, block, txn_hash, txns, maker))


if __name__ == "__main__":

    import os

    # Initialize Web3
    INFURA_API_KEY = str(os.getenv("INFURA_API_KEYS")).split(",")[0]
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
    )

    txn.aggregate_act()
