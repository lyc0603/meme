"""
Class to filter the event from Ethereum
"""

import json
import logging
import os
import time
from typing import Any, Dict, Iterable

from dotenv import load_dotenv
from eth_abi.codec import ABICodec
from tenacity import retry, stop_after_attempt, wait_exponential
from web3 import Web3
from web3._utils.events import get_event_data
from web3._utils.filters import construct_event_filter_params
from web3.providers import HTTPProvider
from web3.exceptions import ContractLogicError

from environ.constants import (
    ABI_PATH,
    INFURA_API_BASE_DICT,
    UNISWAP_V3_NATIVE_USDC_500_DICT,
)

load_dotenv()
logger = logging.getLogger(__name__)
default_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)

INFURA_API_KEYS = str(os.getenv("INFURA_API_KEYS")).split(",")

ERC20_DECIMAL = 18
decimal_set = {}


# Functions to fetch single data point


@default_retry
def fetch_current_block(w3: Web3) -> int:
    """Fetch the current block number"""
    time.sleep(0.2)
    return w3.eth.block_number


@default_retry
def get_token_decimals(w3: Web3, token_address: str) -> int:
    """Get the number of decimals for a token"""

    try:
        return call_function(
            w3,
            token_address,
            json.load(open(ABI_PATH / "erc20.json", encoding="utf-8")),
            "decimals",
        )
    except ContractLogicError:
        return ERC20_DECIMAL


def fetch_token_decimal(w3: Web3, token: str) -> int:
    """Fetch the token decimal"""
    if token in decimal_set:
        return decimal_set[token]

    decimal = get_token_decimals(w3, token)
    decimal_set[token] = decimal
    return decimal


@default_retry
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


def fetch_native_price(block: int | str, w3: Web3, chain: str) -> float:
    """Fetch the WETH price from Uniswap V3"""
    return (
        1
        / fetch_price(
            w3,
            UNISWAP_V3_NATIVE_USDC_500_DICT[chain]["pool"],
            block,
            token0_decimal=UNISWAP_V3_NATIVE_USDC_500_DICT[chain]["token0_decimal"],
            token1_decimal=UNISWAP_V3_NATIVE_USDC_500_DICT[chain]["token1_decimal"],
        )
        if UNISWAP_V3_NATIVE_USDC_500_DICT[chain]["token0"] in ["USDC", "USDB"]
        else fetch_price(
            w3,
            UNISWAP_V3_NATIVE_USDC_500_DICT[chain]["pool"],
            block,
            token0_decimal=UNISWAP_V3_NATIVE_USDC_500_DICT[chain]["token0_decimal"],
            token1_decimal=UNISWAP_V3_NATIVE_USDC_500_DICT[chain]["token1_decimal"],
        )
    )


@default_retry
def fetch_block_timestamp(block_num: int, w3: Web3) -> int:
    """
    Given a block number and a Web3 instance, return the block's timestamp (as an int).
    """
    time.sleep(0.2)
    block_info = w3.eth.get_block(block_num)
    return block_info["timestamp"]


def estimate_block_freq(w3: Web3) -> int:
    """
    Estimate the block gap between the current block and the latest block.
    """
    return max(
        fetch_block_timestamp(fetch_current_block(w3), w3)
        - fetch_block_timestamp(fetch_current_block(w3) - 1, w3),
        1,
    )


@default_retry
def fetch_transaction(tx_hash: str, w3: Web3) -> Dict[str, Any]:
    """
    Given a transaction hash and a Web3 instance, return transaction data.
    """
    tx = w3.eth.get_transaction(tx_hash)
    return dict(tx)


@default_retry
def fetch_events_for_all_contracts(
    w3: Web3,
    event: Any,
    argument_filters: Dict[str, Any],
    from_block: int,
    to_block: int,
) -> Iterable:
    """Method to get events

    Args:
        w3 (Web3): The Web3 instance
        event (Any): The event to fetch
        argument_filters (Dict[str, Any]): The filters to apply to the event
        from_block (int): The block number to start fetching events from, inclusive
        to_block (int): The block number to stop fetching events from, inclusive
    """

    time.sleep(0.2)
    if from_block is None:
        raise ValueError("Missing mandatory keyword argument 'from_block'")

    # Construct the event filter parameters
    abi = event._get_event_abi()
    codec: ABICodec = w3.codec
    _, event_filter_params = construct_event_filter_params(
        abi,
        codec,
        address=argument_filters.get("address"),
        argument_filters=argument_filters,
        from_block=from_block,
        to_block=to_block,
    )

    # logging
    logs = w3.eth.get_logs(event_filter_params)

    all_events = []
    for log in logs:
        evt = get_event_data(codec, abi, log)
        all_events.append(evt)

    return all_events


@default_retry
def call_function(
    w3: Web3,
    address: str,
    abi: Dict,
    func_name: str,
    block: int | str = "latest",
    *args,
) -> Any:
    """Method to call a function on a contract"""

    time.sleep(0.2)
    contract = w3.eth.contract(address=Web3.to_checksum_address(address), abi=abi)

    if not hasattr(contract.functions, func_name):
        raise ValueError(f"Function {func_name} not found in contract")

    return getattr(contract.functions, func_name)(*args).call(block_identifier=block)


if __name__ == "__main__":

    import os

    from environ.constants import INFURA_API_BASE_DICT

    CHAIN = "blast"

    INFURA_API_KEY = str(os.getenv("INFURA_API_KEYS")).rsplit(",", maxsplit=1)[-1]
    w3 = Web3(HTTPProvider(f"{INFURA_API_BASE_DICT[CHAIN]}{INFURA_API_KEY}"))

    # _ = estimate_block_freq(w3)
    _ = fetch_native_price(fetch_current_block(w3), w3, CHAIN)
