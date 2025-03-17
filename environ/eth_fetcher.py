"""
Class to filter the event from Ethereum
"""

import json
import logging
import multiprocessing
import os
import time
from functools import partial
from typing import Any, Callable, Dict, Iterable

from dotenv import load_dotenv
from eth_abi.codec import ABICodec
from tqdm import tqdm
from web3 import Web3
from web3._utils.events import get_event_data
from web3._utils.filters import construct_event_filter_params
from web3.providers import HTTPProvider

from environ.constants import ABI_PATH, INFURA_API_BASE_DICT, USDC_WETH_500_POOL

load_dotenv()
logger = logging.getLogger(__name__)

INFURA_API_KEYS = str(os.getenv("INFURA_API_KEYS")).split(",")

decimal_set = {}


# Functions to fetch single data point


def fetch_current_block(w3: Web3) -> int:
    """Fetch the current block number"""

    return w3.eth.block_number


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


def fetch_weth_price(block: int | str, w3: Web3) -> float:
    """Fetch the WETH price from Uniswap V3"""
    return 1 / fetch_price(
        w3, USDC_WETH_500_POOL, block, token0_decimal=6, token1_decimal=18
    )


def fetch_block_timestamp(block_num: int, w3: Web3) -> int:
    """
    Given a block number and a Web3 instance, return the block's timestamp (as an int).
    """
    block_info = w3.eth.get_block(block_num)
    return block_info["timestamp"]


def fetch_transaction(tx_hash: str, w3: Web3) -> Dict[str, Any]:
    """
    Given a transaction hash and a Web3 instance, return transaction data.
    """
    tx = w3.eth.get_transaction(tx_hash)
    return dict(tx)


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


def call_function(
    w3: Web3,
    address: str,
    abi: Dict,
    func_name: str,
    block: int | str = "latest",
    *args,
) -> Any:
    """Method to call a function on a contract"""

    contract = w3.eth.contract(address=Web3.to_checksum_address(address), abi=abi)

    if not hasattr(contract.functions, func_name):
        raise ValueError(f"Function {func_name} not found in contract")

    return getattr(contract.functions, func_name)(*args).call(block_identifier=block)


def get_token_decimals(w3: Web3, token_address: str) -> int:
    """Get the number of decimals for a token"""
    return call_function(
        w3,
        token_address,
        json.load(open(ABI_PATH / "erc20.json", encoding="utf-8")),
        "decimals",
    )


# Functions to fetch multiple data points concurrently


def worker_function(
    item: Any,
    fetch_func: Callable[[Any, Web3], Any],
    http_queue: multiprocessing.Queue,
    res_queue: multiprocessing.Queue,
) -> None:
    """Generic worker function to fetch one item (block, tx, etc.) with concurrency."""
    http = http_queue.get()
    w3 = Web3(HTTPProvider(http))
    result = fetch_func(item, w3)
    http_queue.put(http)
    res_queue.put({item: result})


def run_in_parallel(
    chain: str,
    items: Iterable[Any],
    fetch_func: Callable[[Any, Web3], Any],
    infura_keys: Iterable[str],
    infura_base_dict: Dict[str, str],
    num_processes: int = None,
    sleep_between: float = 0.0,
) -> Dict[Any, Any]:
    """
    Generic concurrency function to fetch data for many 'items'.
    """

    if num_processes is None:
        num_processes = min(len(infura_keys), os.cpu_count())

    with multiprocessing.Manager() as manager:
        http_queue = manager.Queue()
        res_queue = manager.Queue()

        for api_key in infura_keys:
            http_queue.put(infura_base_dict[chain] + api_key)

        partial_worker = partial(
            worker_function,
            fetch_func=fetch_func,
            http_queue=http_queue,
            res_queue=res_queue,
        )

        with multiprocessing.Pool(num_processes) as pool:
            for _ in tqdm(pool.imap_unordered(partial_worker, items), total=len(items)):
                if sleep_between > 0:
                    time.sleep(sleep_between)

        # Collect the results
        results = {}
        while not res_queue.empty():
            results.update(res_queue.get())

    return results


def get_block_timestamp_concurrent(
    chain: str,
    blocks: Iterable[int],
) -> Dict[int, int]:
    """
    Return a dictionary of {block_number: timestamp}.
    """
    return run_in_parallel(
        chain=chain,
        items=blocks,
        fetch_func=fetch_block_timestamp,
        infura_keys=INFURA_API_KEYS,
        infura_base_dict=INFURA_API_BASE_DICT,
    )


def get_transaction_concurrent(
    chain: str,
    txns: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Return a dictionary of {transaction_hash: transaction_data}.
    """
    return run_in_parallel(
        chain=chain,
        items=txns,
        fetch_func=fetch_transaction,
        infura_keys=INFURA_API_KEYS,
        infura_base_dict=INFURA_API_BASE_DICT,
    )


def get_weth_price_concurrent(
    chain: str,
    blocks: Iterable[int],
) -> Dict[str, Dict[str, Any]]:
    """
    Return the WETH price.
    """
    return run_in_parallel(
        chain=chain,
        items=blocks,
        fetch_func=fetch_weth_price,
        infura_keys=INFURA_API_KEYS,
        infura_base_dict=INFURA_API_BASE_DICT,
    )
