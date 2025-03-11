"""
Class to filter the event from Ethereum
"""

import datetime
import json
import logging
import multiprocessing
import os
import time
from functools import partial
from typing import Any, Dict, Iterable

from dotenv import load_dotenv
from eth_abi.codec import ABICodec
from tqdm import tqdm
from web3 import Web3
from web3._utils.events import get_event_data
from web3._utils.filters import construct_event_filter_params
from web3.providers import HTTPProvider

from environ.constants import ABI_PATH, INFURA_API_BASE_DICT

load_dotenv()
logger = logging.getLogger(__name__)

INFURA_API_KEYS = str(os.getenv("INFURA_API_KEYS")).split(",")


def fetch_current_block(w3: Web3) -> int:
    """Fetch the current block number"""

    return w3.eth.block_number


def get_block_timestamp(
    block_num: int, http_queue: multiprocessing.Queue, res_queue: multiprocessing.Queue
) -> None:
    """Get Ethereum block timestamp for a block number"""
    time.sleep(0.5)
    http = http_queue.get()
    w3 = Web3(HTTPProvider(http))
    block_info = w3.eth.get_block(block_num)
    last_time = block_info["timestamp"]
    res_queue.put({block_num: last_time})
    http_queue.put(http)


def get_block_timestamp_concurrent(
    chain: str,
    blocks: Iterable[int],
) -> Dict[int, datetime.datetime]:
    """Get Ethereum block timestamp in batch using multiprocessing"""

    with multiprocessing.Manager() as manager:
        http_queue = manager.Queue()
        res_queue = manager.Queue()
        for api_key in INFURA_API_KEYS:
            http_queue.put(INFURA_API_BASE_DICT[chain] + api_key)

        num_processes = min(len(INFURA_API_KEYS), os.cpu_count())
        with multiprocessing.Pool(num_processes) as pool:
            partial_process = partial(
                get_block_timestamp, http_queue=http_queue, res_queue=res_queue
            )
            for _ in tqdm(
                pool.imap_unordered(
                    partial_process,
                    blocks,
                ),
            ):
                pass

        block_timestamp = {}

        while not res_queue.empty():
            block_timestamp.update(res_queue.get())

    return block_timestamp


def get_transaction(
    tx_hash: str, http_queue: multiprocessing.Queue, res_queue: multiprocessing.Queue
) -> None:
    """Get Blockchain transaction"""
    http = http_queue.get()
    w3 = Web3(HTTPProvider(http))
    tx = w3.eth.get_transaction(tx_hash)
    res_queue.put(
        {
            tx_hash: tx,
        }
    )
    http_queue.put(http)


def get_transaction_concurrent(
    chain: str,
    txns: Iterable[str],
) -> Dict[int, datetime.datetime]:
    """Get Blockchain transaction in batch using multiprocessing"""

    with multiprocessing.Manager() as manager:
        http_queue = manager.Queue()
        res_queue = manager.Queue()
        for api_key in INFURA_API_KEYS:
            http_queue.put(INFURA_API_BASE_DICT[chain] + api_key)

        num_processes = min(len(INFURA_API_KEYS), os.cpu_count())
        with multiprocessing.Pool(num_processes) as pool:
            partial_process = partial(
                get_transaction, http_queue=http_queue, res_queue=res_queue
            )
            for _ in tqdm(
                pool.imap_unordered(
                    partial_process,
                    txns,
                ),
            ):
                pass

        block_timestamp = {}

        while not res_queue.empty():
            block_timestamp.update(res_queue.get())

    return block_timestamp


def get_token_decimals(w3: Web3, token_address: str) -> int:
    """Get the number of decimals for a token"""
    return call_function(
        w3,
        token_address,
        json.load(open(ABI_PATH / "erc20.json", encoding="utf-8")),
        "decimals",
    )


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
