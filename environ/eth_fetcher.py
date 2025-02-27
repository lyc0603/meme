"""
Class to filter the event from Ethereum
"""

import datetime
import json
import logging
from typing import Any, Dict, Iterable

from eth_abi.codec import ABICodec
from web3 import Web3
from web3._utils.events import get_event_data
from web3._utils.filters import construct_event_filter_params
from web3.exceptions import BlockNotFound

from environ.constants import ABI_PATH

logger = logging.getLogger(__name__)


def _fetch_current_block(w3: Web3) -> int:
    """Fetch the current block number"""

    return w3.eth.block_number


def _get_block_timestamp(w3: Web3, block_num) -> datetime.datetime | None:
    """Get Ethereum block timestamp"""
    try:
        block_info = w3.eth.get_block(block_num)
    except BlockNotFound:
        return
    last_time = block_info["timestamp"]
    return datetime.datetime.utcfromtimestamp(last_time)


def _get_transaction(w3: Web3, tx_hash: str) -> Iterable:
    """Get Ethereum transaction"""
    return w3.eth.get_transaction(tx_hash)


def _get_token_decimals(w3: Web3, token_address: str) -> int:
    """Get the number of decimals for a token"""
    return _call_function(
        w3,
        token_address,
        json.load(open(ABI_PATH / "erc20.json", encoding="utf-8")),
        "decimals",
    )


def _fetch_events_for_all_contracts(
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


def _call_function(
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


if __name__ == "__main__":
    from web3 import Web3
    from web3.providers import HTTPProvider

    w3 = Web3(HTTPProvider(f"https://mainnet.infura.io/v3/{INFURA_API_KEY}"))
    _ = _get_transaction(
        w3, "0x26d448c756af6caa5c5467d096ffc17c18489aa8bb7280f8ce4e9e6b975cf502"
    )
