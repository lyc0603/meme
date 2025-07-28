"""
Class to filter the event from Ethereum
"""

import glob
import json
import logging
from typing import Any, Dict, Iterable

from eth_abi.codec import ABICodec
from hexbytes import HexBytes
from web3 import Web3
from web3._utils.events import get_event_data
from web3._utils.filters import construct_event_filter_params
from web3.datastructures import AttributeDict
from web3.providers import HTTPProvider
from environ.meme_analyzer import MemeAnalyzer

from environ.constants import (
    DATA_PATH,
)

logger = logging.getLogger(__name__)


def asterisk(pval: float) -> str:
    """Return asterisks based on standard significance levels."""
    if pval < 0.01:
        return "***"
    elif pval < 0.05:
        return "**"
    elif pval < 0.10:
        return "*"
    else:
        return ""


def extract_pool_set() -> set:
    """Fetch the set of pools from the file"""

    # get the list of all files in the folder
    glob_path = f"{DATA_PATH}/polygon/pool/*.json"
    files = glob.glob(glob_path)

    pool_set = set()
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                event = json.loads(line)
                pool_set.add(event["args"]["pool"])

    return pool_set


def to_dict(obj):
    """Convert an AttributeDict to a regular dictionary"""
    if isinstance(obj, AttributeDict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_dict(item) for item in obj]
    elif isinstance(obj, HexBytes):
        return "0x" + obj.hex()
    return obj


def fetch_current_block(http: str) -> int:
    """Fetch the current block number"""

    w3 = Web3(HTTPProvider(http))

    return w3.eth.block_number


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


def handle_first_comment_bot(
    meme: MemeAnalyzer, token_add: str, launch_time: Any
) -> list[dict]:
    """Extract only the first comment of the earliest comment bot."""
    bot_comments = [c for c in meme.comment_list if c["bot"]]
    if not bot_comments:
        return []
    earliest = min(bot_comments, key=lambda c: c["time"])
    return [
        {
            "token_address": token_add,
            "bot_address": earliest["replier"],
            "first_comment_time": earliest["time"],
            "launch_time": launch_time,
        }
    ]


def handle_first_wash_bot(
    meme: MemeAnalyzer, token_add: str, launch_time: Any
) -> list[dict]:
    """Extract the first wash-trading bot based on earliest first_trade_time."""
    valid_bots = [
        (addr, bot)
        for addr, bot in meme.bots.items()
        if bot.first_trade_time is not None
    ]
    if not valid_bots:
        return []
    bot_add, bot = min(valid_bots, key=lambda pair: pair[1].first_trade_time)
    return [
        {
            "token_address": token_add,
            "bot_address": bot_add,
            "wash_trading_score": bot.wash_trading_score,
            "first_trade_time": bot.first_trade_time,
            "launch_time": launch_time,
        }
    ]
