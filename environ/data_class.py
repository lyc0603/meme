"""
Data Classes
"""

import datetime
from dataclasses import dataclass
from typing import Any, Dict


# pool data class
@dataclass
class NewPool:
    """Class to store the new pool"""

    token0: str
    token1: str
    fee: int
    pool_add: str
    block_number: int


@dataclass
class NewTokenPool(NewPool):
    """Class to store the new token pool"""

    meme_token: str
    pair_token: str
    txns: Dict[int, str]


# transaction data class
@dataclass
class Txn:
    """Class to store the transaction"""

    date: datetime.datetime
    block: int
    txn_hash: str
    acts: Dict[str, Any]
    maker: str


@dataclass
class Action:
    """Class to store the transaction"""

    block: int
    txn_hash: str
    log_index: int


@dataclass
class Swap(Action):
    """Class to store the swap transaction"""

    typ: str
    usd: float
    price: float
    meme: float
    pair: float


@dataclass
class Mint(Action):
    """Class to store the mint transaction"""

    meme: float
    pair: float


@dataclass
class Burn(Action):
    """Class to store the burn transaction"""

    meme: float
    pair: float


@dataclass
class Collect(Action):
    """Class to store the collect transaction"""

    meme: float
    pair: float


# wallet data class
@dataclass
class Wallet:
    """Class to store the wallet data"""

    address: str
    txn: Dict[str, Any]


@dataclass
class Trader:
    """Class to store the trader data"""

    address: str
    token: str
    txns: Dict[str, Any]
