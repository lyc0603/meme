"""
Data Classes
"""

import datetime
from dataclasses import dataclass
from typing import Any, Dict, Optional


# pool data class
@dataclass
class NewPool:
    """Class to store the new pool"""

    token0: str
    token1: str
    fee: int
    pool_add: str
    block_number: int
    chain: str

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class NewTokenPool(NewPool):
    """Class to store the new token pool"""

    base_token: str
    quote_token: str
    txns: Dict[int, str]
    creator: str = ""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


# transaction data class
@dataclass
class Txn:
    """Class to store the transaction"""

    date: datetime.datetime
    block: int
    txn_hash: str
    acts: Dict[str, Any]
    maker: str

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class Action:
    """Class to store the transaction"""

    block: int
    txn_hash: str
    log_index: int

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class Transfer(Action):
    """Class to store the transfer transaction"""

    date: datetime.datetime
    from_: str
    to: str
    value: float

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class Swap(Action):
    """Class to store the swap transaction"""

    typ: str
    usd: float
    price: float
    base: float
    quote: float
    dex: Optional[str] = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class Multiswap(Swap):
    """Class to store the multiswap transaction"""

    meme: Optional[str] = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class Mint(Action):
    """Class to store the mint transaction"""

    base: float
    quote: float

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class Burn(Action):
    """Class to store the burn transaction"""

    base: float
    quote: float

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class Collect(Action):
    """Class to store the collect transaction"""

    base: float
    quote: float

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)
