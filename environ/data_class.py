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


# wallet data class
@dataclass
class Wallet:
    """Class to store the wallet data"""

    address: str
    txn: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class Trader:
    """Class to store the trader data"""

    address: str
    token: str
    txns: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


# smart contract data class
@dataclass
class Contract:
    """Class to store the contract data"""

    source_code: str
    abi: str
    contract_name: str
    compiler_version: str
    optimization_used: str
    runs: str
    contructor_arguments: str
    evm_version: str
    library: str
    license_type: str
    swarm_source: str
    similar_match: str
    proxy: str
    implementation: str

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)
