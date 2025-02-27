"""
Class to monitor the new token in Uniswap
"""

import json
import logging

from tqdm import tqdm
from web3 import HTTPProvider, Web3

from environ.constants import (
    ABI_PATH,
    FEE_TIER_LIST,
    NULL_ADDRESS,
    PAIR_LIST,
    UNISWAP_V3_FACTORY_CONTRACT,
)
from environ.eth_fetcher import (
    _call_function,
    _fetch_current_block,
    _fetch_events_for_all_contracts,
    _get_block_timestamp,
)
from environ.data_class import NewPool, NewTokenPool

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _get_pool(w3: Web3, token0: str, token1: str, fee_tier: int) -> str:
    """Check if the fee tier exists"""

    return _call_function(
        w3,
        UNISWAP_V3_FACTORY_CONTRACT,
        json.load(
            open(ABI_PATH / f"{UNISWAP_V3_FACTORY_CONTRACT}.json", encoding="utf-8")
        ),
        "getPool",
        "latest",
        token0,
        token1,
        fee_tier,
    )


def _check_pool(w3: Web3, token0: str, token1: str, fee_tier: int) -> bool:
    """Method to check if the fee tier exists"""

    if _get_pool(w3, token0, token1, fee_tier) == NULL_ADDRESS:
        return False

    return True


def _is_meme_token(token: str) -> bool:
    """Check if the token is a meme token"""
    return token not in PAIR_LIST


class TokenMonitor:
    """Class to monitor the new token in Uniswap"""

    def __init__(self, w3: Web3, from_block: int, to_block: int) -> None:
        self.from_block = from_block
        self.to_block = to_block
        self.from_block_timestamp = _get_block_timestamp(w3, from_block)
        self.to_block_timestamp = _get_block_timestamp(w3, to_block)
        self.w3 = w3
        self.new_pool = []
        self.new_token = []
        logger.info(
            "TokenMonitor initialized: from block %d (%s) to block %d (%s)",
            from_block,
            self.from_block_timestamp,
            to_block,
            self.to_block_timestamp,
        )

    def fetch_pool_created(self) -> None:
        """Fetch the historical issued pool created by the token"""

        new_pool_data = _fetch_events_for_all_contracts(
            self.w3,
            self.w3.eth.contract(
                abi=json.load(
                    open(
                        ABI_PATH / f"{UNISWAP_V3_FACTORY_CONTRACT}.json",
                        encoding="utf-8",
                    )
                ),
            ).events.PoolCreated,
            {"address": Web3.to_checksum_address(UNISWAP_V3_FACTORY_CONTRACT)},
            self.from_block,
            self.to_block,
        )
        self.new_pool.extend(
            [
                NewPool(
                    token0=event["args"]["token0"],
                    token1=event["args"]["token1"],
                    fee=event["args"]["fee"],
                    pool_add=event["args"]["pool"],
                    block_number=event["blockNumber"],
                )
                for event in new_pool_data
            ]
        )

    def parse_pool_created(self) -> None:
        """Parse the historical issued pool created by the token."""

        def _isolate_new_token(
            token0: str,
            token1: str,
            meme_token: str,
            pair_token: str,
            fee_tier: int,
            pool: str,
            block_number: int,
        ) -> None:
            """Method to isolate the new token"""

            for token in PAIR_LIST:
                for fee in FEE_TIER_LIST:
                    if (pair_token == token) & (fee_tier == fee):
                        continue
                    if _check_pool(self.w3, meme_token, token, fee):
                        return

            logger.info(
                "New token found: %s, pair token: %s, fee tier: %d, pool: %s",
                meme_token,
                pair_token,
                fee_tier,
                pool,
            )

            self.new_token.append(
                NewTokenPool(
                    token0=token0,
                    token1=token1,
                    fee=fee_tier,
                    pool_add=pool,
                    block_number=block_number,
                    meme_token=meme_token,
                    pair_token=pair_token,
                    txns={},
                )
            )

        for pool in tqdm(self.new_pool):
            token0, token1 = pool.token0, pool.token1
            fee_tier, pool_add = pool.fee, pool.pool_add
            block_number = pool.block_number

            # check whether at least one token is in the PAIR_LIST
            if (token0 not in PAIR_LIST) & (token1 not in PAIR_LIST):
                continue

            for meme_token, pair_token in [(token0, token1), (token1, token0)]:
                if _is_meme_token(meme_token):
                    _isolate_new_token(
                        token0,
                        token1,
                        meme_token,
                        pair_token,
                        fee_tier,
                        pool_add,
                        block_number,
                    )

    # def parse_swap(
    #     self, swaps: Iterable, token1: str, meme_token: str, pair_token: str
    # ) -> pd.DataFrame:
    #     """Method to parse the swap data"""

    #     panel = {
    #         "DATE": [],
    #         "BLOCK": [],
    #         "TYPE": [],
    #         "USD": [],
    #         "MEME": [],
    #         "WETH": [],
    #         "PRICE": [],
    #         "TXN": [],
    #         "MAKER": [],
    #     }

    #     for swap in tqdm(swaps):

    #         if meme_token == token1:
    #             price = -swap["args"]["amount0"] / swap["args"]["amount1"]
    #             typ = "Sell" if swap["args"]["amount1"] > 0 else "Buy"
    #             meme_amount = abs(swap["args"]["amount1"] / 10**18)
    #             pair_amount = abs(swap["args"]["amount0"] / 10**18)
    #         else:
    #             price = -swap["args"]["amount1"] / swap["args"]["amount0"]
    #             typ = "Sell" if swap["args"]["amount0"] > 0 else "Buy"
    #             meme_amount = abs(swap["args"]["amount0"] / 10**18)
    #             pair_amount = abs(swap["args"]["amount1"] / 10**18)

    #         block = swap["blockNumber"]
    #         txn_hash = "0x" + str(swap["transactionHash"].hex())

    #         panel["DATE"].append(_get_block_timestamp(self.w3, block))
    #         panel["BLOCK"].append(block)
    #         panel["TYPE"].append(typ)
    #         if pair_token == WETH_ADDRESS:
    #             weth_price = _fetch_weth_price(self.w3, block)
    #             panel["USD"].append(pair_amount * weth_price)
    #             panel["PRICE"].append(price * weth_price)
    #         else:
    #             panel["USD"].append(pair_amount)
    #             panel["PRICE"].append(price)
    #         panel["MEME"].append(meme_amount)
    #         panel["WETH"].append(pair_amount)
    #         panel["TXN"].append(txn_hash)
    #         panel["MAKER"].append(_get_transaction(self.w3, txn_hash)["from"])

    #     panel = pd.DataFrame(panel).sort_values("BLOCK", ascending=False)
    #     panel.set_index("DATE")["PRICE"].plot(marker="o")

    #     return panel


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    INFURA_API_KEY = str(os.getenv("INFURA_API_KEY"))

    w3 = Web3(HTTPProvider(f"https://mainnet.infura.io/v3/{INFURA_API_KEY}"))
    monitor = TokenMonitor(w3, 21814314, 21814483)
    # monitor.fetch_pool_created()

    # print(
    #     monitor.get_pool(
    #         "0x2206DeBc266B55A141650BD1E5585c31e0deB14C",
    #         "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    #         3000,
    #     )
    # )
    # w3 = Web3(HTTPProvider(f"https://mainnet.infura.io/v3/{INFURA_API_KEY}"))
    current_block = _fetch_current_block(w3)
    # monitor = TokenMonitor(w3, current_block - 100_000, current_block - 90_000)
    # logger.info("Fetching PoolCreated events...")
    # monitor.fetch_pool_created()
    # logger.info("Isolating new tokens...")
    # monitor.parse_pool_created()

    # logger.info(f"Found %d new tokens", len(monitor.new_token))
    # # Test Fetch Swap
    # w3 = Web3(HTTPProvider(f"https://mainnet.infura.io/v3/{INFURA_API_KEY}"))

    # current_block = _fetch_current_block(w3)
    # monitor = TokenMonitor(w3, current_block - 100_000, current_block - 90_000)

    # # Test WETH Price
    # print(_fetch_weth_price(w3))
