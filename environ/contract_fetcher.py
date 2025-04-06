"""
Script to fetch smart contract from the blockexplorer
"""

import pickle

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from environ.constants import (
    BLOCK_EXPLORER_API_DICT,
    BLOCK_EXPLORER_BASE_URL_DICT,
    PROCESSED_DATA_PATH,
)
from environ.data_class import Contract, NewTokenPool

default_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)


class TokenContractFecther:
    """
    Class to fetch the contract from the block explorer
    """

    def __init__(self, new_token_pool: NewTokenPool):
        """Initialize the class

        Args:
            new_token_pool (str): The new token pool to fetch the contract from
        """
        self.new_token_pool = new_token_pool
        self.contract = self.parse_contract(self.get_contract()["result"][0])

    @default_retry
    def get_contract(self) -> dict:
        """
        Method to get the contract from the block explorer
        """
        url = (
            BLOCK_EXPLORER_BASE_URL_DICT[self.new_token_pool.chain]
            + "/api?module=contract&action=getsourcecode&address="
            + self.new_token_pool.base_token
            + "&apikey="
            + BLOCK_EXPLORER_API_DICT[self.new_token_pool.chain]
        )
        results = requests.get(url, timeout=10)
        return results.json()

    def parse_contract(self, contract: dict) -> Contract:
        """
        Method to parse the contract from the block explorer
        """

        return Contract(
            source_code=contract["SourceCode"],
            abi=contract["ABI"],
            contract_name=contract["ContractName"],
            compiler_version=contract["CompilerVersion"],
            optimization_used=contract["OptimizationUsed"],
            runs=contract["Runs"],
            contructor_arguments=contract["ConstructorArguments"],
            evm_version=contract["EVMVersion"],
            library=contract["Library"],
            license_type=contract["LicenseType"],
            swarm_source=contract["SwarmSource"],
            similar_match=contract["SimilarMatch"],
            proxy=contract["Proxy"],
            implementation=contract["Implementation"],
        )

    def save_contract(self) -> None:
        """
        Method to save the contract to the database
        """
        with open(
            f"{PROCESSED_DATA_PATH}/smart_contract/{self.new_token_pool.chain}/{self.new_token_pool.pool_add}.pkl",
            "wb",
        ) as file:
            pickle.dump(self.contract, file)


if __name__ == "__main__":
    # Example usage
    new_token_pool = NewTokenPool(
        token0="0x4200000000000000000000000000000000000006",
        token1="0x9A487b50c0E98BF7c4c63E8E09A5A21A34B1E579",
        fee=10000,
        pool_add="0x40Fae4EE4d8C5A1629571A6aA363C99Ae11D28e5",
        block_number=25168417,
        chain="base",
        base_token="0xA551F0ed440D9A5A0535DeE03EBa7bD6341BaBd2",
        quote_token="0x4200000000000000000000000000000000000006",
        txns={},
    )
    _ = TokenContractFecther(new_token_pool)
