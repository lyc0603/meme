"""This file contains the configuration settings for the market environment."""

import os

import dotenv

from environ.settings import PROJECT_ROOT

dotenv.load_dotenv()

DATA_PATH = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = PROJECT_ROOT / "processed_data"
FIGURE_PATH = PROJECT_ROOT / "figures"
ABI_PATH = DATA_PATH / "abi"

# Constants for the TRUMP block
TRUMP_BLOCK = {"base": 25166580}

# Infura API Base URLs
INFURA_API_BASE_DICT = {
    "ethereum": "https://mainnet.infura.io/v3/",
    "arbitrum": "https://arbitrum-mainnet.infura.io/v3/",
    "optimism": "https://optimism-mainnet.infura.io/v3/",
    "polygon": "https://polygon-mainnet.infura.io/v3/",
    "base": "https://base-mainnet.infura.io/v3/",
    "bnb": "https://bsc-mainnet.infura.io/v3/",
    "avalanche": "https://avalanche-mainnet.infura.io/v3/",
    "celo": "https://celo-mainnet.infura.io/v3/",
    "blast": "https://blast-mainnet.infura.io/v3/",
    "zksync": "https://zksync-mainnet.infura.io/v3/",
}

# Block Explorer BASE URLs
BLOCK_EXPLORER_BASE_URL_DICT = {
    "base": "https://api.basescan.org",
}

# Ethereum Mainnet Constants
NULL_ADDRESS = "0x0000000000000000000000000000000000000000"
WETH_ADDRESS = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
USDT_ADDRESS = "0xdAC17F958D2ee523a2206206994597C13D831ec7"

PAIR_LIST = set([WETH_ADDRESS, USDC_ADDRESS, USDT_ADDRESS])

# Uniswap V3 Native Token - USDC 500 Pool used for price calculation
UNISWAP_V3_NATIVE_USDC_500_DICT = {
    "ethereum": {
        "pool": "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
        "token0": "USDC",
        "token1": "WETH",
        "token0_decimal": 6,
        "token1_decimal": 18,
    },
    "base": {
        "pool": "0xd0b53D9277642d899DF5C87A3966A349A798F224",
        "token0": "WETH",
        "token1": "USDC",
        "token0_decimal": 18,
        "token1_decimal": 6,
    },
}

NATIVE_ADDRESS_DICT = {
    "ethereum": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "base": "0x4200000000000000000000000000000000000006",
}
# Block Explorer API Keys
BLOCK_EXPLORER_API_DICT = {"base": str(os.getenv("BASE_SCAN_API"))}

# Uniswap V3 Factory Addresses
UNISWAP_V3_FACTORY_DICT = {
    "ethereum": {
        "address": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        "block": 12369621,
        "timestamp": "2021-05-04",
        "step": 100000,
        "color": "grey",
        "name": "Ethereum",
    },
    "arbitrum": {
        "address": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        "block": 165,
        "timestamp": "2021-06-01",
        "step": 1000000,
        "color": "skyblue",
        "name": "Arbitrum",
    },
    "optimism": {
        "address": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        "block": 0,
        "timestamp": "2021-11-11",
        "step": 1000000,
        "color": "lightcoral",
        "name": "Optimism",
    },
    "base": {
        "address": "0x33128a8fC17869897dcE68Ed026d694621f6FDfD",
        "block": 1371680,
        "timestamp": "2023-07-16",
        "step": 50000,
        "color": "blue",
        "name": "Base",
    },
    "polygon": {
        "address": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        "block": 22757547,
        "timestamp": "2021-12-20",
        "step": 500000,
        "color": "purple",
        "name": "Polygon",
    },
    "bnb": {
        "address": "0xdB1d10011AD0Ff90774D0C6Bb92e5C5c8b4461F7",
        "block": 26324014,
        "timestamp": "2023-03-09",
        "step": 500000,
        "color": "orange",
        "name": "BSC",
    },
    "avalanche": {
        "address": "0x740b1c1de25031C31FF4fC9A62f554A55cdC1baD",
        "block": 26324014,
        "timestamp": "2023-02-16",
        "step": 500000,
        "color": "red",
        "name": "Avalanche",
    },
    # "celo": {
    #     "address": "0xAfE208a311B21f13EF87E33A90049fC17A7acDEc",
    #     "block": 13916355,
    #     "step": 500000,
    # },
    "blast": {
        "address": "0x792edAdE80af5fC680d96a2eD80A44247D2Cf6Fd",
        "block": 400903,
        "timestamp": "2024-03-05",
        "step": 500000,
        "color": "gold",
        "name": "Blast",
    },
    # "zksync": {
    #     "address": "0x8FdA5a7a8dCA67BBcDd10F02Fa0649A937215422",
    #     "block": 12637075,
    #     "timestamp": "2023-08-31",
    #     "step": 500000,
    # },
    "solana": {
        "color": "green",
        "name": "Solana",
    },
}

# Uniswap V3 Fee Tiers
FEE_TIER_LIST = [100, 500, 3000, 10000]
