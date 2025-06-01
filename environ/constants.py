"""This file contains the configuration settings for the market environment."""

import os

import dotenv

from environ.settings import PROJECT_ROOT

dotenv.load_dotenv()

DATA_PATH = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = PROJECT_ROOT / "processed_data"
FIGURE_PATH = PROJECT_ROOT / "figures"
TABLE_PATH = PROJECT_ROOT / "tables"
ABI_PATH = DATA_PATH / "abi"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
}

# Naming dictionary for regression variables
NAMING_DICT = {
    "size": {
        "duration": "$\\text{Duration}$",
        "#trader": "$\#\\text{Traders}$",
        "#txn": "$\#\\text{Txns}$",
        "#transfer": "$\#\\text{Transfers}$",
    },
    "bundle_bot": {
        "holding_herf": "$\\text{HHI}_{\\text{Holding}}$",
        "bundle": "$\\text{HHI}_{\\text{Bundle}}$",
        "transfer_amount": "$\\text{Transfer Amount (\%)}$",
    },
    "volume_bot": {
        "max_same_txn": "$\\text{Max Same Txn}$",
        "pos_to_number_of_swaps_ratio": "$\\text{Pos/\#Swaps}$",
    },
    "comment_bot": {
        "unique_replies": "$\#\\text{Replies}$",
        "reply_interval_herf": "$\\text{HHI}_{\\text{Interval}}$",
        "unique_repliers": "$\#\\text{Repliers}$",
        "non_swapper_repliers": "$\#\\text{Non-Trader Repliers}$",
    },
    "dev": {
        "dev_transfer": "$\\text{Dev Transfer}$",
        "dev_buy": "$\\text{Dev Buy}$",
        "dev_sell": "$\\text{Dev Sell}$",
    },
}

# before block for the TRUMP block 2025-01-17 14:01:48
TRUMP_BLOCK = {
    "base": 25166580,
    "ethereum": 21644677,
    "bnb": 45847073,
    "optimism": 130761865,
    "blast": 14156346,
    "polygon": 66811123,
    "arbitrum": 296388510,
    "avalanche": 55939100,
}

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
    "ethereum": "https://api.etherscan.io",
    "bnb": "https://api.bscscan.com",
    "optimism": "https://api-optimistic.etherscan.io",
    "polygon": "https://api.polygonscan.com",
    "arbitrum": "https://api.arbiscan.io",
}

# Block Explorer API Keys
BLOCK_EXPLORER_API_DICT = {
    "base": str(os.getenv("BASESCAN_API")),
    "ethereum": str(os.getenv("ETHERSCAN_API")),
    "bnb": str(os.getenv("BNBSCAN_API")),
    "optimism": str(os.getenv("OPSCAN_API")),
    "polygon": str(os.getenv("POLYSCAN_API")),
    "arbitrum": str(os.getenv("ARBISCAN_API")),
}

# Uniswap V3 Native Token - most liquid USDC Pool used for price calculation
UNISWAP_V3_NATIVE_USDC_500_DICT = {
    # 0.5% fee tier
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
    "optimism": {
        "pool": "0x1fb3cf6e48F1E7B10213E7b6d87D4c073C7Fdb7b",
        "token0": "USDC",
        "token1": "WETH",
        "token0_decimal": 6,
        "token1_decimal": 18,
    },
    "polygon": {
        "pool": "0x45dDa9cb7c25131DF268515131f647d726f50608",
        "token0": "WPOL",
        "token1": "USDC",
        "token0_decimal": 18,
        "token1_decimal": 6,
    },
    "arbitrum": {
        "pool": "0xC6962004f452bE9203591991D15f6b388e09E8D0",
        "token0": "WETH",
        "token1": "USDC",
        "token0_decimal": 18,
        "token1_decimal": 6,
    },
    "avalanche": {
        "pool": "0xfAe3f424a0a47706811521E3ee268f00cFb5c45E",
        "token0": "WAVAX",
        "token1": "USDC",
        "token0_decimal": 18,
        "token1_decimal": 6,
    },
    # 0.01% fee tier
    "bnb": {
        "pool": "0x4141325bAc36aFFe9Db165e854982230a14e6d48",
        "token0": "USDC",
        "token1": "WETH",
        "token0_decimal": 18,
        "token1_decimal": 18,
    },
    # 0.3% fee tier
    "blast": {
        "pool": "0xf52B4b69123CbcF07798AE8265642793b2E8990C",
        "token0": "USDB",
        "token1": "WETH",
        "token0_decimal": 18,
        "token1_decimal": 18,
    },
}

NATIVE_ADDRESS_DICT = {
    "ethereum": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "bnb": "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",
    "base": "0x4200000000000000000000000000000000000006",
    "optimism": "0x4200000000000000000000000000000000000006",
    "polygon": "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",
    "arbitrum": "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
    "avalanche": "0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7",
    "blast": "0x4300000000000000000000000000000000000004",
}

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
    "pumpfun": {
        "color": "green",
        "name": "PumpFun",
    },
    "raydium": {
        "color": "red",
        "name": "Raydium",
    },
}

# Uniswap V3 Fee Tiers
FEE_TIER_LIST = [100, 500, 3000, 10000]
