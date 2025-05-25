"""Maxium Drawdown Detector (MDD) for Meme"""

from environ.data_class import NewTokenPool
from environ.meme_analyzer import MemeAnalyzer
from environ.sol_fetcher import import_pool
from tqdm import tqdm
import pandas as pd

CHAIN = "raydium"
NUM_OF_OBSERVATIONS = 100
SOL_TOKEN_ADDRESS = "So11111111111111111111111111111111111111112"

mdd_list = []

for pool in tqdm(
    import_pool(
        CHAIN,
        NUM_OF_OBSERVATIONS,
    )
):

    meme = MemeAnalyzer(
        NewTokenPool(
            token0=SOL_TOKEN_ADDRESS,
            token1=pool["token_address"],
            fee=0,
            pool_add=pool["token_address"],
            block_number=0,
            chain=CHAIN,
            base_token=pool["token_address"],
            quote_token=SOL_TOKEN_ADDRESS,
            txns={},
        ),
    )

    mdd_df = pd.DataFrame(
        {
            f"{freq}_{before}": meme.get_mdd(freq, before)
            for freq, before in (
                [("1h", _) for _ in range(1, 13, 1)]
                # + [("1min", _) for _ in range(1, 61, 5)]
            )
        },
        index=[0],
    )

    mdd_list.append(mdd_df)

mdd_df = pd.concat(mdd_list, ignore_index=True)
