"""Process MDD (Maximum Drawdown) for pools on Raydium."""

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


from environ.data_class import NewTokenPool
from environ.meme_analyzer import MemeAnalyzer
from environ.sol_fetcher import import_pool
from environ.constants import PROCESSED_DATA_PATH

CHAIN = "raydium"
NUM_OF_OBSERVATIONS = 1000
SOL_TOKEN_ADDRESS = "So11111111111111111111111111111111111111112"
plt.rcParams["figure.figsize"] = (6, 4)

mdd_list = []

FREQ_DICT = {
    "1 Min": {"freq": "1min", "before": 1},
    "5 Mins": {"freq": "1min", "before": 5},
    "10 Mins": {"freq": "1min", "before": 10},
    "15 Mins": {"freq": "1min", "before": 15},
    "30 Mins": {"freq": "1min", "before": 30},
    "1 Hour": {"freq": "1h", "before": 1},
    "6 Hours": {"freq": "1h", "before": 6},
    "12 Hours": {"freq": "1h", "before": 12},
}

for pool in tqdm(
    import_pool(
        CHAIN,
        NUM_OF_OBSERVATIONS,
    )
):
    token_add = pool["token_address"]
    meme = MemeAnalyzer(
        NewTokenPool(
            token0=SOL_TOKEN_ADDRESS,
            token1=token_add,
            fee=0,
            pool_add=token_add,
            block_number=0,
            chain=CHAIN,
            base_token=token_add,
            quote_token=SOL_TOKEN_ADDRESS,
            txns={},
        ),
    )

    # out_degree_herf, in_degree_herf = meme.analyze_non_swap_transfer_graph()

    mdd_df = pd.DataFrame(
        {
            **{
                name: meme.get_mdd(info["freq"], info["before"])
                for name, info in FREQ_DICT.items()
            },
            **{
                "duration": meme.migration_duration,
                "unique_address": meme.get_unique_swapers(),
                "unique_transfer": len(meme.non_swap_transfer_hash),
                "holding_herf": meme.get_holdings_herf(),
                "dev_transfer": meme.dev_transfer,
                "dev_txn": meme.dev_txn,
                # "dev_transfer_amount": meme.dev_transfer_amount,
                "bundle": meme.get_block_bundle_herf(),
                # "transfer_amount": meme.get_non_swap_transfer_amount(),
                # "degree": meme.analyze_non_swap_transfer_graph(),
                "max_same_txn": meme.get_max_same_txn_per_swaper(),
                "total_txn": len(meme.txn),
            },
        },
        index=[0],
    )

    mdd_list.append(mdd_df)

mdd_df = pd.concat(mdd_list, ignore_index=True)
mdd_df.to_csv(
    f"{PROCESSED_DATA_PATH}/mdd.csv",
    index=False,
)
