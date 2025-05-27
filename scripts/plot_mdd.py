"""Maxium Drawdown Detector (MDD) for Meme"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import numpy as np
from matplotlib import font_manager


from environ.data_class import NewTokenPool
from environ.meme_analyzer import MemeAnalyzer
from environ.sol_fetcher import import_pool
from environ.constants import FIGURE_PATH

CHAIN = "raydium"
NUM_OF_OBSERVATIONS = 100
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

    mdd_df = pd.DataFrame(
        {
            **{
                name: meme.get_mdd(info["freq"], info["before"])
                for name, info in FREQ_DICT.items()
            },
            **{
                "duration": np.log(meme.migration_duration),
                "unqiue_address": np.log(meme.get_unique_swapers()),
            },
        },
        index=[0],
    )

    mdd_list.append(mdd_df)

mdd_df = pd.concat(mdd_list, ignore_index=True)

# # plot the correlation heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(
#     mdd_df.corr(),
#     annot=True,
#     fmt=".2f",
#     cmap="coolwarm",
#     square=True,
#     cbar_kws={"shrink": 0.8},
# )

for name, _ in FREQ_DICT.items():
    survival_dict = {
        "mdd": [],
        "freq": [],
    }
    # generate a survival plot
    for i in range(-100, 1, 1):
        survival_dict["mdd"].append(i)
        survival_dict["freq"].append((mdd_df[name] > i / 100).sum() / len(mdd_df) * 100)

    survival_df = pd.DataFrame(survival_dict)
    plt.plot(
        survival_df["mdd"],
        survival_df["freq"],
        label=name,
    )
plt.xticks(
    fontsize=12,
    fontweight="bold",
)
plt.yticks(
    fontsize=12,
    fontweight="bold",
)

plt.xlabel("Maximum Drawdown (MDD) (%)", fontsize=12, fontweight="bold")
plt.ylabel("Percentage of Coin with MDD > x (%)", fontsize=12, fontweight="bold")
plt.legend(
    frameon=False,
    ncols=2,
    title="Time After Migration",
    prop={"size": 10, "weight": "bold"},
    title_fontproperties=font_manager.FontProperties(weight="bold", size=10),
)
plt.grid()
plt.tight_layout()
plt.savefig(f"{FIGURE_PATH}/mdd_survival.pdf", dpi=300)
plt.show()
