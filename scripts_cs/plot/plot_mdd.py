"""Maxium Drawdown Detector (MDD) for Meme"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import numpy as np
from matplotlib import font_manager
import statsmodels.api as sm


from environ.data_class import NewTokenPool
from environ.meme_analyzer import MemeAnalyzer
from environ.sol_fetcher import import_pool
from environ.constants import FIGURE_PATH, PROCESSED_DATA_PATH
from pathlib import Path

CHAIN = "raydium"
NUM_OF_OBSERVATIONS = 1000
SOL_TOKEN_ADDRESS = "So11111111111111111111111111111111111111112"
plt.rcParams["figure.figsize"] = (6, 4)

processed_path = Path(PROCESSED_DATA_PATH)
mdd_path = processed_path / "mdd.csv"
mdd_df = pd.read_csv(mdd_path)

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

# plot the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    mdd_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    vmin=-1,
    vmax=1,
    cbar_kws={"shrink": 0.8},
)

# linear regression for each column against duration
for name in FREQ_DICT.keys():
    sm_model = sm.OLS(
        mdd_df[name],
        sm.add_constant(mdd_df["duration"]),
    ).fit()
    print(f"{name} vs Duration: {sm_model.summary()}")

# linear regression for each column against unique address
for name in FREQ_DICT.keys():
    sm_model = sm.OLS(
        mdd_df[name],
        sm.add_constant(mdd_df["unqiue_address"]),
    ).fit()
    print(f"{name} vs Unique Address: {sm_model.summary()}")


# for name, _ in FREQ_DICT.items():
#     survival_dict = {
#         "mdd": [],
#         "freq": [],
#     }
#     # generate a survival plot
#     for i in range(-100, 1, 1):
#         survival_dict["mdd"].append(i)
#         survival_dict["freq"].append((mdd_df[name] > i / 100).sum() / len(mdd_df) * 100)

#     survival_df = pd.DataFrame(survival_dict)
#     plt.plot(
#         survival_df["mdd"],
#         survival_df["freq"],
#         label=name,
#     )
# plt.xticks(
#     fontsize=12,
#     fontweight="bold",
# )
# plt.yticks(
#     fontsize=12,
#     fontweight="bold",
# )

# plt.xlabel("Maximum Drawdown (MDD) (%)", fontsize=12, fontweight="bold")
# plt.ylabel("Percentage of Coin with MDD > x (%)", fontsize=12, fontweight="bold")
# plt.legend(
#     frameon=False,
#     ncols=2,
#     title="Time After Migration",
#     prop={"size": 10, "weight": "bold"},
#     title_fontproperties=font_manager.FontProperties(weight="bold", size=10),
# )
# plt.grid()
# plt.tight_layout()
# plt.savefig(f"{FIGURE_PATH}/mdd_survival.pdf", dpi=300)
# plt.show()
