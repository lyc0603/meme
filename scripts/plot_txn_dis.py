"""
Script to process the transactions data in the pool
"""

import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from environ.constants import (
    PROCESSED_DATA_PATH,
    TRUMP_BLOCK,
    UNISWAP_V3_FACTORY_DICT,
    FIGURE_PATH,
)
from environ.db import fetch_native_pool_since_block

txns_len = []

for chain in [
    "ethereum",
    "base",
    "polygon",
    "bnb",
    "avalanche",
    "arbitrum",
    "optimism",
    "blast",
]:

    pools = [
        pool["args"]["pool"]
        for pool in fetch_native_pool_since_block(
            chain, TRUMP_BLOCK[chain], pool_number=100
        )
    ]

    for pool in tqdm(pools, total=len(pools), desc=f"Processing {chain} data"):
        with open(f"{PROCESSED_DATA_PATH}/txn/{chain}/{pool}.pkl", "rb") as f:
            pool_data = pickle.load(f)

        txns_len.append(
            pd.DataFrame(
                {
                    "Chain": UNISWAP_V3_FACTORY_DICT[chain]["name"],
                    "# of Transactions in 12 Hours": len(pool_data),
                },
                index=[0],
            )
        )
txns_len = pd.concat(txns_len, ignore_index=True)

plt.figure(figsize=(8, 3))
sns.set_style("white")
iris = sns.load_dataset("iris")
palette = {v["name"]: v["color"] for _, v in UNISWAP_V3_FACTORY_DICT.items()}
ax = sns.violinplot(
    x="Chain",
    y="# of Transactions in 12 Hours",
    data=txns_len,
    hue="Chain",
    dodge=False,
    palette=palette,
    scale="width",
    inner=None,
    cut=0,
    width=0.6,
)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    bbox = violin.get_paths()[0].get_extents()
    x0, y0, width, height = bbox.bounds
    violin.set_clip_path(
        plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData)
    )

for violin in ax.collections[:5]:  # one for each violin
    violin.set_alpha(0.7)

sns.boxplot(
    x="Chain",
    y="# of Transactions in 12 Hours",
    data=txns_len,
    saturation=1,
    showfliers=False,
    width=0.3,
    boxprops={"zorder": 3, "facecolor": "none"},
    ax=ax,
)
old_len_collections = len(ax.collections)
sns.stripplot(
    x="Chain",
    y="# of Transactions in 12 Hours",
    data=txns_len,
    hue="Chain",
    palette=palette,
    dodge=False,
    ax=ax,
)
for dots in ax.collections[old_len_collections:]:
    dots.set_alpha(0.5)
    dots.set_offsets(dots.get_offsets() + np.array([0.12, 0]))
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_ylim(bottom=0)
ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.8)
ax.set_xlabel("")
plt.savefig(
    f"{FIGURE_PATH}/txn_dis.pdf",
    bbox_inches="tight",
    dpi=300,
)
plt.show()
