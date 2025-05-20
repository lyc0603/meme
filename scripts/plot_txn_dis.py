"""
Script to process the transactions data in the pool
"""

import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

from environ.constants import (
    PROCESSED_DATA_PATH,
    TRUMP_BLOCK,
    UNISWAP_V3_FACTORY_DICT,
    FIGURE_PATH,
)
from environ.db import fetch_native_pool_since_block
from environ.sol_fetcher import import_pool

POOL_NUM = 100
txns_len = []

# Step 1: Load all transaction lengths
for chain in [
    "raydium",
    "pumpfun",
    "ethereum",
    "base",
    "polygon",
]:

    pools = (
        [
            pool["args"]["pool"]
            for pool in fetch_native_pool_since_block(
                chain, TRUMP_BLOCK[chain], pool_number=POOL_NUM
            )
        ]
        if chain not in ["pumpfun", "raydium"]
        else [
            _[0]
            for _ in import_pool(
                chain,
                POOL_NUM,
            )
        ]
    )

    if chain in ["pumpfun", "raydium"]:
        chain_label = "Solana"
        protocol_label = chain.capitalize()
    else:
        chain_label = chain.capitalize()
        protocol_label = "Uniswap v3"

    for pool in tqdm(pools, total=len(pools), desc=f"Processing {chain} data"):
        try:
            with open(f"{PROCESSED_DATA_PATH}/txn/{chain}/{pool}.pkl", "rb") as f:
                pool_data = pickle.load(f)

            txns_len.append(
                pd.DataFrame(
                    {
                        "Chain": chain_label,
                        "Protocol": protocol_label,
                        "# of Transactions in 12 Hours": len(pool_data),
                        "Color": UNISWAP_V3_FACTORY_DICT[chain]["color"],
                        "Name": UNISWAP_V3_FACTORY_DICT[chain]["name"],
                    },
                    index=[0],
                )
            )
        except FileNotFoundError:
            continue

txns_len = pd.concat(txns_len, ignore_index=True)
txns_len["Label"] = txns_len["Chain"] + "\n" + txns_len["Protocol"]

# Full palette
palette = {
    row["Label"]: row["Color"]
    for _, row in txns_len[["Label", "Color"]].drop_duplicates().iterrows()
}

# Subset for zoomed-in labels
zoom_chains = ["Solana", "Ethereum", "Base", "Polygon"]
zoom_protocols = ["Pumpfun", "Uniswap v3", "Uniswap v3", "Uniswap v3"]
zoom_labels = [f"{c}\n{p}" for c, p in zip(zoom_chains, zoom_protocols)]
zoom_df = txns_len[txns_len["Label"].isin(zoom_labels)]
zoom_palette = {k: palette[k] for k in zoom_labels}

# Step 2: Plot main + inset together
fig, ax = plt.subplots(figsize=(6, 4))
sns.set_style("white")

# Main plot
sns.violinplot(
    x="Label",
    y="# of Transactions in 12 Hours",
    data=txns_len,
    hue="Label",
    dodge=False,
    palette=palette,
    scale="width",
    inner=None,
    cut=0,
    width=0.6,
    ax=ax,
)

# Clip violins
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    bbox = violin.get_paths()[0].get_extents()
    x0, y0, width, height = bbox.bounds
    violin.set_clip_path(Rectangle((x0, y0), width / 2, height, transform=ax.transData))

for violin in ax.collections[: len(palette)]:
    violin.set_alpha(0.7)

sns.boxplot(
    x="Label",
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
    x="Label",
    y="# of Transactions in 12 Hours",
    data=txns_len,
    hue="Label",
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
ax.grid(True, axis="y", linestyle="-", linewidth=1, alpha=0.8)
ax.set_xlabel("")
ax.set_ylabel("# of Transactions in 12 Hours", fontsize=12, fontweight="bold")
ax.yaxis.set_label_coords(-0.15, 0.4)
ax.tick_params(axis="both", which="major", labelsize=10)
for label in ax.get_xticklabels():
    label.set_fontweight("bold")
    label.set_linespacing(1.5)
for label in ax.get_yticklabels():
    label.set_fontweight("bold")
if ax.legend_ is not None:
    ax.legend_.remove()

# Step 3: Inset with aligned x-axis categories (in same order)
inset_ax = fig.add_axes([0.35, 0.32, 0.58, 0.62])  # [left, bottom, width, height]

# Set category order to match full x-axis
category_order = [
    label.get_text()
    for label in ax.get_xticklabels()
    if label.get_text() in zoom_labels
]

sns.violinplot(
    x="Label",
    y="# of Transactions in 12 Hours",
    data=zoom_df,
    hue="Label",
    order=category_order,
    dodge=False,
    palette=zoom_palette,
    scale="width",
    inner=None,
    cut=0,
    width=0.6,
    ax=inset_ax,
)
for violin in inset_ax.collections:
    bbox = violin.get_paths()[0].get_extents()
    x0, y0, width, height = bbox.bounds
    violin.set_clip_path(
        Rectangle((x0, y0), width / 2, height, transform=inset_ax.transData)
    )
for violin in inset_ax.collections[: len(zoom_palette)]:
    violin.set_alpha(0.7)

sns.boxplot(
    x="Label",
    y="# of Transactions in 12 Hours",
    data=zoom_df,
    order=category_order,
    saturation=1,
    showfliers=False,
    width=0.3,
    boxprops={"zorder": 3, "facecolor": "none"},
    ax=inset_ax,
)

old_len_collections = len(inset_ax.collections)
sns.stripplot(
    x="Label",
    y="# of Transactions in 12 Hours",
    data=zoom_df,
    order=category_order,
    hue="Label",
    palette=zoom_palette,
    dodge=False,
    ax=inset_ax,
)
for dots in inset_ax.collections[old_len_collections:]:
    dots.set_alpha(0.5)
    dots.set_offsets(dots.get_offsets() + np.array([0.12, 0]))

inset_ax.set_ylim(bottom=0)
inset_ax.set_xlabel("")
inset_ax.set_ylabel("")
inset_ax.tick_params(axis="both", which="major", labelsize=8)
inset_ax.grid(True, axis="y", linestyle="-", linewidth=1, alpha=0.8)
for label in inset_ax.get_xticklabels():
    label.set_fontweight("bold")
    label.set_linespacing(1.5)
for label in inset_ax.get_yticklabels():
    label.set_fontweight("bold")

if inset_ax.legend_ is not None:
    inset_ax.legend_.remove()

# Save and show
plt.tight_layout()
plt.savefig(f"{FIGURE_PATH}/txn_dis.pdf", bbox_inches="tight", dpi=300)
plt.show()
