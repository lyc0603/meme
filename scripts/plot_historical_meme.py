"""
Script to plot historical meme data
"""

import datetime
import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from environ.constants import PROCESSED_DATA_PATH, UNISWAP_V3_FACTORY_DICT

fig1, ax_1 = plt.subplots(figsize=(6, 4))
fig2, ax_2 = plt.subplots(figsize=(6, 4))

for i, (chain, chain_info) in enumerate(UNISWAP_V3_FACTORY_DICT.items()):
    with open(
        f"{PROCESSED_DATA_PATH}/{chain}_historical_meme.jsonl", "r", encoding="utf-8"
    ) as f:
        data_list = [json.loads(line) for line in f]

    date_list = [
        datetime.datetime.strptime(data["date"], "%Y-%m-%d") for data in data_list
    ]
    pool_num_list = [data["pool_num"] for data in data_list]
    token_list = [data["token_num"] for data in data_list]

    ax_1.plot(
        date_list,
        pool_num_list,
        label=(
            chain_info["name"]
            if i < len(UNISWAP_V3_FACTORY_DICT) // 2
            else "_nolegend_"
        ),
        color=chain_info["color"],
    )
    ax_2.plot(
        date_list,
        token_list,
        label=(
            chain_info["name"]
            if i >= len(UNISWAP_V3_FACTORY_DICT) // 2
            else "_nolegend_"
        ),
        color=chain_info["color"],
    )

zm_1 = ax_1.inset_axes([0.15, 0.37, 0.6, 0.6])
ax_1.indicate_inset_zoom(zm_1, edgecolor="black", linestyle="--")
zm_2 = ax_2.inset_axes([0.15, 0.37, 0.6, 0.6])
ax_2.indicate_inset_zoom(zm_2, edgecolor="black", linestyle="--")


for chain, chain_info in UNISWAP_V3_FACTORY_DICT.items():
    with open(
        f"{PROCESSED_DATA_PATH}/{chain}_historical_meme.jsonl", "r", encoding="utf-8"
    ) as f:
        data_list = [json.loads(line) for line in f]

    date_list = [
        datetime.datetime.strptime(data["date"], "%Y-%m-%d") for data in data_list
    ]
    pool_num_list = [data["pool_num"] for data in data_list]
    token_list = [data["token_num"] for data in data_list]

    zm_1.plot(
        date_list,
        pool_num_list,
        label=chain_info["name"],
        color=chain_info["color"],
    )
    zm_2.plot(
        date_list,
        token_list,
        label=chain_info["name"],
        color=chain_info["color"],
    )

    zm_1.set_ylim(0, 35000)
    zm_2.set_ylim(0, 25000)

    for zm in [zm_1, zm_2]:
        zm.tick_params(axis="x", labelrotation=90, labelsize=8)
        zm.tick_params(axis="y", labelsize=8)
        zm.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        zm.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        zm.grid(True)

for ax in [ax_1, ax_2]:
    ax.tick_params(axis="x", labelrotation=90, labelsize=10)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=4, frameon=False)

ax_1.set_ylabel("# of Uniswap V3 Pools", fontsize=12)
ax_2.set_ylabel("# of Uniswap V3 Tokens", fontsize=12)

fig1.tight_layout()
fig2.tight_layout()

fig1.savefig(f"{PROCESSED_DATA_PATH}/historical_meme_pool.pdf", dpi=300)
fig2.savefig(f"{PROCESSED_DATA_PATH}/historical_meme_token.png", dpi=300)
