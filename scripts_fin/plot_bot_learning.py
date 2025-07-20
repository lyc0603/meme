"""Script to plot the learning curve of all trader profits (raw + smoothed, no confidence bands)."""

import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from environ.constants import PROCESSED_DATA_PATH, FIGURE_PATH
from tqdm import tqdm
from environ.constants import NAMING_DICT

# Constants
CRITICAL_VAL = 2.576
FONT_SIZE = 18
BOT_LIST = ["launch_bundle", "volume_bot", "sniper_bot"]


# Load trader token list
with open(
    PROCESSED_DATA_PATH / "kol_non_kol_traded_tokens.json", "r", encoding="utf-8"
) as f:
    token_list = json.load(f)

# Load bot token mapping
with open(PROCESSED_DATA_PATH / "bot_token_map.json", "r", encoding="utf-8") as file:
    bot_token_map = json.load(file)

bot_token_df = pd.DataFrame.from_dict(bot_token_map, orient="index").reset_index()
bot_token_df.rename(columns={"index": "token_address"}, inplace=True)


# Load all trader profits
pj_pft = pd.read_csv(PROCESSED_DATA_PATH / "trader_project_profits.csv")
pj_pft.drop_duplicates(subset=["trader_address", "meme"], keep="last", inplace=True)
pj_pft = pj_pft.loc[pj_pft["meme_num"] < 1000]
pj_pft = pj_pft.loc[pj_pft["trader_address"].isin(token_list)]

pj_pft.rename(columns={"meme": "token_address"}, inplace=True)
pj_pft = pj_pft.merge(bot_token_df, how="left", on="token_address").dropna()

winner = pj_pft.loc[(pj_pft["t_stat"] > CRITICAL_VAL)].dropna(subset=["ret", "t_stat"])
non_winner = pj_pft.loc[pj_pft["t_stat"] <= CRITICAL_VAL]


def compute_learning_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Compute raw and smoothed mean returns per transaction group."""
    df = df.sort_values(["trader_address", "date"]).copy()
    df["txn_idx"] = df.groupby("trader_address").cumcount()
    df["txn_group"] = df["txn_idx"] // 1
    # keep first 100 meme coins
    df = df.loc[df["txn_group"] < 100]

    grouped = (
        df.groupby("txn_group")[["launch_bundle", "volume_bot", "sniper_bot"]]
        .mean()
        .reset_index()
    )
    return grouped


# Compute learning curves
winner_curve = compute_learning_curve(winner)
non_winner_curve = compute_learning_curve(non_winner)
WINDOW = 10

for bot in BOT_LIST:
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot raw curves (transparent)
    ax.plot(
        winner_curve["txn_group"],
        winner_curve[bot],
        color="green",
        linewidth=2,
        alpha=0.1,
        label="Winner",
    )
    ax.plot(
        non_winner_curve["txn_group"],
        non_winner_curve[bot],
        color="red",
        linewidth=2,
        alpha=0.1,
        label="Non-Winner",
    )

    # Plot smoothed curves
    ax.plot(
        winner_curve["txn_group"],
        winner_curve[bot].rolling(window=WINDOW, min_periods=1).mean(),
        color="green",
        linewidth=2,
        label="Winner 10 Projects Moving Average",
    )
    ax.plot(
        non_winner_curve["txn_group"],
        non_winner_curve[bot].rolling(window=WINDOW, min_periods=1).mean(),
        color="red",
        linewidth=2,
        label="Non-Winner 10 Projects Moving Average",
    )

    # Axis formatting
    ax.set_xlabel("Meme Coin Projects", fontsize=FONT_SIZE)
    ax.set_ylabel(f"Average {NAMING_DICT[bot]} Probability", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)

    # formatter = ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((0, 0))
    # ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE)

    # Legend
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=True,
        fontsize=FONT_SIZE,
    )
    legend.get_frame().set_edgecolor("black")

    # Grid and layout
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])

    # Save and/or show
    plt.savefig(FIGURE_PATH / f"{bot}_learning_curve.pdf", bbox_inches="tight")
    plt.show()
