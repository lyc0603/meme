"""Script to plot the learning curve of all trader profits (raw + smoothed, no confidence bands)."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from environ.constants import PROCESSED_DATA_PATH, FIGURE_PATH

# Constants
CRITICAL_VAL = 2.576
FONT_SIZE = 18

# Load and filter data
pj_pft = pd.read_csv(PROCESSED_DATA_PATH / "trader_project_profits.csv")
pj_pft.drop_duplicates(subset=["trader_address", "meme"], keep="last", inplace=True)

winner = pj_pft.loc[
    (pj_pft["meme_num"] < 1000) & (pj_pft["t_stat"] > CRITICAL_VAL)
].dropna()
neutral = pj_pft.loc[
    (pj_pft["meme_num"] < 1000) & (pj_pft["t_stat"].abs() <= CRITICAL_VAL)
].dropna()
loser = pj_pft.loc[
    (pj_pft["meme_num"] < 1000) & (pj_pft["t_stat"] < -CRITICAL_VAL)
].dropna()


def compute_learning_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Compute raw and smoothed mean returns per transaction group."""
    df = df.sort_values(["trader_address", "date"]).copy()
    df["txn_idx"] = df.groupby("trader_address").cumcount()
    df["txn_group"] = df["txn_idx"] // 10

    grouped = df.groupby("txn_group")["ret"].mean().reset_index()
    return grouped


# Compute learning curves
winner_curve = compute_learning_curve(winner)
neutral_curve = compute_learning_curve(neutral)
loser_curve = compute_learning_curve(loser)

# Plotting
fig, ax = plt.subplots(figsize=(10, 7))

# Raw lines
ax.plot(
    winner_curve["txn_group"],
    winner_curve["ret"],
    color="green",
    linewidth=2,
    label="Winner",
)
ax.plot(
    neutral_curve["txn_group"],
    neutral_curve["ret"],
    color="gray",
    linewidth=2,
    label="Neutral",
)
ax.plot(
    loser_curve["txn_group"],
    loser_curve["ret"],
    color="red",
    linewidth=2,
    label="Loser",
)

# Axis formatting
ax.set_xlabel("Every 10 Meme Coin Projects", fontsize=FONT_SIZE)
ax.set_ylabel("Average Return", fontsize=FONT_SIZE)
ax.tick_params(axis="both", labelsize=FONT_SIZE)

formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))
ax.yaxis.set_major_formatter(formatter)
ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE)

# Legend (2 rows, 3 columns)
legend = ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=3,
    frameon=True,
    fontsize=FONT_SIZE,
)
legend.get_frame().set_edgecolor("black")

# Grid and layout
ax.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout(rect=[0, 0.07, 1, 0.95])

# Save and show
plt.savefig(FIGURE_PATH / "profit_learning.pdf", bbox_inches="tight")
plt.show()
