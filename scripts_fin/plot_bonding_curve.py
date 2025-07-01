"""Script to plot a bonding curve for a token sale on Solana."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from environ.constants import FIGURE_PATH

FONT_SIZE = 18

# bonding curve constants
initial_sol = 30
initial_tokens = 1_073_000_191
k = initial_sol * initial_tokens


def token_amount(x):
    """Calculate the total tokens received for a given amount of SOL deposited."""
    return initial_tokens - (k / (initial_sol + x))


def price(x):
    """Calculate the price of the token for a given amount of SOL deposited."""
    return (initial_sol + x) ** 2 / k


# range of SOL
x = np.linspace(0, 90, 10)
y_tokens = token_amount(x)
z_price = price(x)

fig, ax1 = plt.subplots(figsize=(10, 7))

# Total tokens (left y)
ax1.plot(
    x,
    y_tokens,
    color="red",
    marker="o",
    markersize=4,
    linewidth=2,
    label="Total Meme Coin Received",
    markeredgewidth=5,
)
ax1.set_xlabel("Total SOL Deposited", fontsize=FONT_SIZE)
ax1.set_ylabel("Total Meme Coin Received", fontsize=FONT_SIZE)
ax1.tick_params(axis="y", labelsize=FONT_SIZE)

formatter1 = ScalarFormatter(useMathText=True)
formatter1.set_scientific(True)
formatter1.set_powerlimits((0, 0))
ax1.yaxis.set_major_formatter(formatter1)

# Price (right y)
ax2 = ax1.twinx()
ax2.plot(
    x,
    z_price,
    color="blue",
    marker="^",
    markersize=4,
    linewidth=2,
    label="Price",
    markeredgewidth=5,
)
ax2.set_ylabel("Price, SOL per Meme Coin", fontsize=FONT_SIZE)
ax2.tick_params(axis="y", labelsize=FONT_SIZE)

# legend in a box below, Stata style
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
all_lines = lines_1 + lines_2
all_labels = labels_1 + labels_2
legend = fig.legend(
    all_lines,
    all_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.07),
    ncol=2,
    frameon=True,
    fontsize=FONT_SIZE,
)
legend.get_frame().set_edgecolor("black")
ax1.tick_params(axis="x", labelsize=FONT_SIZE)

# enlarge offset text font sizes
ax1.yaxis.get_offset_text().set_fontsize(FONT_SIZE)
ax2.yaxis.get_offset_text().set_fontsize(FONT_SIZE)


formatter2 = ScalarFormatter(useMathText=True)
formatter2.set_scientific(True)
formatter2.set_powerlimits((0, 0))
ax2.yaxis.set_major_formatter(formatter2)

initial_price = price(0)
ax2.axhline(y=initial_price, color="blue", linestyle="--", linewidth=2)

yticks = ax2.get_yticks()
new_yticks = np.sort(np.append(yticks, initial_price))
ax2.set_yticks(new_yticks)


# title
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(FIGURE_PATH / "bonding_curve.pdf", bbox_inches="tight")
plt.show()
