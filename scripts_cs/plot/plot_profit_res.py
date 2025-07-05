"""Script to plot cumulative profit for copy trading with a single y-axis, no markers, bonding curve style."""

import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from environ.constants import PROCESSED_DATA_PATH, FIGURE_PATH

FONT_SIZE = 18

# Load token results
with open(PROCESSED_DATA_PATH / "trader_analyzer.pkl", "rb") as f:
    wallet_stats = pickle.load(f)

df = pd.DataFrame(wallet_stats)

df_token = (
    df[["token_address", "migration_block"]]
    .drop_duplicates()
    .sort_values("migration_block", ascending=True)
    .reset_index(drop=True)
)

df_token = df_token.iloc[50:]

wallet = pd.read_csv(PROCESSED_DATA_PATH / "tab" / "wallet.csv")
wallet = wallet.loc[wallet["prediction"]]
wallet = wallet.groupby("eval_token")["profit_mean"].sum().reset_index()
wallet.rename(
    columns={"eval_token": "token_address", "profit_mean": "profit"}, inplace=True
)

wallet = df_token.merge(wallet, on="token_address", how="left")
wallet["profit"] = wallet["profit"].fillna(0.0)

# load coin agent results
records = {
    "token_address": [],
    "good_farming": [],
    "reasoning": [],
}

with open(
    PROCESSED_DATA_PATH / "batch" / "coin_agent.jsonl", "r", encoding="utf-8"
) as f:
    for line in f:
        data = json.loads(line)
        token_address = data["custom_id"]
        content = json.loads(
            data["response"]["body"]["choices"][0]["message"]["content"]
        )
        records["token_address"].append(token_address)
        records["good_farming"].append(content["good_farming"])
        records["reasoning"].append(content["reasoning"])

coin = pd.DataFrame(records)
coin_list = coin.loc[coin["good_farming"], "token_address"].tolist()

wallet["profit_wo_wallet"] = wallet.apply(
    lambda row: row["profit"] if row["token_address"] in coin_list else 0.0,
    axis=1,
)

wallet["profit_cum"] = wallet["profit"].cumsum()
wallet["profit_wo_wallet_cum"] = wallet["profit_wo_wallet"].cumsum()

# --- plot on a single y-axis
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    wallet.index,
    wallet["profit_wo_wallet_cum"],
    color="blue",
    linewidth=2,
)

# add grid with alpha
ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

ax.set_xlabel("# of Migration Meme Coin", fontsize=FONT_SIZE)
ax.set_ylabel("Cumulative Wallet Profit", fontsize=FONT_SIZE)
ax.tick_params(axis="both", labelsize=FONT_SIZE)

formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))
ax.yaxis.set_major_formatter(formatter)
ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE)

# # legend in a box below
# lines, labels = ax.get_legend_handles_labels()
# legend = fig.legend(
#     lines,
#     labels,
#     loc="upper center",
#     bbox_to_anchor=(0.5, 0.07),
#     ncol=1,
#     frameon=True,
#     fontsize=FONT_SIZE,
# )
# legend.get_frame().set_edgecolor("black")

# final layout
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(PROCESSED_DATA_PATH / "profit_cs.pdf", bbox_inches="tight")
plt.show()
