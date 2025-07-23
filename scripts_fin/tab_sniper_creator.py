import pandas as pd
from environ.constants import PROCESSED_DATA_PATH, TABLE_PATH

# Load trader-level performance data
pft = pd.read_csv(PROCESSED_DATA_PATH / "pft.csv")
pft = pft.loc[(pft["winner"] == 1) | (pft["loser"] == 1) | (pft["neutral"] == 1)]
pft.drop_duplicates(subset=["token_address", "trader_address"], inplace=True)

pft = pft[
    [
        "token_address",
        "trader_address",
        "winner",
        "loser",
        "neutral",
        "creator",
        "sniper",
    ]
]

# Load project-level metadata and merge
pfm = pd.read_csv(f"{PROCESSED_DATA_PATH}/pfm.csv")
pfm = pft.merge(pfm, how="left", on="token_address")

# # Aggregate to trader level
# pfm = (
#     pfm.groupby(["trader_address"])[["winner", "loser", "neutral", "creator", "sniper"]]
#     .mean()
#     .reset_index()
# )

# # Convert creator and sniper to binary
# pfm["creator"] = pfm["creator"].apply(lambda x: int(x > 0))
# pfm["sniper"] = pfm["sniper"].apply(lambda x: int(x > 0))


# Function to compute winner/loser/neutral percentages within a group
def compute_group_percentages(df: pd.DataFrame, group_var: str) -> pd.Series:
    """Compute percentages of winner, loser, and neutral traders within a group."""
    sub_df = df[df[group_var] == 1]
    counts = {
        "Winner": sub_df["winner"].sum(),
        "Loser": sub_df["loser"].sum(),
        "Neutral": sub_df["neutral"].sum(),
    }
    total = sum(counts.values())
    percentages = {
        k: f"{(v / total * 100):.1f}\%" if total > 0 else "-" for k, v in counts.items()
    }
    return pd.Series(percentages, name=group_var.capitalize())


# Compute percentages
creator_pct = compute_group_percentages(pfm, "creator")
sniper_pct = compute_group_percentages(pfm, "sniper")

# Combine into a table
result = pd.concat([creator_pct, sniper_pct], axis=1)

# Generate LaTeX table
lines = [
    "\\begin{tabular}{lcc}",
    "\\toprule",
    "Trader Type & Creator & Sniper \\\\",
    "\\midrule",
]

for row in result.index:
    lines.append(
        f"{row} & {result.loc[row, 'Creator']} & {result.loc[row, 'Sniper']} \\\\"
    )

lines.extend(["\\bottomrule", "\\end{tabular}"])

# Write LaTeX file
with open(TABLE_PATH / "sniper_creator.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
