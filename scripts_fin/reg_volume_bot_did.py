"""Build the cohorts for wash trading analysis"""

import json
import pickle
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm
import pyfixest as pf

from environ.constants import PROCESSED_DATA_PATH, TABLE_PATH, WINDOW
from environ.utils import asterisk

# Constants
CHAINS = [
    "raydium",
    "pre_trump_raydium",
    "pumpfun",
    "pre_trump_pumpfun",
    "no_one_care",
    "pre_trump_no_one_care",
]

CPU_COUNT = 10
NAMING_DICT = {
    "log_number_of_traders": "$\\text{Number of Traders}_{i}$",
    "treat_coef": "$\\text{Treat}$",
    "treat_stderr": "",
    "post_coef": "$\\text{Post}$",
    "post_stderr": "",
    "treat_post_coef": "$\\text{Treat} \\times \\text{Post}$",
    "treat_post_stderr": "",
    "obs": "$\\text{Observations}$",
    "r2": "$R^2$",
    "cohort_project_fe": "$\\text{Cohort-Project FE}$",
    "cohort_time_fe": "$\\text{Cohort-Time FE}$",
}


# Load project ratios
with open(PROCESSED_DATA_PATH / "meme_project_ratio.json", "r", encoding="utf-8") as f:
    meme_project_ratio = json.load(f)

# 1) Load & preprocess wash-trade data
wt_list = []
for chain in CHAINS:
    df = pd.read_csv(f"{PROCESSED_DATA_PATH}/wt_cm/wash_trading_{chain}.csv")
    df = (
        df.dropna()
        .dropna()
        .assign(
            category=chain,
            weight=meme_project_ratio[chain],
            first_trade_time=lambda x: pd.to_datetime(x["first_trade_time"]),
            launch_time=lambda x: pd.to_datetime(x["launch_time"]),
        )
    )
    df["delta_time_int"] = (
        (df["first_trade_time"] - df["launch_time"]).dt.total_seconds() / 60
    ).astype(int)
    # df = df[df["delta_time_int"] >= -WINDOW[0]]
    # First wash trade only
    df = df.sort_values(["token_address", "first_trade_time"]).drop_duplicates(
        subset=["token_address"], keep="first"
    )
    print("chain:", chain, "num of wash bots:", len(df))
    wt_list.append(df)
wt = pd.concat(wt_list, ignore_index=True)

# Persist delta_set
unique_deltas = wt["delta_time_int"].unique()
delta_set = {d + o for d in unique_deltas for o in WINDOW}
with open(PROCESSED_DATA_PATH / "wt_cm" / "delta_set_wt.pkl", "wb") as f:
    pickle.dump(delta_set, f)

# 2) Load time-trader series
tt = {
    chain: json.load(
        open(
            PROCESSED_DATA_PATH / "wt_cm" / f"time_traders_{chain}.json",
            "r",
            encoding="utf-8",
        )
    )
    for chain in CHAINS
}

# 3) Load volume_bot flags
pfm = pd.read_csv(PROCESSED_DATA_PATH / "pfm.csv")
volume_bot = set(pfm.loc[pfm["volume_bot"] == 1, "token_address"])


# Function to process one cohort
def process_cohort(item):
    """Process a single cohort of wash trading data."""
    cohort_id, cohort_df = item
    treats, controls = [], []

    # Build treated
    for _, row in cohort_df.iterrows():
        token = row["token_address"]
        cat = row["category"]
        # check full window exists
        # keys = set(tt[cat].get(token, {}).keys())
        # req = {str(cohort_id + o) for o in WINDOW}
        # if req.issubset(keys):
        for offset in WINDOW:
            if str(cohort_id + offset) in tt[cat][token].keys():
                treats.append(
                    {
                        "cohort": cohort_id,
                        "token": token,
                        "time": cohort_id + offset,
                        "cohort*token": f"{cohort_id}_{token}",
                        "cohort*time": f"{cohort_id}_{cohort_id + offset}",
                        "treat": 1,
                        "post": int(offset >= 0),
                        "offset": offset,
                        "log_number_of_traders": tt[cat][token][
                            str(cohort_id + offset)
                        ]["log_number_of_traders"],
                        "log_ret": tt[cat][token][str(cohort_id + offset)]["log_ret"],
                        "price": tt[cat][token][str(cohort_id + offset)]["price"],
                        "category": cat,
                        "weight": row["weight"],
                    }
                )

    # Build controls
    for cat in CHAINS:
        for token, series in tt[cat].items():
            if token in volume_bot:
                continue
            # keys = set(series.keys())
            # req = {str(cohort_id + o) for o in WINDOW}
            # if req.issubset(keys):
            for offset in WINDOW:
                if str(cohort_id + offset) in series:
                    controls.append(
                        {
                            "cohort": cohort_id,
                            "token": token,
                            "time": cohort_id + offset,
                            "cohort*token": f"{cohort_id}_{token}",
                            "cohort*time": f"{cohort_id}_{cohort_id + offset}",
                            "treat": 0,
                            "post": int(offset >= 0),
                            "offset": offset,
                            "log_number_of_traders": series[str(cohort_id + offset)][
                                "log_number_of_traders"
                            ],
                            "log_ret": series[str(cohort_id + offset)]["log_ret"],
                            "price": series[str(cohort_id + offset)]["price"],
                            "category": cat,
                            "weight": meme_project_ratio[cat],
                        }
                    )

    # Combine
    if treats and controls:
        df = pd.concat(
            [pd.DataFrame(treats), pd.DataFrame(controls)], ignore_index=True
        )
        return df
    return None


# 4) Parallel execution
tasks = list(wt.groupby("delta_time_int"))

with Pool(processes=CPU_COUNT) as pool:
    results = list(tqdm(pool.imap(process_cohort, tasks), total=len(tasks)))

# Filter & concat
cohort_dfs = [df for df in results if df is not None]
cohorts_df = pd.concat(cohort_dfs, ignore_index=True)
cohorts_df.to_csv(PROCESSED_DATA_PATH / "reg_volume_did.csv", index=False)

# # implement the Cohort DiD
# model = pf.feols(
#     "log_number_of_traders ~ treat*post|cohort*time+cohort*token",
#     data=cohorts_df,
#     demeaner_backend="numba",
#     # vcov={"CRV3": "cohort*token"},
#     weights="weight",
# )

# TABLE_KEYS = [
#     "treat_coef",
#     "treat_stderr",
#     "post_coef",
#     "post_stderr",
#     "treat_post_coef",
#     "treat_post_stderr",
#     "obs",
#     "r2",
#     "cohort_project_fe",
#     "cohort_time_fe",
# ]

# res_dict = {k: [] for k in TABLE_KEYS}


# # Extract regression results
# est = model
# res_dict["treat_coef"].append(
#     f"{est.coef()['treat']:.2f}{asterisk(est.pvalue()['treat'])}"
# )
# res_dict["treat_stderr"].append(f"({est.tstat()['treat']:.2f})")
# res_dict["post_coef"].append(
#     f"{est.coef()['post']:.2f}{asterisk(est.pvalue()['post'])}"
# )
# res_dict["post_stderr"].append(f"({est.tstat()['post']:.2f})")
# res_dict["treat_post_coef"].append(
#     f"{est.coef()['treat:post']:.2f}{asterisk(est.pvalue()['treat:post'])}"
# )
# res_dict["treat_post_stderr"].append(f"({est.tstat()['treat:post']:.2f})")
# res_dict["obs"].append(f"{est._N}")
# res_dict["r2"].append(f"{est._adj_r2:.3f}")
# res_dict["cohort_project_fe"].append("Y")
# res_dict["cohort_time_fe"].append("Y")

# # Render LaTeX line-by-line
# latex_path = TABLE_PATH / "reg_volume_did.tex"
# with open(latex_path, "w", encoding="utf-8") as f:
#     f.write("\\begin{tabular}{lc}\n")
#     f.write("\\toprule\n")
#     f.write(f"& {NAMING_DICT.get('log_number_of_traders', 'Number of Traders')} \\\\\n")
#     f.write("\\midrule\n")
#     for var in [
#         "treat_post_coef",
#         "treat_post_stderr",
#         "treat_coef",
#         "treat_stderr",
#         "post_coef",
#         "post_stderr",
#         "obs",
#         "r2",
#         "cohort_time_fe",
#         "cohort_project_fe",
#     ]:
#         label = NAMING_DICT.get(var, var)
#         if var in {"obs"}:
#             f.write("\\midrule\n")
#         f.write(f"{label} & {res_dict[var][0]} \\\\\n")
#     f.write("\\bottomrule\n")
#     f.write("\\end{tabular}\n")
