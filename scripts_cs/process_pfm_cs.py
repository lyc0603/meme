"""Script to get the performance of meme coin projects"""

import numpy as np
import pandas as pd
import json
from environ.constants import PROCESSED_DATA_PATH, PROCESSED_DATA_CS_PATH

# Load data
pfm = pd.read_csv(f"{PROCESSED_DATA_PATH}/pfm.csv")

with open(PROCESSED_DATA_PATH / "meme_project_ratio.json", "r", encoding="utf-8") as f:
    meme_project_ratio = json.load(f)

# Resample per-category with exact counts
resample_pfm = []
for chain, n_rows in meme_project_ratio.items():
    chain_data = pfm[pfm["category"] == chain]
    if not chain_data.empty:
        resampled = chain_data.sample(n=int(n_rows), replace=False, random_state=42)
        resample_pfm.append(resampled)
pfm = pd.concat(resample_pfm, ignore_index=True)

# Winsorize
pfm["max_ret"] = pfm["max_ret"].clip(upper=pfm["max_ret"].quantile(0.99))
pfm["dump_duration"] = pfm["dump_duration"].clip(upper=pfm["dump_duration"].quantile(0.99))

pfm.to_csv(f"{PROCESSED_DATA_CS_PATH}/pfm_cs.csv", index=False)