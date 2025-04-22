"""
Script to process the transactions data in the pool
"""

import glob
import os
import pickle

from matplotlib import pyplot as plt
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH

chain = "arbitrum"

pools = set(
    [
        os.path.basename(path).split(".")[0]
        for path in glob.glob(f"{PROCESSED_DATA_PATH}/txn/{chain}/*.pkl")
    ]
)

txns_len = {}

for pool in tqdm(pools, total=len(pools), desc=f"Processing {chain} data"):
    with open(f"{PROCESSED_DATA_PATH}/txn/{chain}/{pool}.pkl", "rb") as f:
        pool_data = pickle.load(f)

    if len(pool_data) < 10:
        continue
    txns_len[pool] = len(pool_data)

# plot the histogram
# plt.hist(txns_len, bins=100)
