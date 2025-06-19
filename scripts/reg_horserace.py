"""Script to run the horserace regression analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from environ.constants import FREQ_DICT, NAMING_DICT, PROCESSED_DATA_PATH, TABLE_PATH

mdd_df = pd.read_csv(Path(PROCESSED_DATA_PATH) / "ret_mdd.csv")

# Preprocess the data
# size
mdd_df["duration"] = np.log(mdd_df["duration"])
mdd_df["#trader"] = np.log(mdd_df["#trader"])
mdd_df["#txn"] = np.log(mdd_df["#txn"])
mdd_df["#transfer"] = np.log(mdd_df["#transfer"] + 1)

# VIF
for y_var in FREQ_DICT:
    reg_df = mdd_df.loc[mdd_df[f"death_{y_var}"] == 0, :].copy()
    X = reg_df[["duration", "#trader", "#transfer"]].dropna()

    vif_df = pd.DataFrame()
    vif_df["feature"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(f"\nVIF for ret_{y_var}:")
    print(vif_df)

# Collect regression results
res_dict = {}
for y_var in FREQ_DICT:
    reg_df = mdd_df.loc[mdd_df[f"death_{y_var}"] == 0, :].copy()

    # regress size variables
    X = sm.add_constant(reg_df[["duration", "#trader", "#transfer"]])
    y = reg_df[f"ret_{y_var}"]
    model = sm.OLS(y, X).fit()
    print(model.summary())

    # X = sm.add_constant(reg_df[x_var])
    # y = reg_df[f"ret_{y_var}"]
    # model = sm.OLS(y, X).fit()
    # pval = model.pvalues[x_var]
    # res_dict[x_var][f"ret_{y_var}"] = {
    #     "coef": model.params[x_var],
    #     "coef_stderr": model.bse[x_var],
    #     "coef_pval": model.pvalues[x_var],
    #     "con": model.params["const"],
    #     "con_stderr": model.bse["const"],
    #     "con_pval": model.pvalues["const"],
    #     "obs": model.nobs,
    #     "r2": model.rsquared,
    # }
    # res
