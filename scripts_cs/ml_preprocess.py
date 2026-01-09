"""Script to preprocess the ML data"""

import pandas as pd

from environ.constants import PROCESSED_DATA_CS_PATH

# Load the data
df = pd.read_csv(PROCESSED_DATA_CS_PATH / "trader_features_merged.csv")
df["date"] = pd.to_datetime(df["date"])
df.dropna(inplace=True)

# Binary label
df["label_cls"] = (df["label"] > 0).astype(int)

# Chronological split
df = df.sort_values("date")
cutoff_1 = df["date"].quantile(0.70)
cutoff_2 = df["date"].quantile(0.85)

train_df = df[df["date"] <= cutoff_1]
val_df = df[(df["date"] > cutoff_1) & (df["date"] <= cutoff_2)]
test_df = df[df["date"] > cutoff_2]

X_train, y_train = train_df, train_df["label_cls"]
X_val, y_val = val_df, val_df["label_cls"]
X_test, y_test = test_df, test_df["label_cls"]
