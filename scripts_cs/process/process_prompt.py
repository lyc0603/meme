"""ChatGPT Batch Processing Script"""

import pandas as pd
import json
import os
from tqdm import tqdm
import numpy as np

from environ.data_class import NewTokenPool
from environ.sol_fetcher import import_pool
from environ.constants import SOL_TOKEN_ADDRESS
from environ.data_loader import DataLoader

from environ.constants import PROCESSED_DATA_PATH
from environ.prompt import (
    JSON_SCHEMA_COIN_AGENT,
    # # Transaction Agent
    # COT_TRANSACTION_AGENT,
    # PROMPT_TRANSACTION_AGENT,
    # SYSTEM_INSTRUCTION_TRANSACTION_AGENT,
    # # Comment Agent
    # SYSTEM_INSTRUCTION_COMMENT_AGENT,
    # FEW_SHOT_EXAMPLES_COMMENT_AGENT,
    # PROMPT_COMMENT_AGENT,
    # # Technical Agent
    # COT_TECHNICAL_AGENT,
    # PROMPT_TECHNICAL_AGENT,
    # SYSTEM_INSTRUCTION_TECHNICAL_AGENT,
    # # Coin Agent
    PROMPT_COIN_AGENT,
    SYSTEM_INSTRUCTION_COIN_AGENT,
    COT_COIN_AGENT,
    # # Wallet Agent
    PROMPT_WALLET_AGENT,
    SYSTEM_INSTRUCTION_WALLET_AGENT,
    COT_WALLET_AGENT,
    JSON_SCHEMA_WALLET_AGENT,
)
from environ.agent import batch, send_batch, retrieve_batch
import warnings

warnings.filterwarnings("ignore")

for name in ["prompt", "batch"]:
    os.makedirs(PROCESSED_DATA_PATH / name, exist_ok=True)

lst = []

NUM_OF_OBSERVATIONS = 1000
IMAGE_URL_TEMP = "https://raw.githubusercontent.com/lyc0603/meme/\
refs/heads/main/figures/candle/{ca}.png"

# Wallet Agent Batch Request
batch_name = "wallet_agent"
df_trader = pd.read_csv(PROCESSED_DATA_PATH / "wallet_agent.csv")

with open(
    f"{PROCESSED_DATA_PATH}/prompt/{batch_name}.jsonl", "w", encoding="utf-8"
) as out_f:
    for idx, row in tqdm(df_trader.iterrows(), total=df_trader.shape[0]):
        # Create a dictionary for the current row
        data = {
            "custom_idx": row["wallet_address"] + row["eval_token"],
            "user_msg": PROMPT_WALLET_AGENT.format(
                # 1 decimal place for profit and std
                total_profit=round(row["learn_profit_sum"], 1),
                total_profit_std=round(row["learn_profit_std"], 1),
                # 1 decimal place for avg transaction number and std
                total_avg_transaction_number=round(row["txn_number_mean"], 1),
                total_transaction_number_std=round(row["txn_number_std"], 1),
                total_token_participated=row["learn_token_count"],
            ),
            "model": "gpt-4o-mini",
        }

        req = batch(
            custom_idx=data["custom_idx"],
            user_msg=data["user_msg"],
            system_instruction=SYSTEM_INSTRUCTION_WALLET_AGENT,
            json_schema=JSON_SCHEMA_WALLET_AGENT,
            few_shot_examples=COT_WALLET_AGENT,
            model=data["model"],
        )
        out_f.write(json.dumps(req, ensure_ascii=False) + "\n")

# vars = [
#     "bundle_buy",
#     "bundle_sell",
#     "duration",
#     "#trader",
#     "#txn",
#     "holding_herf",
# ]

# dummies = [
#     "launch_bundle_transfer",
#     "bundle_creator_buy",
# ]

# df = pd.read_csv(f"{PROCESSED_DATA_PATH}/ret_ma.csv")
# df.sort_values("migration_block", ascending=True, inplace=True)


# # Function to assign tertiles (Low, Medium, High)
# def assign_tertile(value: float, past_values: pd.Series) -> str:
#     """Assigns a tertile class based on the value compared to past values."""
#     quantiles = np.nanpercentile(past_values, [33.33, 66.66])
#     if value <= quantiles[0]:
#         return "Low"
#     elif value <= quantiles[1]:
#         return "Medium"
#     else:
#         return "High"


# # Initialize new columns
# for var in vars:
#     df[f"{var}_class"] = np.nan

# # Assign classes based on previous 50 coins
# for i in range(len(df)):
#     if i < 50:
#         continue  # Skip first 50 rows
#     past_df = df.iloc[i - 50 : i]
#     for var in vars:
#         value = df.loc[i, var]
#         past_values = past_df[var].dropna()
#         df.loc[i, f"{var}_class"] = assign_tertile(value, past_values)

# df.dropna(inplace=True)

# # change the dummies to Yes and No
# for dummy in dummies:
#     df[dummy] = df[dummy].apply(lambda x: "Yes" if x else "No")


# # Coin Agent Batch Request
# batch_name = "coin_agent"

# with open(
#     f"{PROCESSED_DATA_PATH}/prompt/{batch_name}.jsonl", "w", encoding="utf-8"
# ) as out_f:
#     for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
#         meme = DataLoader(
#             NewTokenPool(
#                 token0=SOL_TOKEN_ADDRESS,
#                 token1=row["token_address"],
#                 fee=0,
#                 pool_add=row["token_address"],
#                 block_number=0,
#                 chain="raydium",
#                 base_token=row["token_address"],
#                 quote_token=SOL_TOKEN_ADDRESS,
#                 txns={},
#             ),
#         )
#         # Create a dictionary for the current row
#         data = {
#             "custom_idx": row["token_address"],
#             "user_msg": PROMPT_COIN_AGENT.format(
#                 creator_funded_bundle=row["launch_bundle_transfer"],
#                 creator_funded_buy_bundle=row["bundle_creator_buy"],
#                 buy_bundle=row["bundle_buy_class"],
#                 sell_bundle=row["bundle_sell_class"],
#                 pre_migration_duration=row["duration_class"],
#                 number_of_unique_traders=row["#trader_class"],
#                 number_of_transactions=row["#txn_class"],
#                 holding_centralization=row["holding_herf_class"],
#                 comment_history="\n".join(meme.comment_list),
#             ),
#             "image_url": IMAGE_URL_TEMP.format(ca=row["token_address"]),
#             "model": "gpt-4o",
#         }

#         req = batch(
#             custom_idx=row["token_address"],
#             user_msg=data["user_msg"],
#             system_instruction=SYSTEM_INSTRUCTION_COIN_AGENT,
#             json_schema=JSON_SCHEMA_COIN_AGENT,
#             image_url=data["image_url"],
#             few_shot_examples=COT_COIN_AGENT,
#             model=data["model"],
#         )
#         out_f.write(json.dumps(req, ensure_ascii=False) + "\n")

# # Comment Agent Batch Request
# few_shot_example_list = []
# for pool_add, reasoning in FEW_SHOT_EXAMPLES_COMMENT_AGENT.items():
#     meme = DataLoader(
#         NewTokenPool(
#             token0=SOL_TOKEN_ADDRESS,
#             token1=pool_add,
#             fee=0,
#             pool_add=pool_add,
#             block_number=0,
#             chain="raydium",
#             base_token=pool_add,
#             quote_token=SOL_TOKEN_ADDRESS,
#             txns={},
#         ),
#     )
#     few_shot_example_list.extend(
#         [
#             {
#                 "role": "user",
#                 "content": PROMPT_COMMENT_AGENT.format(
#                     comment_history="\n".join(meme.comment_list),
#                 ),
#             },
#             {
#                 "role": "assistant",
#                 "content": reasoning,
#             },
#         ]
#     )

# batch_name = "comment_agent"
# with open(
#     f"{PROCESSED_DATA_PATH}/prompt/{batch_name}.jsonl", "w", encoding="utf-8"
# ) as out_f:
#     for chain in ["raydium"]:
#         for pool in tqdm(
#             import_pool(
#                 chain,
#                 NUM_OF_OBSERVATIONS,
#             )
#         ):
#             meme = DataLoader(
#                 NewTokenPool(
#                     token0=SOL_TOKEN_ADDRESS,
#                     token1=pool["token_address"],
#                     fee=0,
#                     pool_add=pool["token_address"],
#                     block_number=0,
#                     chain=chain,
#                     base_token=pool["token_address"],
#                     quote_token=SOL_TOKEN_ADDRESS,
#                     txns={},
#                 ),
#             )
#             # Comment Agent Batch Request
#             req = batch(
#                 custom_idx=pool["token_address"],
#                 user_msg=PROMPT_COMMENT_AGENT.format(
#                     comment_history="\n".join(meme.comment_list),
#                 ),
#                 system_instruction=SYSTEM_INSTRUCTION_COMMENT_AGENT,
#                 json_schema=JSON_SCHEMA_COIN_AGENT,
#                 few_shot_examples=few_shot_example_list,
#                 model="gpt-4o",
#             )
#             out_f.write(json.dumps(req, ensure_ascii=False) + "\n")

# # Transaction Agent Batch Request
# batch_name = "transaction_agent"
# vars = [
#     "bundle_buy",
#     "bundle_sell",
#     "duration",
#     "#trader",
#     "#txn",
#     "holding_herf",
# ]

# dummies = [
#     "launch_bundle_transfer",
#     "bundle_creator_buy",
# ]

# df = pd.read_csv(f"{PROCESSED_DATA_PATH}/ret_ma.csv")
# df.sort_values("migration_block", ascending=True, inplace=True)


# # Function to assign tertiles (Low, Medium, High)
# def assign_tertile(value: float, past_values: pd.Series) -> str:
#     """Assigns a tertile class based on the value compared to past values."""
#     quantiles = np.nanpercentile(past_values, [33.33, 66.66])
#     if value <= quantiles[0]:
#         return "Low"
#     elif value <= quantiles[1]:
#         return "Medium"
#     else:
#         return "High"


# # Initialize new columns
# for var in vars:
#     df[f"{var}_class"] = np.nan

# # Assign classes based on previous 50 coins
# for i in range(len(df)):
#     if i < 50:
#         continue  # Skip first 50 rows
#     past_df = df.iloc[i - 50 : i]
#     for var in vars:
#         value = df.loc[i, var]
#         past_values = past_df[var].dropna()
#         df.loc[i, f"{var}_class"] = assign_tertile(value, past_values)

# df.dropna(inplace=True)

# # change the dummies to Yes and No
# for dummy in dummies:
#     df[dummy] = df[dummy].apply(lambda x: "Yes" if x else "No")

# with open(
#     f"{PROCESSED_DATA_PATH}/prompt/{batch_name}.jsonl", "w", encoding="utf-8"
# ) as out_f:
#     for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
#         # Create a dictionary for the current row
#         data = {
#             "custom_idx": row["token_address"],
#             "user_msg": PROMPT_TRANSACTION_AGENT.format(
#                 creator_funded_bundle=row["launch_bundle_transfer"],
#                 creator_funded_buy_bundle=row["bundle_creator_buy"],
#                 buy_bundle=row["bundle_buy_class"],
#                 sell_bundle=row["bundle_sell_class"],
#                 pre_migration_duration=row["duration_class"],
#                 number_of_unique_traders=row["#trader_class"],
#                 number_of_transactions=row["#txn_class"],
#                 holding_centralization=row["holding_herf_class"],
#             ),
#             "model": "gpt-4o",
#         }

#         req = batch(
#             custom_idx=row["token_address"],
#             user_msg=data["user_msg"],
#             system_instruction=SYSTEM_INSTRUCTION_TRANSACTION_AGENT,
#             json_schema=JSON_SCHEMA_TRANSACTION_AGENT,
#             few_shot_examples=COT_TRANSACTION_AGENT,
#             model=data["model"],
#         )
#         out_f.write(json.dumps(req, ensure_ascii=False) + "\n")

# # Technical Agent Batch Request
# batch_name = "technical_agent"

# with open(
#     f"{PROCESSED_DATA_PATH}/prompt/{batch_name}.jsonl", "w", encoding="utf-8"
# ) as out_f:
#     for chain in [
#         "raydium",
#     ]:
#         for pool in tqdm(
#             import_pool(
#                 chain,
#                 NUM_OF_OBSERVATIONS,
#             )
#         ):
#             # Technical Agent Batch Request
#             req = batch(
#                 custom_idx=pool["token_address"],
#                 image_url=IMAGE_URL_TEMP.format(ca=pool["token_address"]),
#                 user_msg=PROMPT_TECHNICAL_AGENT,
#                 system_instruction=SYSTEM_INSTRUCTION_TECHNICAL_AGENT,
#                 json_schema=JSON_SCHEMA_COIN_AGENT,
#                 few_shot_examples=COT_TECHNICAL_AGENT,
#                 model="gpt-4o",
#             )
#             out_f.write(json.dumps(req, ensure_ascii=False) + "\n")

batch_id = send_batch(f"{PROCESSED_DATA_PATH}/prompt/{batch_name}.jsonl")
result = retrieve_batch(batch_id)

with open(f"{PROCESSED_DATA_PATH}/batch/{batch_name}.jsonl", "wb") as file:
    file.write(result)
