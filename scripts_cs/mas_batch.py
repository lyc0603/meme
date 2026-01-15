"""Script to run multi-agent system (MAS) model in batch."""

import datetime
import json
import os

from tqdm import tqdm

from environ.constants import PROCESSED_DATA_CS_PATH
from environ.llm import batch, retrieve_batch, send_batch
from scripts_cs.ml_preprocess import X_test
from scripts_cs.prompt import (
    COT_COIN_AGENT,
    COT_WALLET_AGENT,
    IMAGE_URL_TEMP,
    JSON_SCHEMA,
    JSON_SCHEMA_COIN_AGENT,
    JSON_SCHEMA_WALLET_AGENT,
    PROMPT_COIN_AGENT,
    PROMPT_WALLET_AGENT,
    SYSTEM_INSTRUCTION_COIN_AGENT,
    SYSTEM_INSTRUCTION_WALLET_AGENT,
    load_comment,
)

SUFFIX = datetime.datetime.now().strftime("%Y%m%d")

for dir_name in ["batch", "batch_res"]:
    os.makedirs(PROCESSED_DATA_CS_PATH / dir_name, exist_ok=True)

# # Few-Shot CoT Multi-agent System
# with open(
#     PROCESSED_DATA_CS_PATH / "batch" / f"mas_{SUFFIX}.jsonl", "w", encoding="utf-8"
# ) as f:
#     for idx, row in tqdm(X_test.iterrows(), total=X_test.shape[0]):
#         # Wallet Agent
#         data_wallet = {
#             "custom_idx": f"wallet_{row['token_address']}_{row['trader_address']}",
#             "user_msg": PROMPT_WALLET_AGENT.format(
#                 t_stat=round(row["t_stat"], 2),
#                 average_ret=round(row["average_ret"], 2),
#                 std_ret=round(row["std_ret"], 2),
#                 five_to_one_ret=round(row["five_to_one_ret"], 2),
#                 ten_to_six_ret=round(row["ten_to_six_ret"], 2),
#                 fifteen_to_eleven_ret=round(row["fifteen_to_eleven_ret"], 2),
#                 num_trades=int(round(row["num_trades"], 0)),
#                 time_since_last_trade=int(round(row["time_since_last_trade"], 0)),
#                 time_since_first_trade=int(round(row["time_since_first_trade"], 0)),
#             ),
#             "model": "gpt-4o",
#         }
#         req_wallet = batch(
#             custom_idx=data_wallet["custom_idx"],
#             user_msg=data_wallet["user_msg"],
#             system_instruction=SYSTEM_INSTRUCTION_WALLET_AGENT,
#             json_schema=JSON_SCHEMA_WALLET_AGENT,
#             few_shot_examples=COT_WALLET_AGENT,
#             model=data_wallet["model"],
#         )
#         f.write(json.dumps(req_wallet) + "\n")

#         # Coin Agent
#         data_coin = {
#             "custom_idx": f"coin_{row['token_address']}_{row['trader_address']}",
#             "user_msg": PROMPT_COIN_AGENT.format(
#                 launch_bundle="Yes" if row["launch_bundle"] == 1 else "No",
#                 sniper_bot="Yes" if row["sniper_bot"] == 1 else "No",
#                 bump_bot="Yes" if row["volume_bot"] == 1 else "No",
#                 comment_bot="Yes" if row["comment_bot"] == 1 else "No",
#                 comment_history=load_comment(
#                     row["token_address"], row["trader_address"]
#                 ),
#             ),
#             "image_url": IMAGE_URL_TEMP.format(
#                 ca=f"{row["token_address"]}_{row["trader_address"]}"
#             ),
#             "model": "gpt-4o",
#         }
#         req_coin = batch(
#             custom_idx=data_coin["custom_idx"],
#             user_msg=data_coin["user_msg"],
#             system_instruction=SYSTEM_INSTRUCTION_COIN_AGENT,
#             json_schema=JSON_SCHEMA_COIN_AGENT,
#             image_url=data_coin["image_url"],
#             few_shot_examples=COT_COIN_AGENT,
#             model=data_coin["model"],
#         )
#         f.write(json.dumps(req_coin) + "\n")

# batch_id = send_batch(f"{PROCESSED_DATA_CS_PATH}/batch/mas_{SUFFIX}.jsonl")
# result = retrieve_batch(batch_id)

# with open(f"{PROCESSED_DATA_CS_PATH}/batch_res/mas_{SUFFIX}.jsonl", "wb") as file:
#     file.write(result)


# Zero-Shot Multi-agent System
for dir_name in ["batch", "batch_res"]:
    os.makedirs(PROCESSED_DATA_CS_PATH / dir_name, exist_ok=True)

with open(
    PROCESSED_DATA_CS_PATH / "batch" / f"mas_zero_shot_{SUFFIX}.jsonl",
    "w",
    encoding="utf-8",
) as f:
    for idx, row in tqdm(X_test.iterrows(), total=X_test.shape[0]):
        # Wallet Agent
        data_wallet = {
            "custom_idx": f"wallet_{row['token_address']}_{row['trader_address']}",
            "user_msg": PROMPT_WALLET_AGENT.format(
                t_stat=round(row["t_stat"], 2),
                average_ret=round(row["average_ret"], 2),
                std_ret=round(row["std_ret"], 2),
                five_to_one_ret=round(row["five_to_one_ret"], 2),
                ten_to_six_ret=round(row["ten_to_six_ret"], 2),
                fifteen_to_eleven_ret=round(row["fifteen_to_eleven_ret"], 2),
                num_trades=int(round(row["num_trades"], 0)),
                time_since_last_trade=int(round(row["time_since_last_trade"], 0)),
                time_since_first_trade=int(round(row["time_since_first_trade"], 0)),
            ),
            "model": "gpt-4o",
        }
        req_wallet = batch(
            custom_idx=data_wallet["custom_idx"],
            user_msg=data_wallet["user_msg"],
            system_instruction=SYSTEM_INSTRUCTION_WALLET_AGENT,
            json_schema=JSON_SCHEMA_WALLET_AGENT,
            model=data_wallet["model"],
        )
        f.write(json.dumps(req_wallet) + "\n")

        # Coin Agent
        data_coin = {
            "custom_idx": f"coin_{row['token_address']}_{row['trader_address']}",
            "user_msg": PROMPT_COIN_AGENT.format(
                launch_bundle="Yes" if row["launch_bundle"] == 1 else "No",
                sniper_bot="Yes" if row["sniper_bot"] == 1 else "No",
                bump_bot="Yes" if row["volume_bot"] == 1 else "No",
                comment_bot="Yes" if row["comment_bot"] == 1 else "No",
                comment_history=load_comment(
                    row["token_address"], row["trader_address"]
                ),
            ),
            "image_url": IMAGE_URL_TEMP.format(
                ca=f"{row["token_address"]}_{row["trader_address"]}"
            ),
            "model": "gpt-4o",
        }
        req_coin = batch(
            custom_idx=data_coin["custom_idx"],
            user_msg=data_coin["user_msg"],
            system_instruction=SYSTEM_INSTRUCTION_COIN_AGENT,
            json_schema=JSON_SCHEMA_COIN_AGENT,
            image_url=data_coin["image_url"],
            model=data_coin["model"],
        )
        f.write(json.dumps(req_coin) + "\n")

batch_id = send_batch(f"{PROCESSED_DATA_CS_PATH}/batch/mas_zero_shot_{SUFFIX}.jsonl")
result = retrieve_batch(batch_id)

with open(
    f"{PROCESSED_DATA_CS_PATH}/batch_res/mas_zero_shot_{SUFFIX}.jsonl", "wb"
) as file:
    file.write(result)
