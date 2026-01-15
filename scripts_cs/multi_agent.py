"""Multi-agent Memecoin Copy Trading."""

import json
import os
import time

import numpy as np
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_CS_PATH
from environ.llm import ChatGPT
from scripts_cs.ml_preprocess import X_test
from scripts_cs.prompt import (
    COT_COIN_AGENT,
    COT_WALLET_AGENT,
    JSON_SCHEMA,
    PROMPT_COIN_AGENT,
    PROMPT_WALLET_AGENT,
    SYSTEM_INSTRUCTION_COIN_AGENT,
    SYSTEM_INSTRUCTION_WALLET_AGENT,
    load_comment,
)

os.makedirs(PROCESSED_DATA_CS_PATH / "multi_agent_res", exist_ok=True)

for idx, row in tqdm(X_test.iterrows(), total=X_test.shape[0]):
    if os.path.exists(
        PROCESSED_DATA_CS_PATH
        / "multi_agent_res"
        / f"{row['token_address']}_{row['trader_address']}.json"
    ):
        continue
    time.sleep(1)
    try:
        # Wallet Agent
        wallet_msg = PROMPT_WALLET_AGENT.format(
            t_stat=round(row["t_stat"], 2),
            average_ret=round(row["average_ret"], 2),
            std_ret=round(row["std_ret"], 2),
            five_to_one_ret=round(row["five_to_one_ret"], 2),
            ten_to_six_ret=round(row["ten_to_six_ret"], 2),
            fifteen_to_eleven_ret=round(row["fifteen_to_eleven_ret"], 2),
            num_trades=int(round(row["num_trades"], 0)),
            time_since_last_trade=int(round(row["time_since_last_trade"], 0)),
            time_since_first_trade=int(round(row["time_since_first_trade"], 0)),
        )

        wallet_agent = ChatGPT(model="gpt-4o")
        wallet_response = wallet_agent(
            message=wallet_msg,
            instruction=SYSTEM_INSTRUCTION_WALLET_AGENT,
            cot=COT_WALLET_AGENT,
            json_schema=JSON_SCHEMA,
            temperature=0,
            logprobs=True,
            top_logprobs=2,
        )

        wallet_res = {
            "res": wallet_response[0],
            "confidence": np.exp(wallet_response[1][3].logprob),
        }

        # Coin Agent
        coin_msg = PROMPT_COIN_AGENT.format(
            launch_bundle="Yes" if row["launch_bundle"] == 1 else "No",
            sniper_bot="Yes" if row["sniper_bot"] == 1 else "No",
            bump_bot="Yes" if row["volume_bot"] == 1 else "No",
            comment_bot="Yes" if row["comment_bot"] == 1 else "No",
            comment_history=load_comment(row["token_address"], row["trader_address"]),
        )
        coin_agent = ChatGPT(model="gpt-4o")
        coin_response = coin_agent(
            message=coin_msg,
            instruction=SYSTEM_INSTRUCTION_COIN_AGENT,
            cot=COT_COIN_AGENT,
            json_schema=JSON_SCHEMA,
            temperature=0,
            logprobs=True,
            top_logprobs=2,
        )
        coin_res = {
            "res": coin_response[0],
            "confidence": np.exp(coin_response[1][3].logprob),
        }

        # save results
        with open(
            PROCESSED_DATA_CS_PATH
            / "multi_agent_res"
            / f"{row['token_address']}_{row['trader_address']}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "wallet_agent": wallet_res,
                    "coin_agent": coin_res,
                    "label": row["label_cls"],
                },
                f,
                ensure_ascii=False,
                indent=4,
            )
    except Exception as e:
        print(f"Error processing {row['token_address']}_{row['trader_address']}: {e}")
