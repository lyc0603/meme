"""ChatGPT Batch Processing Script"""

import glob
import json
import os
from openai import OpenAI
from tqdm import tqdm
from collections import defaultdict

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH
from environ.prompt import (
    SYSTEM_INSTRUCTION_COMMENT_BOT,
    FEW_SHOT_EXAMPLES_COMMENT_BOT,
    JSON_SCHEMA_COMMENT_BOT,
)
from environ.agent import send_batch, retrieve_batch

client = OpenAI(api_key=os.getenv("OPENAI_API"))

BATCH_NAME = "pre_trump_pumpfun"

# Load comments from JSONL files
comments_path = glob.glob(f"{DATA_PATH}/solana/{BATCH_NAME}/reply/*.jsonl")

comments_batch = []
comments_chunk = {}
comments = {}
counter = 0
for file_path in comments_path:
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            counter += 1
            comment = json.loads(line.strip())
            comments[str(counter)] = {
                "id": comment["id"],
                "comment": comment,
                "token_add": file_path.split("/")[-1].split(".")[0],
            }
            comments_chunk[str(counter)] = {
                "id": comment["id"],
                "comment": comment,
                "token_add": file_path.split("/")[-1].split(".")[0],
            }

            if counter % 25_000 == 0:
                comments_batch.append(comments_chunk)
                comments_chunk = {}

comments_batch.append(comments_chunk)


def batch_few_shot_learning(
    custom_idx: str,
    user_msg: str,
    system_instruction: str,
    few_shot_example: str,
    json_schema: str,
    model: str = "gpt-4o-mini",
) -> dict:
    """Function to implement the few-shot learning"""
    request_payload = {
        "custom_id": custom_idx,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system_instruction},
                *few_shot_example,
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 1000,
            "response_format": {
                "type": "json_schema",
                "json_schema": json_schema,
            },
            "temperature": 0,
        },
    }
    return request_payload


result_batch = []

for comments in comments_batch:
    batch_input_file_path = (
        f"{PROCESSED_DATA_PATH}/{BATCH_NAME}_comments_sentiment_batch.jsonl"
    )

    with open(batch_input_file_path, "w", encoding="utf-8") as out_f:
        for idx, comment in tqdm(
            comments.items(),
            desc="Preparing batch inputs",
            total=len(comments),
            leave=False,
        ):
            req = batch_few_shot_learning(
                custom_idx=idx,
                user_msg=comment["comment"]["text"],
                system_instruction=SYSTEM_INSTRUCTION_COMMENT_BOT,
                few_shot_example=FEW_SHOT_EXAMPLES_COMMENT_BOT,
                json_schema=JSON_SCHEMA_COMMENT_BOT,
                model="gpt-4o-mini",
            )
            out_f.write(json.dumps(req, ensure_ascii=False) + "\n")

    batch_id = send_batch(batch_input_file_path)
    result = retrieve_batch(batch_id)

    result_batch.append(result)

with open(
    f"{PROCESSED_DATA_PATH}/{BATCH_NAME}_comments_sentiment_batch_res.jsonl",
    "w",
    encoding="utf-8",
) as file:
    for res in result_batch:
        file.write(json.dumps(res, ensure_ascii=False) + "\n")

with open(
    f"{PROCESSED_DATA_PATH}/{BATCH_NAME}_comments_sentiment_batch_res.jsonl",
    "r",
    encoding="utf-8",
) as res_f:
    lines = res_f.readlines()
    for line in lines:
        line = json.loads(line.strip())
        custom_id = line["custom_id"]
        response_content = json.loads(
            line["response"]["body"]["choices"][0]["message"]["content"]
        )
        comments[custom_id]["bot"] = response_content["bot"]
        comments[custom_id]["sentiment"] = response_content["sentiment"]

token_dict = defaultdict(list)
for comment in comments.values():
    token_dict[comment["token_add"]].append(comment)

os.makedirs(PROCESSED_DATA_PATH / "comment" / BATCH_NAME, exist_ok=True)
for token, comment_list in token_dict.items():
    with open(
        PROCESSED_DATA_PATH / "comment" / BATCH_NAME / f"{token}.jsonl",
        "a",
        encoding="utf-8",
    ) as f:
        for comment in comment_list:
            f.write(json.dumps(comment, ensure_ascii=False) + "\n")
