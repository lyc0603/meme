"""Environ for agents"""

import os
import time
from typing import Optional

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API"))


def send_batch(
    batch_path: str,
) -> str:
    """Function to send a batch request to the GPT-4o API."""
    batch_input_file = client.files.create(
        file=open(batch_path, "rb"),
        purpose="batch",
    )
    batch_input_file_id = batch_input_file.id

    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "meme comment bot detection"},
    )
    return batch.id


def retrieve_batch(batch_id: str) -> dict:
    """Function to retrieve the status of a batch request."""
    while True:
        current_batch = client.batches.retrieve(batch_id)
        status = current_batch.status
        print(f"Batch status: {status}")
        if status in ("completed", "failed", "cancelled", "expired"):
            break
        time.sleep(10)  # Wait before polling again

    if status != "completed":
        raise RuntimeError(f"Batch ended with status: {status}")

    # Download output file
    output_file_id = current_batch.output_file_id

    return client.files.content(output_file_id).content


def batch(
    custom_idx: str,
    user_msg: str,
    system_instruction: str,
    json_schema: dict,
    image_url: Optional[str] = None,
    few_shot_examples: Optional[dict] = None,
    model: str = "gpt-4o",
) -> dict:
    """Function to construct a valid GPT-4o batch request with image and schema."""

    # system instruction
    messages = [
        {"role": "system", "content": system_instruction},
    ]

    # few shot examples
    if few_shot_examples:
        messages = messages + few_shot_examples

    # user message with image URL if provided
    if image_url:
        messages += [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_msg},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "high"},
                    },
                ],
            },
        ]
    else:
        # user message without image URL
        messages += [
            {"role": "user", "content": user_msg},
        ]

    return {
        "custom_id": custom_idx,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": json_schema,
            },
            "temperature": 0,
        },
    }
