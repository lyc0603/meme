"""
Class for the LLM model
"""

import os
import time
from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from environ.settings import OPENAI_API_KEY

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
    few_shot_examples: Optional[list] = None,
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
            "logprobs": True,
            "top_logprobs": 3,
        },
    }


class ChatGPT:
    """
    ChatGPT class to interact with the OpenAI API
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = OPENAI_API_KEY,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _build_prompt(
        self,
        message: str,
        instruction: str | None = None,
        cot: str | None = None,
    ) -> list[dict[str, str]]:
        """
        Function to build the prompt
        """

        prompt = [{"role": "user", "content": message}]

        if cot:
            prompt = cot + prompt

        if instruction:
            prompt = [{"role": "system", "content": instruction}] + prompt

        return prompt

    @retry(
        stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def __call__(
        self,
        message: str,
        instruction: str | None = None,
        cot: str | None = None,
        json_schema: dict | None = None,
        temperature: float = 0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
    ) -> str | tuple[str, dict[str, float]]:
        time.sleep(2)

        params = {
            "model": self.model,
            "messages": self._build_prompt(message, instruction, cot),
            "temperature": temperature,
        }

        if json_schema:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }

        if logprobs:
            params["logprobs"] = logprobs
            params["top_logprobs"] = top_logprobs

        response = self.client.chat.completions.create(**params).choices[0]
        if logprobs:
            return (
                response.message.content,
                response.logprobs.content,
            )
        return response.message.content

    def send_batch(
        self,
        batch_path: str,
    ) -> str:
        """Function to send a batch request to the GPT-4o API."""
        batch_input_file = self.client.files.create(
            file=open(batch_path, "rb"),
            purpose="batch",
        )
        batch_input_file_id = batch_input_file.id

        batch = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return batch.id

    def retrieve_batch(self, batch_id: str) -> dict:
        """Function to retrieve the status of a batch request."""
        while True:
            current_batch = self.client.batches.retrieve(batch_id)
            status = current_batch.status
            print(f"Batch status: {status}")
            if status in ("completed", "failed", "cancelled", "expired"):
                break
            time.sleep(10)  # Wait before polling again

        if status != "completed":
            raise RuntimeError(f"Batch ended with status: {status}")

        # Download output file
        output_file_id = current_batch.output_file_id

        return self.client.files.content(output_file_id).content


if __name__ == "__main__":
    chat_gpt = ChatGPT()
    response = chat_gpt(
        "Is Shanghai the capital of China?",
        logprobs=True,
    )
    print(response)
