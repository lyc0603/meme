"""Script to monitor Raydium Newly Created Pairs"""

import asyncio
import logging
import os
from time import sleep
from typing import AsyncIterator, Iterator, List, Tuple

from asyncstdlib import enumerate
from solana.exceptions import SolanaRpcException
from solana.rpc.api import Client

# Type hinting imports
from solana.rpc.commitment import Commitment, Finalized
from solana.rpc.websocket_api import SolanaWsClientProtocol, connect
from solders.pubkey import Pubkey
from solders.rpc.config import RpcTransactionLogsFilterMentions
from solders.rpc.responses import (
    GetTransactionResp,
    LogsNotification,
    RpcLogsResponse,
    SubscriptionResult,
)
from solders.signature import Signature
from solders.transaction_status import ParsedInstruction, UiPartiallyDecodedInstruction
from websockets.exceptions import ConnectionClosedError, ProtocolError

from environ.constants import DATA_PATH

# Raydium Liquidity Pool V4
RAYDIUM_LP = Pubkey.from_string("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8")
URI = "https://api.mainnet-beta.solana.com"
WSS = "wss://api.mainnet-beta.solana.com"
solana_client = Client(URI)
LOG_INSTRUCTION = "initialize2"
seen_signatures = set()
logging.basicConfig(filename="app.log", filemode="a", level=logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)
os.makedirs(DATA_PATH / "solana" / "meme", exist_ok=True)


async def main():
    """The client as an infinite asynchronous iterator:"""
    async for websocket in connect(WSS):
        try:
            subscription_id = await subscribe_to_logs(
                websocket, RpcTransactionLogsFilterMentions(RAYDIUM_LP), Finalized
            )
            async for _, signature in enumerate(
                process_messages(websocket, LOG_INSTRUCTION)
            ):
                try:
                    get_tokens(signature, RAYDIUM_LP)
                except SolanaRpcException as err:
                    logging.exception(err)
                    logging.info("sleep for 5 seconds and try again")
                    sleep(5)
                    continue
        except (ProtocolError, ConnectionClosedError) as err:
            logging.exception(err)
            continue
        except KeyboardInterrupt:
            if websocket:
                await websocket.logs_unsubscribe(subscription_id)


async def subscribe_to_logs(
    websocket: SolanaWsClientProtocol,
    mentions: RpcTransactionLogsFilterMentions,
    commitment: Commitment,
) -> int:
    """Subscribe to logs

    Args:
        websocket (SolanaWsClientProtocol): websocket connection
        mentions (RpcTransactionLogsFilterMentions): filter for logs
        commitment (Commitment): commitment level"""
    await websocket.logs_subscribe(filter_=mentions, commitment=commitment)
    first_resp = await websocket.recv()
    return get_subscription_id(first_resp)


def get_subscription_id(response: SubscriptionResult) -> int:
    """Get subscription id from response

    Args:
        response (SubscriptionResult): response from websocket"""
    return response[0].result


async def process_messages(
    websocket: SolanaWsClientProtocol, instruction: str
) -> AsyncIterator[Signature]:
    """Async generator, main websocket's loop

    Args:
        websocket (SolanaWsClientProtocol): websocket connection
        instruction (str): instruction to filter"""
    async for idx, msg in enumerate(websocket):
        value = get_msg_value(msg)
        if not idx % 100:
            pass
        for log in value.logs:
            if instruction not in log:
                continue
            yield value.signature


def get_msg_value(msg: List[LogsNotification]) -> RpcLogsResponse:
    """Get message value from logs

    Args:
        msg (List[LogsNotification]): logs notification"""
    return msg[0].result.value


def get_tokens(signature: Signature, RAYDIUM_LP: Pubkey) -> None:
    """Get tokens from instruction

    Args:
        signature (Signature): transaction signature
        RAYDIUM_LP (Pubkey): Raydium LP V4 program id"""
    transaction = solana_client.get_transaction(
        signature, encoding="jsonParsed", max_supported_transaction_version=0
    )
    instructions = get_instructions(transaction)
    filtred_instuctions = instructions_with_program_id(instructions, RAYDIUM_LP)
    logging.info(filtred_instuctions)
    for instruction in filtred_instuctions:
        tokens = get_tokens_info(instruction)
        with open(
            f"{DATA_PATH}/solana/meme/{tokens[0]}.jsonl", "a", encoding="utf-8"
        ) as raw_tokens:
            pass


def get_instructions(
    transaction: GetTransactionResp,
) -> List[UiPartiallyDecodedInstruction | ParsedInstruction]:
    """Get instructions from transaction

    Args:
        transaction (GetTransactionResp): transaction response"""
    instructions = transaction.value.transaction.transaction.message.instructions
    return instructions


def instructions_with_program_id(
    instructions: List[UiPartiallyDecodedInstruction | ParsedInstruction],
    program_id: str,
) -> Iterator[UiPartiallyDecodedInstruction | ParsedInstruction]:
    """Filter instructions by program id

    Args:
        instructions (List[UiPartiallyDecodedInstruction | ParsedInstruction]): instructions
        program_id (str): program id"""
    return (
        instruction
        for instruction in instructions
        if instruction.program_id == program_id
    )


def get_tokens_info(
    instruction: UiPartiallyDecodedInstruction | ParsedInstruction,
) -> Tuple[Pubkey, Pubkey, Pubkey]:
    """Get tokens info from instruction

    Args:
        instruction (UiPartiallyDecodedInstruction | ParsedInstruction): instruction"""
    accounts = instruction.accounts
    pair = accounts[4]
    token_0 = accounts[8]
    token_1 = accounts[9]
    logging.info("\n Token0: %s, \n Token1: %s, \n Pair: %s", token_0, token_1, pair)
    return (pair, token_0, token_1)


if __name__ == "__main__":
    asyncio.run(main())
