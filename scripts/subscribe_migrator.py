"""Script to subscribe to the migrator"""

import asyncio
import json
import logging
from time import sleep
from typing import AsyncIterator, Iterator, List, Tuple

from solana.exceptions import SolanaRpcException
from solana.rpc.api import Client
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

RaydiumLPV4 = "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg"
URI = "https://api.mainnet-beta.solana.com"
WSS = "wss://api.mainnet-beta.solana.com"
solana_client = Client(URI)
log_instruction = "initialize2"
seen_signatures = set()

logging.basicConfig(filename="app.log", filemode="a", level=logging.DEBUG)


async def task():
    async for websocket in connect(WSS):
        try:
            # subscription_id = await subscribe_to_logs(
            #     websocket, RpcTransactionLogsFilterMentions(RaydiumLPV4), Finalized
            # )
            await websocket.logs_subscribe(
                filter_=RpcTransactionLogsFilterMentions(RaydiumLPV4),
                commitment=Finalized,
            )
            first_resp = await websocket.recv()
            print(first_resp)
            subscription_id = first_resp[0].result
            print(subscription_id)

        except (ProtocolError, ConnectionClosedError) as err:
            logging.exception(err)
            print(f"Danger! Danger!", err)
            continue
        except KeyboardInterrupt:
            if websocket:
                await websocket.logs_unsubscribe(subscription_id)


if __name__ == "__main__":
    RaydiumLPV4 = Pubkey.from_string(RaydiumLPV4)
    asyncio.run(task())

# async def subscribe_to_logs(
#     websocket: SolanaWsClientProtocol,
#     mentions: RpcTransactionLogsFilterMentions,
#     commitment: Commitment,
# ) -> int:
#     """Subscribe to logs."""
#     await websocket.logs_subscribe(filter_=mentions, commitment=commitment)
#     first_resp = await websocket.recv()
#     return get_subscription_id(first_resp)  # type: ignore


# def get_subscription_id(response: SubscriptionResult) -> int:
#     """Get subscription id."""
#     return response[0].result


# async def process_messages(
#     websocket: SolanaWsClientProtocol, instruction: str
# ) -> AsyncIterator[Signature]:
#     """Async generator, main websocket's loop"""
#     async for idx, msg in enumerate(websocket):
#         value = get_msg_value(msg)
#         if not idx % 100:
#             pass
#             # print(f"{idx=}")
#         for log in value.logs:
#             if instruction not in log:
#                 continue
#             # Start logging
#             logging.info(value.signature)
#             logging.info(log)
#             # Logging to messages.json
#             with open("messages.jsonl", "a", encoding="utf-8") as raw_messages:
#                 json.dump({value.signature: log}, raw_messages)
#             # End logging
#             yield value.signature


# def get_msg_value(msg: List[LogsNotification]) -> RpcLogsResponse:
#     return msg[0].result.value


# def get_instructions(
#     transaction: GetTransactionResp,
# ) -> List[UiPartiallyDecodedInstruction | ParsedInstruction]:
#     instructions = transaction.value.transaction.transaction.message.instructions
#     return instructions


# def instructions_with_program_id(
#     instructions: List[UiPartiallyDecodedInstruction | ParsedInstruction],
#     program_id: str,
# ) -> Iterator[UiPartiallyDecodedInstruction | ParsedInstruction]:
#     return (
#         instruction
#         for instruction in instructions
#         if instruction.program_id == program_id
#     )


# def get_tokens_info(
#     instruction: UiPartiallyDecodedInstruction | ParsedInstruction,
# ) -> Tuple[Pubkey, Pubkey, Pubkey]:
#     accounts = instruction.accounts
#     Pair = accounts[4]
#     Token0 = accounts[8]
#     Token1 = accounts[9]
#     # Start logging
#     logging.info("find LP !!!")
#     logging.info(f"\n Token0: {Token0}, \n Token1: {Token1}, \n Pair: {Pair}")
#     # End logging
#     return (Token0, Token1, Pair)


# def print_table(tokens: Tuple[Pubkey, Pubkey, Pubkey]) -> None:
#     data = [
#         {"Token_Index": "Token0", "Account Public Key": tokens[0]},  # Token0
#         {"Token_Index": "Token1", "Account Public Key": tokens[1]},  # Token1
#         {"Token_Index": "LP Pair", "Account Public Key": tokens[2]},  # LP Pair
#     ]
#     print("============NEW POOL DETECTED====================")
#     header = ["Token_Index", "Account Public Key"]
#     print("│".join(f" {col.ljust(15)} " for col in header))
#     print("|".rjust(18))
#     for row in data:
#         print("│".join(f" {str(row[col]).ljust(15)} " for col in header))


# def get_tokens(signature: Signature, RaydiumLPV4: Pubkey) -> None:
#     """Fubction to get tokens"""

#     transaction = solana_client.get_transaction(
#         signature, encoding="jsonParsed", max_supported_transaction_version=0
#     )
#     with open("transactions.jsonl", "a", encoding="utf-8") as raw_transactions:
#         json.dump({signature: transaction}, raw_transactions)

#     instruction = get_instructions(transaction)
#     filtered_instructions = instructions_with_program_id(instruction, RaydiumLPV4)
#     logging.info(filtered_instructions)
#     for instruction in filtered_instructions:
#         tokens = get_tokens_info(instruction)
#         tokens = get_tokens_info(instruction)
#         print_table(tokens)
#         print(f"True, https://solscan.io/tx/{signature}")


# async def main():
#     """The client as an infinite asynchronous iterator:"""
#     async for websocket in connect(WSS):
#         try:
#             subscription_id = await subscribe_to_logs(
#                 websocket, RpcTransactionLogsFilterMentions(RaydiumLPV4), Finalized
#             )
#             logging.getLogger().setLevel(logging.INFO)
#             async for i, signature in enumerate(
#                 process_messages(websocket, LOG_INSTRUCTION)
#             ):
#                 logging.info(f"{i=}")
#                 try:
#                     get_tokens(signature, RaydiumLPV4)
#                 except SolanaRpcException as err:
#                     logging.exception(err)
#                     logging.info("sleep for 5 seconds and try again")
#                     sleep(5)
#                     continue

#         except (ProtocolError, ConnectionClosedError) as err:
#             # Restart socket connection if ProtocolError: invalid status code
#             logging.exception(err)  # Logging
#             print(f"Danger! Danger!", err)
#             continue
#         except KeyboardInterrupt:
#             if websocket:
#                 await websocket.logs_unsubscribe(subscription_id)


# if __name__ == "__main__":
#     RaydiumLPV4 = Pubkey.from_string(RaydiumLPV4)
#     asyncio.run(main())
