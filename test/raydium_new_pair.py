"Detect  New Pools Created on Solana Raydium DEX"

# MAnually see transactions of new pairs GThUX1Atko4tqhN2NaiTazWSeFWMuiUvfFnyJyUghFMJ under spl transfer section

import asyncio
import logging
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

# Raydium Liquidity Pool V4
RaydiumLPV4 = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
URI = "https://api.mainnet-beta.solana.com"  # "https://api.devnet.solana.com" | "https://api.mainnet-beta.solana.com"
WSS = "wss://api.mainnet-beta.solana.com"  # "wss://api.devnet.solana.com" | "wss://api.mainnet-beta.solana.com"
solana_client = Client(URI)
# Raydium function call name, look at raydium-amm/program/src/instruction.rs
log_instruction = "initialize2"

seen_signatures = set()


def getTimestamp():
    while True:
        timeStampData = datetime.datetime.now()
        currentTimeStamp = "[" + timeStampData.strftime("%H:%M:%S.%f")[:-3] + "]"
        return currentTimeStamp


class style:
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


async def run(wallet_address: str):
    count = 0
    uri = "wss://api.mainnet-beta.solana.com"
    async with websockets.connect(uri) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "logsSubscribe",
                    "params": [
                        {"mentions": [wallet_address]},
                        {"commitment": "finalized"},
                    ],
                }
            )
        )

        first_resp = await websocket.recv()
        # print(first_resp)
        response_dict = json.loads(first_resp)
        # if 'result' in response_dict:
        #    print("Subscription successful. Subscription ID: ", response_dict['result'])

        async for response in websocket:
            response_dict = json.loads(response)
            if response_dict["params"]["result"]["value"]["err"] == None:
                signature = response_dict["params"]["result"]["value"]["signature"]
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    hash = Signature.from_string(signature)
                    hash_Detail = solana_client.get_transaction(
                        hash, encoding="json", max_supported_transaction_version=0
                    )
                    try:
                        for ui in hash_Detail.value.transaction.meta.pre_token_balances:

                            if ui.owner == Pubkey.from_string(
                                wallet_address
                            ) and ui.mint == Pubkey.from_string(
                                "So11111111111111111111111111111111111111112"
                            ):
                                pre_amount = ui.ui_token_amount.amount
                            if ui.owner == Pubkey.from_string(
                                wallet_address
                            ) and ui.mint != Pubkey.from_string(
                                "So11111111111111111111111111111111111111112"
                            ):
                                token_address = ui.mint
                        for (
                            ui
                        ) in hash_Detail.value.transaction.meta.post_token_balances:
                            if ui.owner == Pubkey.from_string(
                                wallet_address
                            ) and ui.mint == Pubkey.from_string(
                                "So11111111111111111111111111111111111111112"
                            ):
                                post_amount = ui.ui_token_amount.amount
                            if ui.owner == Pubkey.from_string(
                                wallet_address
                            ) and ui.mint != Pubkey.from_string(
                                "So11111111111111111111111111111111111111112"
                            ):
                                token_address = ui.mint
                        if post_amount > pre_amount:
                            print("**************")
                            count += 1
                            print(
                                f"{count}--{getTimestamp()}Account Address: {wallet_address}, {style.YELLOW}[Token SOLD]:{token_address}{style.RESET} , https://solscan.io/tx/{hash}"
                            )
                        else:
                            count += 1

                            print(
                                f"{count}--{getTimestamp()}Account Address: {wallet_address}, {style.GREEN}[Token BOUGHT]:, {token_address}{style.RESET} , https://solscan.io/tx/{hash}"
                            )
                    except Exception as e:
                        print("Error Occured", e, wallet_address, hash)
                        continue


tasks = [run(addr) for addr in wallet_addresses]


async def main():
    await asyncio.gather(*tasks)


asyncio.run(main())
