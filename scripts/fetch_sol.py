"""Script to fetch the solana data"""

from solana.rpc.api import Client
from solders.signature import Signature

# solana_client = Client("https://api.mainnet-beta.solana.com")

# txns = solana_client.get_signatures_for_address(
#     Pubkey.from_string("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"), limit=1000
# )

# txns_info = solana_client.get_transaction(
#     [
#         Signature.from_string(
#             "4zJoq3fazeDfBapW4T17SqBWq6W5SCwMJbbYsUciSrQd74MBvxJbkhiQjwLx6vv2gy8wMwzGjvTDkAcy2yCwTTW2",
#         ),
#         Signature.from_string(
#             "5xUzmpQ7RhgiET8FMR2Jjs7nLendvv7XQ9ZAXUks8AsBuT1ScbzrSmVaXogXeu3Ry4setu3AraH6xAY95KuVmGdd",
#         ),
#     ],
#     encoding="jsonParsed",
#     max_supported_transaction_version=0,
# )

import asyncio
from solana.rpc.providers.async_http import AsyncHTTPProvider
from solders.rpc.requests import GetTransaction
from solders.rpc.responses import (
    GetTransactionResp,
)
from solders.transaction_status import UiTransactionEncoding
from solders.rpc.config import RpcTransactionConfig


async def main():
    provider = AsyncHTTPProvider(
        "https://solana-mainnet.core.chainstack.com/a0db22a6450d2ad8bfabb1b8254b7abb"
    )
    reqs = (
        GetTransaction(
            Signature.from_string(
                "4zJoq3fazeDfBapW4T17SqBWq6W5SCwMJbbYsUciSrQd74MBvxJbkhiQjwLx6vv2gy8wMwzGjvTDkAcy2yCwTTW2",
            ),
            config=RpcTransactionConfig(
                encoding=UiTransactionEncoding.JsonParsed,
                max_supported_transaction_version=0,
            ),
            id=0,
        ),
        GetTransaction(
            Signature.from_string(
                "5xUzmpQ7RhgiET8FMR2Jjs7nLendvv7XQ9ZAXUks8AsBuT1ScbzrSmVaXogXeu3Ry4setu3AraH6xAY95KuVmGdd",
            ),
            config=RpcTransactionConfig(
                encoding=UiTransactionEncoding.JsonParsed,
                max_supported_transaction_version=0,
            ),
            id=1,
        ),
    )
    parsers = (GetTransactionResp, GetTransactionResp)
    result = await provider.make_batch_request(reqs, parsers)
    print(result[0])
    print(result[1])


if __name__ == "__main__":
    asyncio.run(main())
