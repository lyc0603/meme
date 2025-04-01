"""
Script to monitor Raydium transactions for specific base tokens.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from solana.rpc.websocket_api import connect
from solders.pubkey import Pubkey
from solders.rpc.config import RpcTransactionLogsFilterMentions
from solders.rpc.responses import LogsNotification, RpcLogsResponse, SubscriptionResult
from websockets.exceptions import ConnectionClosedError, ProtocolError
from watchfiles import awatch, Change

from environ.constants import DATA_PATH

WATCH_DIR = Path(f"{DATA_PATH}/solana/meme")
WSS = "wss://api.mainnet-beta.solana.com"

logging.basicConfig(
    filename="logs/raydium_txn.log",
    filemode="a",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

task_queue = asyncio.Queue()


def get_subscription_id(response: SubscriptionResult) -> int:
    """Get the subscription ID from the response."""
    return response[0].result


def get_msg_value(msg: list[LogsNotification]) -> RpcLogsResponse:
    """Get the message value from the notification."""
    return msg[0].result.value


async def monitor_base_token(pubkey: Pubkey, file_path: Path, hours: int):
    """Monitor transactions for a specific base token."""
    logging.info("Monitoring base token: %s", pubkey)
    end_time = datetime.now() + timedelta(hours=hours)

    async for websocket in connect(WSS):
        try:
            subscription_id = await websocket.logs_subscribe(
                filter_=RpcTransactionLogsFilterMentions(pubkey), commitment="finalized"
            )
            await websocket.recv()

            async for msg in websocket:
                try:
                    if datetime.now() > end_time:
                        await websocket.logs_unsubscribe(subscription_id)
                        break

                    value = get_msg_value(msg)
                    logs_data = {
                        "signature": str(value.signature),
                        "logs": value.logs,
                        "timestamp": datetime.now().isoformat(),
                    }
                    with open(file_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(logs_data) + "\n")

                except Exception as e:
                    logging.error("Error processing message: %s", e)

        except (ConnectionClosedError, ProtocolError) as e:
            logging.error("WebSocket error: %s", e)
            await asyncio.sleep(5)

        except asyncio.CancelledError:
            logging.info("Monitoring cancelled.")
            break

        except Exception as e:
            logging.error("Unexpected error: %s", e)
            await asyncio.sleep(5)


async def async_file_watcher():
    """Asynchronously watch for new .jsonl files and queue tasks."""
    async for changes in awatch(WATCH_DIR):
        for change, path in changes:
            if change == Change.added and Path(path).suffix == ".jsonl":
                file_path = Path(path)
                try:
                    base_token_str = file_path.stem.split("_")[1]
                    base_token_pubkey = Pubkey.from_string(base_token_str)
                    await task_queue.put((base_token_pubkey, file_path))
                except Exception as e:
                    logging.error("Error parsing %s: %s", path, e)


async def task_dispatcher():
    """Dispatch token monitoring tasks from queue."""
    while True:
        pubkey, file_path = await task_queue.get()
        asyncio.create_task(monitor_base_token(pubkey, file_path, hours=12))


async def main():
    """Run the async file watcher and dispatcher concurrently."""
    await asyncio.gather(
        async_file_watcher(),
        task_dispatcher(),
    )


if __name__ == "__main__":
    asyncio.run(main())
