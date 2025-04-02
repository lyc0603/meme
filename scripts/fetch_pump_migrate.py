"""Script to fetch the Pump.fun migration data."""

import time

from solana.rpc.api import Client
from solders.pubkey import Pubkey
from solders.signature import Signature
from tqdm import tqdm

from environ.sol_fetcher import (
    get_txn,
    get_txns_for_address,
    parse_inner_instruction,
    parse_instruction,
    parse_log_messages,
    parse_signer,
)

PUMPFUN_RAYDIUM_MIGRATOR_ADDRESS = "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg"
PUMPFUN_RAYDIUM_MIGRATOR_PUBKEY = Pubkey.from_string(PUMPFUN_RAYDIUM_MIGRATOR_ADDRESS)
TXN_SINCE_TRUMP = "3UjLEr7A3tMWumLJ3QxkMBmpDnToi7oLF5zUGM87Xhp1oYGNgFBWH3HUZxo6bS8Jzht7sEHZ1X9QCkFjqspqZTRK"


txns = get_txns_for_address(PUMPFUN_RAYDIUM_MIGRATOR_ADDRESS, limit=10)

for txn in tqdm(txns.value):
    time.sleep(0.2)
    transaction = get_txn(txn.signature)
    signer = parse_signer(transaction)

    if signer != PUMPFUN_RAYDIUM_MIGRATOR_PUBKEY:
        continue

    instruction = parse_instruction(transaction)
    inner_instruction = parse_inner_instruction(transaction)
    log_msgs = parse_log_messages(transaction)

    # initialize2 as a key identifier for pool creation
    initialize2 = False
    for log_msg in log_msgs:
        if "initialize2" in log_msg:
            initialize2 = True
            break

    if not initialize2:
        continue

    print(f"Graduation: {txn.signature}")
