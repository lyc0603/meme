from solana.rpc.api import Client
from solders.pubkey import Pubkey
from solders.signature import Signature

from solders.signature import Signature

solana_client = Client("https://api.mainnet-beta.solana.com")

pubkey = Pubkey.from_string("8sLbNZoA1cfnvMJLPfp98ZLAnFSYCFApfJKMbiXNLwxj")
solana_client.get_signatures_for_address(pubkey, limit=1).value[0].signature

# get the transaction
solana_client.get_transaction(
    Signature.from_string(
        "5kTYKaMk9dfMhCGSKk85XVTS1iNhBC54N68eUzXfjirDCM1R9wugXiuGhN6zE6URdchfjSUYeFR2kRn8TR7fSMFB"
    ),
    max_supported_transaction_version=0,
)
