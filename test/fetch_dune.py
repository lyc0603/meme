"""
Script to fetch dune
"""

import os
from dune_client.client import DuneClient
from dotenv import load_dotenv

load_dotenv()

dune = DuneClient(os.getenv("DUNE_API_KEY"))
query_result = dune.get_latest_result(3830470)
