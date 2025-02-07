"""
Script to fetch MEME data from the website
"""

import requests


http = "https://solscan.io/txs"

response = requests.get(http)

# print http

print(response.text)