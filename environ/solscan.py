"""
Class for Solscan API
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time


if __name__ == "__main__":

    # Provide path to the chromedriver (if needed)
    service = Service("/Users/yichenluo/Desktop/Research/meme/environ/chromedriver")  # Update with your WebDriver path if necessary

    # Initialize the WebDriver
    driver = webdriver.Chrome(service=service)

    # Open the webpage
    url = "https://gmgn.ai/trade/s6Slhpa5?chain=sol"
    driver.get(url)

    # allow user to login
    time.sleep(120)

    # Get the html of the page
    html = driver.page_source
    print(html)