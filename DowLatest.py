"""
Dow Jones Data Fetcher and Exporter Script

Description:
    This script fetches the latest historical data for the Dow Jones Industrial Average 
    from Yahoo Finance using the yfinance library and exports it to an Excel file.

Usage:
    python DowLatest.py

Created by: Raymond Beecham
Date: October 10, 2023
Version: 1.0
Contact: rb102@my.tamuct.edu

Dependencies:
    - yfinance
    - openpyxl
"""

import yfinance as yf

# Fetch the Dow Jones Industrial Average data
ticker_symbol = "^DJI"
dow_jones = yf.Ticker(ticker_symbol)

# Get the latest historical daily data (you can adjust the period as needed)
latest_data = dow_jones.history(period="max")

print(latest_data)

# Fetch other relevant information
info = dow_jones.info
print("\nDow Jones Information:")
print(f"Name: {info['shortName']}")
print(f"Previous Close: {info['previousClose']}")
print(f"Open: {info['open']}")
print(f"Volume: {info['volume']}")

# Check if 'marketCap' is in the info dictionary before accessing it
if 'marketCap' in info:
    print(f"Market Cap: {info['marketCap']}")
else:
    print("Market Cap: N/A")

print(f"52 Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}")
print(f"52 Week Low: {info.get('fiftyTwoWeekLow', 'N/A')}")

# Convert timezone-aware datetime columns to timezone-naive
latest_data.index = latest_data.index.tz_localize(None)

# Export to Excel
latest_data.to_excel("dow_jones_data.xlsx", engine='openpyxl')
