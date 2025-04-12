
cred={
    "APP_NAME":"5P53215310",
    "APP_SOURCE":"25362",
    "USER_ID":"znmnClZ6tCY",
    "PASSWORD":"OLl30nLXDt4",
    "USER_KEY":"o1eylfEhyieVKo4VnZrw4DOYFzUg7sUQ",
    "ENCRYPTION_KEY":"mzlxYlRHvLGka7PJ0r5zHgD5fLK2e53S"
    }
client_id=53215310
pin=210794
token='GUZTEMJVGMYTAXZVKBDUWRKZ'

from globals import spot_prices

import json
import os


TICKER_FILE = "tickers.json"

# Build symbol map dynamically
def build_symbol_map():
    if os.path.exists(TICKER_FILE):
        with open(TICKER_FILE, "r") as f:
            tickers = json.load(f)
        mapping = {}
        for t in tickers:
            parts = t.split()
            if len(parts) >= 6:
                strike = parts[5].replace(".00", "")
                opttype = parts[4]
                expiry_month = parts[2][:3].upper()
                expiry_day = parts[1]
                formatted = f"NIFTY{expiry_month}{expiry_day}{strike}{opttype}"
                mapping[t] = formatted
        return mapping
    return {}

# Dynamic symbol map for every call
def get_live_ltp(symbol: str):
    symbol_map = build_symbol_map()
    
    internal = symbol_map.get(symbol, symbol)
    return spot_prices.get(internal, 0.0)






