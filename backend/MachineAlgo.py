import ssl
import certifi
from py5paisa import FivePaisaClient
from py5paisa.order import Order, OrderType, Exchange
import pyotp
import os
import mibian as mb
working_dir = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
import datetime as dt
import auth
import pandas_ta as ta
import numpy as np
import requests
import Telegram_token
import math
from openpyxl import load_workbook
import json
import threading
import pytz
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
UTC = pytz.timezone('Asia/Kolkata')
import time
from pymongo import MongoClient
import sklearn
import requests


ssl_context = ssl.create_default_context(cafile=certifi.where())

print("scikit-learn version:", sklearn.__version__)
uri = "mongodb+srv://singhrajeev1470_db_user:kaPh8sxuaVFWWsSr@cluster0.mtmtbrr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0&tls=true"

# Connect to MongoDB
client1 = MongoClient(uri, tls=True, tlsCAFile=certifi.where())

# Choose a database (it will be created if not exist)
db = client1['AlgoTradingDB']
print(db.list_collection_names())
ce_collection = db["CE_Options"]
pe_collection = db["PE_Options"]

client = FivePaisaClient(cred=auth.cred)
print(pyotp.TOTP(auth.token).now())

# New TOTP based authentication
client.get_totp_session(auth.client_id, pyotp.TOTP(auth.token).now(), auth.pin)

# User Inputs
START_TIME = [9, 16, 0]  # Algo Start Time
EXIT_TIME = [23, 30, 0]  # Algo End Time

Total_Cash = 10000
Max_Position = 1
Total_Cash_per_position = int(Total_Cash / Max_Position)

Take_Profit = 20

#Tickers = ['NIFTY 08 MAY 2025 CE 24400.00','NIFTY 08 MAY 2025 PE 24400.00']


import json
import os

ticker_path = os.path.join(os.path.dirname(__file__), "tickers.json")
print("🔍 Looking for:", ticker_path)

if os.path.exists(ticker_path):
    print("📂 Loading tickers.json...")
    with open(ticker_path, "r") as f:
        Tickers = json.load(f)


# Getting Instrument
instrument_df = pd.read_csv('ScripMasterfno.csv')
instrument_df = instrument_df[(instrument_df.Exch == 'N')]
#print(instrument_df[(instrument_df.Name == 'NIFTY 03 APR 2025 CE 23300.00')])








signal_data = []

# Getting Script Code
def scripcode_lookup(instrument=instrument_df, symbol='TCS'):
    ## This function is used to find the instrument token number
    try:
        return instrument[instrument.Name == symbol].ScripCode.values[0]
    except:
        return -1



def get_cash_market_data_3(timeframe='3m'):
    df = pd.DataFrame(client.historical_data(Exch='N', ExchangeSegment='C', ScripCode=999920000, time=timeframe,
                                             From=dt.date.today() - dt.timedelta(5), To=dt.date.today()))

    print(df.columns)
    df.set_index("Datetime", inplace=True)
    print(df)
    return df



def get_cash_market_data(symbol, timeframe):
    scriptcode = scripcode_lookup(instrument_df, symbol)
    sym=symbol

    parts = symbol.split()
    ticker = parts[0]  # Extract ticker
    expiry = f"{parts[1]} {parts[2]} {parts[3]}"  # Extract expiry date
    opttype = parts[4]  # Extract option type (CE/PE)
    strike = float(parts[5])  # Extract strike price and convert to float

    df = pd.DataFrame(client.historical_data(Exch='N', ExchangeSegment='D', ScripCode=scriptcode, time=timeframe,
                                             From=dt.date.today()-dt.timedelta(3), To=dt.date.today()))

    df.set_index("Datetime", inplace=True)
    df["Option_Type"] = opttype
    df["Strike_Price"] = strike
    



    print(df)
    return df







def opt_exp(ticker):
    # Filter the relevant data for the given ticker and options
    dates = instrument_df[
        (instrument_df.SymbolRoot == ticker) & ((instrument_df.ScripType == 'CE') | (instrument_df.ScripType == 'PE'))]

    # Get the unique expiry dates and convert them to datetime objects
    dates = dates['Expiry'].unique().tolist()

    dates = [dt.datetime.strptime(date, '%Y-%m-%d') for date in dates]

    # Get today's date
    today = dt.datetime.today()

    # Sort the dates in ascending order
    dates.sort()

    # Find the next available date after today (skip today's date)
    future_dates = [date for date in dates if date > today]

    if future_dates:
        # Get the next available future date
        trade = future_dates[0]
    else:
        # If no future date is found, return None or handle the fallback
        return "No future expiry dates available."

    # Return the selected expiry date in the desired format
    return trade.strftime('%d %b %Y')



def process_expiry_date(date_str):
    # Extract the timestamp from the '/Date(...)' format
    timestamp = int(date_str.split('(')[1].split('+')[0])

    # Convert the timestamp to a datetime object
    date = dt.datetime.utcfromtimestamp(timestamp / 1000.0)

    # Convert datetime to the required format
    formatted_date = date.strftime('%d %b %Y')

    return timestamp, formatted_date


DELTA = 30

def fetch_option_data(option_string):
    print(option_string)
    expiry1 = opt_exp("NIFTY")
    print(expiry1)
    parts = option_string.split()
    ticker = parts[0]  # Extract ticker
    expiry = f"{parts[1]} {parts[2]} {parts[3]}"  # Extract expiry date
    opttype = parts[4]  # Extract option type (CE/PE)
    strike = float(parts[5])  # Extract strike price and convert to float

    target_strike = int(strike)  # Convert strike price to integer

    a = client.get_expiry("N", ticker)
    expiry_list = pd.DataFrame(a['Expiry'])
    spot_price = a['lastrate'][0]['LTP']

    # Process expiry dates
    expiry_list['Timestamp'], expiry_list['Format'] = zip(*expiry_list['ExpiryDate'].apply(process_expiry_date))

    # Get timestamp for target expiry
    timestamp_row = expiry_list[expiry_list.Format == expiry1]
    if timestamp_row.empty:
        print("Error: Expiry date not found")
        return None
    timestamp = timestamp_row.Timestamp.values[0]

    # Fetch option chain for the expiry
    option_chain = client.get_option_chain("N", ticker, timestamp)
    option_chain = pd.DataFrame(option_chain['Options'])

    # Filter for specified option type (CE or PE) & non-zero last traded price
    option_chain = option_chain[(option_chain.CPType == opttype) & (option_chain.LastRate != 0)]

    # Filter only for the target strike price
    option_chain = option_chain[option_chain.StrikeRate == target_strike]

    if option_chain.empty:
        print("Error: No data for target strike price")
        return None

    option_chain['SPOT'] = spot_price
    startTime = dt.datetime.today()
    date_obj = dt.datetime.strptime(expiry, "%d %b %Y")
    daysToExpiry = max((date_obj-startTime).days, 1)  # Ensure non-negative days

    # Create DataFrame
    opt_data = pd.DataFrame()
    opt_data['SPOT'] = option_chain['SPOT']
    opt_data['STRIKE'] = option_chain['StrikeRate']
    opt_data[f'{opttype}_LTP'] = option_chain['LastRate']
    opt_data['OI'] = option_chain['OpenInterest']
    opt_data['SYMBOL'] = option_chain['Name']
    opt_data = opt_data.reset_index(drop=True)

    Delta, Gamma, Theta, IV = [], [], [], []

    # Calculate Implied Volatility, Delta, Gamma, Theta
    r = 10  # Risk-free rate
    for i in range(len(opt_data)):
        c = mb.BS([opt_data['SPOT'][i], opt_data['STRIKE'][i], r, daysToExpiry],
                  callPrice=opt_data[f'{opttype}_LTP'][i])
        civ = c.impliedVolatility  # Fetch implied volatility
        cg = mb.BS([opt_data['SPOT'][i], opt_data['STRIKE'][i], r, daysToExpiry], volatility=civ)

        if opttype == 'CE':
            Delta.append(cg.callDelta * 100)
            Theta.append(cg.callTheta)
        else:
            Delta.append(cg.putDelta * 100)
            Theta.append(cg.putTheta)

        Gamma.append(cg.gamma * 100)  # Convert to percentage
        IV.append(civ)  # Store IV

    # Storing calculated Greeks in DataFrame
    opt_data[f'{opttype}_Delta'] = Delta
    opt_data[f'{opttype}_Gamma'] = Gamma
    opt_data[f'{opttype}_Theta'] = Theta
    opt_data['Implied_Volatility'] = IV

    opt_data["inserted_at"] = dt.datetime.now(UTC).strftime("%d-%b-%Y %I:%M%p")
    records = opt_data.to_dict("records")

    # Save to Excel
    if opttype == 'CE':
        db['NIFTY_CE'].insert_many(opt_data.to_dict('records'))
    else:
        db['NIFTY_PE'].insert_many(opt_data.to_dict('records'))

    return opt_data



def volume_oscillator(df, fast=14, slow=28):
    ema_fast = df['Volume'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Volume'].ewm(span=slow, adjust=False).mean()
    vo = ((ema_fast - ema_slow) / ema_slow) * 100
    return vo


def detect_trendline_touches_for_strategy(df: pd.DataFrame,
                                          swing_period: int = 8,
                                          min_hl_points: int = 2,
                                          lookback_candles: int = 50,
                                          tolerance: float = 0.007,
                                          extend_back: bool = True):
    """
    Trendline touch detector that extends line BACKWARDS to find ALL touches

    Args:
        df: DataFrame with OHLC data
        swing_period: Candles to look left/right (8 = 24 minutes for 3m)
        min_hl_points: Minimum Higher Lows to connect (2-3)
        lookback_candles: Analyze last N candles (50 = 2.5 hours for 3m)
        tolerance: Touch tolerance (0.007 = 0.7%)
        extend_back: Extend trendline backwards to find earlier touches

    Returns:
        List of indices where touches occur
    """

    # Only analyze recent candles
    if len(df) > lookback_candles:
        start_idx = len(df)-lookback_candles
        df_recent = df.iloc[start_idx:].copy()
        offset = start_idx
    else:
        df_recent = df.copy()
        offset = 0

    # Find swing lows
    swing_lows = []
    for i in range(swing_period, len(df_recent)):
        # Allow recent candles without full right-side confirmation
        if i+swing_period > len(df_recent):
            current_low = df_recent['Low'].iloc[i]
            left_lows = df_recent['Low'].iloc[max(0, i-swing_period):i]

            if len(left_lows) > 0 and current_low <= left_lows.min():
                swing_lows.append((i+offset, current_low))
        else:
            current_low = df_recent['Low'].iloc[i]
            left_lows = df_recent['Low'].iloc[i-swing_period:i]
            right_lows = df_recent['Low'].iloc[i+1:min(len(df_recent), i+swing_period+1)]

            if current_low <= left_lows.min() and current_low <= right_lows.min():
                swing_lows.append((i+offset, current_low))

    if len(swing_lows) < 2:
        # print(f"⚠️ Only found {len(swing_lows)} swing lows. Need at least 2.")
        return []

    # print(f"✓ Found {len(swing_lows)} swing lows")

    # Filter for Higher Lows
    higher_lows = [swing_lows[0]]
    for i in range(1, len(swing_lows)):
        idx, low = swing_lows[i]
        prev_idx, prev_low = higher_lows[-1]

        # Accept if higher or nearly equal (0.2% tolerance)
        if low >= prev_low * 0.998:
            higher_lows.append((idx, low))

    if len(higher_lows) < min_hl_points:
        # print(f"⚠️ Only found {len(higher_lows)} Higher Lows. Need at least {min_hl_points}.")
        return []

    # print(f"✓ Found {len(higher_lows)} Higher Lows")
    # print(f"   Points: {[(idx, round(price, 2)) for idx, price in higher_lows]}")

    # Take most recent Higher Lows for trendline
    recent_hl = higher_lows[-min_hl_points:]
    idx1, price1 = recent_hl[0]
    idx2, price2 = recent_hl[-1]

    # Calculate trendline
    slope = (price2-price1) / (idx2-idx1)

    # Accept flat or upward lines
    if slope < -0.0001:
        # print(f"⚠️ Negative slope: {slope:.6f}. Not an uptrend.")
        return []

    intercept = price1-slope * idx1


    print(f"✓ Trendline: slope={slope:.6f}, intercept={intercept:.2f}")
    print(f"   Line connects indices {idx1} to {idx2}")

    # Extend trendline backwards
    if extend_back:
        search_start = max(0, len(df)-lookback_candles)
        # print(f"   Extending line backwards from index {idx1} to index {search_start}")
    else:
        search_start = idx1

    # Detect touches from the START of lookback period to END of data
    touch_indices = []

    for i in range(search_start, len(df)):
        line_value = slope * i+intercept
        threshold = line_value * tolerance

        current_low = df['Low'].iloc[i]
        current_high = df['High'].iloc[i]
        current_close = df['Close'].iloc[i]

        # Skip anchor points
        is_anchor_point = any(i == hl_idx for hl_idx, _ in recent_hl)

        if i > 0 and not is_anchor_point:
            prev_close = df['Close'].iloc[i-1]

            # Detection methods
            touch_from_above = (prev_close > line_value and
                                current_low <= line_value+threshold)

            crosses_line = (current_low <= line_value+threshold and
                            current_high >= line_value-threshold)

            close_near_line = abs(current_close-line_value) <= threshold

            low_touches = abs(current_low-line_value) <= threshold

            # Detect touch
            if touch_from_above or crosses_line or close_near_line or low_touches:
                # Additional validation: don't add if price breaks significantly below
                if current_close > line_value-(threshold * 2):
                    touch_indices.append(i)
                    # touch_type = 'from_above' if touch_from_above else ('cross' if crosses_line else ('close' if close_near_line else 'low_bounce'))

                    # Only print recent touches to avoid spam
                    # if i >= len(df) - 20:
                    #     print(f"   Touch at index {i}: Low={current_low:.2f}, Close={current_close:.2f}, Line={line_value:.2f} ({touch_type})")

    # print(f"✓ Found {len(touch_indices)} total touches (from index {search_start} to {len(df)-1})")

    # Show distribution of touches
    # if len(touch_indices) > 0:
    #     print(f"   First touch at index: {touch_indices[0]}")
    #     print(f"   Last touch at index: {touch_indices[-1]}")
    #     print(f"   Touch indices: {touch_indices}")

    return touch_indices










GROQ_API_KEY = os.getenv("GROQ_API_KEY")










def fetch_ohlcv(symbol, timeframe="3m", candles=30, existing_data=None):
    """
    Fetch OHLCV data for LLM analysis.
    If existing_data is provided, use it directly.
    Otherwise fetch from 5paisa API.
    """
    print('📡 fetch_ohlcv called')

    # ═══════════════════════════════════════════════════════════════
    #  PATH 1: Use existing data (same data super_trend analyzed)
    # ═══════════════════════════════════════════════════════════════
    if existing_data is not None and isinstance(existing_data, pd.DataFrame):
        print('  ✅ Using existing DataFrame (same as super_trend input)')

        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(existing_data.columns):
            print(f"  ❌ Missing columns: {required - set(existing_data.columns)}")
            return None

        df = existing_data.tail(candles).copy()

        ohlcv = []
        for _, r in df.iterrows():
            try:
                ohlcv.append({
                    "o": round(float(r["Open"]), 2),
                    "h": round(float(r["High"]), 2),
                    "l": round(float(r["Low"]), 2),
                    "c": round(float(r["Close"]), 2),
                    "v": int(float(r["Volume"]))
                })
            except (ValueError, TypeError) as e:
                print(f"  ⚠️ Skipping bad row: {e}")
                continue

        if len(ohlcv) == 0:
            print("  ❌ No valid OHLCV rows")
            return None

        print(f"  📊 Converted {len(ohlcv)} candles from existing data")
        return ohlcv

    # ═══════════════════════════════════════════════════════════════
    #  PATH 2: Fresh API fetch (fallback)
    # ═══════════════════════════════════════════════════════════════
    print('  📡 Fetching fresh from 5paisa API...')

    scripcode = scripcode_lookup(instrument_df, symbol)
    print(f'  ScripCode: {scripcode}')
    if not scripcode:
        print("  ❌ ScripCode not found")
        return None

    try:
        raw = client.historical_data(
            Exch='N', ExchangeSegment='D',
            ScripCode=scripcode,
            time=timeframe,
            From=dt.date.today() - dt.timedelta(5),
            To=dt.date.today()
        )
    except Exception as e:
        print(f"  ❌ API call failed: {e}")
        return None

    # ═══════════════════════════════════════════════════════════════
    #  🔥 FIX: Handle both DataFrame and list/dict returns
    # ═══════════════════════════════════════════════════════════════
    if isinstance(raw, pd.DataFrame):
        df = raw.copy()
    elif isinstance(raw, list):
        if len(raw) == 0:
            print("  ❌ API returned empty list")
            return None
        df = pd.DataFrame(raw)
    elif isinstance(raw, dict):
        df = pd.DataFrame([raw])
    else:
        print(f"  ❌ Unexpected API return type: {type(raw)}")
        return None

    # Check if DataFrame is empty
    if df.empty:
        print("  ❌ API returned empty DataFrame")
        return None

    print(f"  📦 API returned {len(df)} rows")
    print(f"  📋 Columns: {list(df.columns)}")

    # ─── Datetime handling ───
    datetime_col = None
    for col in ["Datetime", "DateTime", "Date", "Time", "datetime", "date"]:
        if col in df.columns:
            datetime_col = col
            break

    if datetime_col:
        try:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df = df.sort_values(datetime_col)
        except Exception as e:
            print(f"  ⚠️ DateTime parse warning: {e}")

    # ─── Column normalization ───
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ('open', 'o'):
            col_map[col] = 'Open'
        elif cl in ('high', 'h'):
            col_map[col] = 'High'
        elif cl in ('low', 'l'):
            col_map[col] = 'Low'
        elif cl in ('close', 'c'):
            col_map[col] = 'Close'
        elif cl in ('volume', 'v', 'vol'):
            col_map[col] = 'Volume'

    if col_map:
        df = df.rename(columns=col_map)

    # ─── Validate columns ───
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(f"  ❌ Missing columns: {missing}")
        print(f"  Available: {list(df.columns)}")
        return None

    # ─── Drop extra columns, keep only OHLCV ───
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df = df.tail(candles)

    # ─── Convert to list of dicts ───
    ohlcv = []
    for _, r in df.iterrows():
        try:
            ohlcv.append({
                "o": round(float(r["Open"]), 2),
                "h": round(float(r["High"]), 2),
                "l": round(float(r["Low"]), 2),
                "c": round(float(r["Close"]), 2),
                "v": int(float(r["Volume"]))
            })
        except (ValueError, TypeError) as e:
            print(f"  ⚠️ Skipping bad row: {e}")
            continue

    if len(ohlcv) == 0:
        print("  ❌ No valid OHLCV rows")
        return None

    print(f"  ✅ Returning {len(ohlcv)} candles")
    return ohlcv





def llm_trade_signal(symbol, ohlcv, timeframe, signal_context=None):
    """
    ADVANCED LLM FILTER — REJECT BAD TRADES ONLY

    Role:
    - DO NOT generate signals
    - DO NOT overthink indicators
    - ONLY detect traps & risky setups

    Output:
    - "buy" → allow trade
    - "no trade" → reject trade
    """

    print('🧠 llm_trade_signal (ADVANCED FILTER MODE)')

    if not ohlcv or len(ohlcv) < 12:
        return _no_trade_response("Not enough candles")

    closes  = [c["c"] for c in ohlcv]
    highs   = [c["h"] for c in ohlcv]
    lows    = [c["l"] for c in ohlcv]
    opens   = [c["o"] for c in ohlcv]
    volumes = [c["v"] for c in ohlcv]

    current_price = closes[-1]

    # ─────────────────────────────────────────────
    # 🔹 Compact Structured Context (IMPORTANT)
    # ─────────────────────────────────────────────
    last_6 = [
        {
            "o": round(opens[i],2),
            "h": round(highs[i],2),
            "l": round(lows[i],2),
            "c": round(closes[i],2)
        }
        for i in range(-6, 0)
    ]

    avg_vol_5 = sum(volumes[-5:]) / 5
    vol_ratio = round(volumes[-1] / avg_vol_5, 2) if avg_vol_5 > 0 else 1

    recent_high = max(highs[-10:])
    recent_low  = min(lows[-10:])
    range_pos = round((current_price - recent_low) / (recent_high - recent_low), 2) if recent_high != recent_low else 0.5

    primary_reason = signal_context.get("reason","") if signal_context else ""

    # ─────────────────────────────────────────────
    # 🔥 ADVANCED PROMPT (THIS IS THE EDGE)
    # ─────────────────────────────────────────────
    prompt = f"""
You are an ELITE OPTIONS SCALPING RISK MANAGER.

You DO NOT give buy signals.
You ONLY REJECT bad trades.

Analyze the structure deeply like a professional trader.

DATA:
Candles (last 6):
{last_6}

Volume ratio (current / avg5): {vol_ratio}
Range position (0=low, 1=high): {range_pos}
Recent High: {recent_high}
Recent Low: {recent_low}

Primary Signal Reason:
{primary_reason}

----------------------------------------

REJECT TRADE if you see ANY of these:

1. CHOPPY MARKET:
- overlapping candles
- no clear direction
- alternating colors

2. FAKE BREAKOUT:
- price near high but weak close
- upper wicks / rejection
- breakout without volume follow-through

3. EXHAUSTION MOVE:
- 2-3 strong candles already done
- late entry near top

4. LOW QUALITY MOMENTUM:
- small bodies
- inconsistent candle structure

5. BAD LOCATION:
- range_pos > 0.75 (too close to resistance)
- no room to move

----------------------------------------

ALLOW TRADE ONLY IF:
- clean directional candles
- strong closes near highs
- no rejection wicks
- not extended already

----------------------------------------

STRICT RULE:
When in doubt → REJECT TRADE

----------------------------------------

Return ONLY JSON:

{{
  "signal": "buy" or "no trade",
  "confidence": 0.0-1.0,
  "reason": "very short explanation"
}}
"""

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "temperature": 0.0,   # ❗ ultra deterministic
                "max_tokens": 120,
                "messages": [
                    {"role": "system", "content": (
                        "You are a strict risk manager. "
                        "You reject bad trades aggressively. "
                        "You NEVER hallucinate. "
                        "Output ONLY valid JSON."
                    )},
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=10
        )

        if response.status_code != 200:
            return _no_trade_response(f"API error {response.status_code}")

        content = response.json()["choices"][0]["message"]["content"].strip()

        if "```" in content:
            content = content.replace("```json","").replace("```","")

        result = json.loads(content[content.index("{"):content.rindex("}")+1])

        signal = str(result.get("signal","no trade")).lower()
        confidence = float(result.get("confidence",0.0))

        if signal not in ("buy","no trade"):
            signal = "no trade"

        # ─────────────────────────────────────────────
        # 🔒 FINAL SAFETY (VERY IMPORTANT)
        # ─────────────────────────────────────────────
        if confidence < 0.55:
            signal = "no trade"

        entry = current_price
        stop_loss = round(current_price - 10, 2)
        target_1  = round(current_price + 20, 2)
        target_2  = round(current_price + 30, 2)

        final = {
            "signal": signal,
            "confidence": round(confidence,2),
            "entry": entry,
            "stop_loss": stop_loss,
            "target_1": target_1,
            "target_2": target_2,
            "reason": result.get("reason","")
        }

        print(f"  🧠 FILTER RESULT: {signal} | {confidence} | {final['reason']}")
        return final

    except Exception as e:
        return _no_trade_response(f"LLM error: {e}")





import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class OrderBlock:
    """Data class to store order block information"""
    value: float
    bar_start: int
    bar_end: int
    block_type: str  # 'bullish' or 'bearish'
    start_datetime: pd.Timestamp
    end_datetime: pd.Timestamp


@dataclass
class TrendLine:
    """Data class to store trend line information"""
    start_idx: int
    end_idx: int
    start_value: float
    end_value: float
    start_datetime: pd.Timestamp
    end_datetime: pd.Timestamp
    trend_type: str  # 'bullish' or 'bearish'
    slope: float


class PriceActionAnalyzer:

    def __init__(self, df: pd.DataFrame, zigzag_length: int = 9, atr_period: int = 14):
        self.df = df.copy().reset_index(drop=True)
        self.df_with_datetime = df.copy()
        self.zigzag_length = zigzag_length
        self.atr_period = atr_period
        self._calculate_atr()

    def _calculate_atr(self):
        high_low    = self.df['High'] - self.df['Low']
        high_close  = abs(self.df['High'] - self.df['Close'].shift())
        low_close   = abs(self.df['Low']  - self.df['Close'].shift())
        true_range  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['ATR'] = true_range.rolling(window=self.atr_period).mean()

    def _identify_zigzag_points(self) -> Tuple[List[int], List[float], List[int], List[float]]:
        high_indices, high_values = [], []
        low_indices,  low_values  = [], []
        trend = 1

        # ── FIX: exclude the last bar (may be incomplete in live market) ──
        safe_end = len(self.df) - self.zigzag_length - 1

        for i in range(self.zigzag_length, safe_end):
            if trend == 1:
                window_high = self.df['High'].iloc[i - self.zigzag_length: i + self.zigzag_length + 1].max()
                if self.df['High'].iloc[i] == window_high:
                    high_indices.append(i)
                    high_values.append(self.df['High'].iloc[i])
                    trend = -1

            if trend == -1:
                window_low = self.df['Low'].iloc[i - self.zigzag_length: i + self.zigzag_length + 1].min()
                if self.df['Low'].iloc[i] == window_low:
                    low_indices.append(i)
                    low_values.append(self.df['Low'].iloc[i])
                    trend = 1

        return high_indices, high_values, low_indices, low_values

    def find_order_blocks(self, max_blocks: int = 20) -> List[OrderBlock]:
        order_blocks = []
        high_indices, high_values, low_indices, low_values = self._identify_zigzag_points()

        if len(high_indices) < 2 or len(low_indices) < 1:
            return order_blocks

        for i in range(len(high_indices) - 1):
            high_idx     = high_indices[i]
            next_low_idx = low_indices[i] if i < len(low_indices) else len(self.df) - 1

            if high_idx < next_low_idx:
                slice_data    = self.df['High'].iloc[high_idx: next_low_idx + 1]
                max_high_value = slice_data.max()
                max_high_pos   = high_idx + slice_data.argmax()

                order_blocks.append(OrderBlock(
                    value=max_high_value,
                    bar_start=max_high_pos,
                    bar_end=len(self.df) - 1,
                    block_type='bearish',
                    start_datetime=self.df_with_datetime.index[max_high_pos],
                    end_datetime=self.df_with_datetime.index[-1]
                ))

        for i in range(len(low_indices) - 1):
            low_idx       = low_indices[i]
            next_high_idx = high_indices[i + 1] if i + 1 < len(high_indices) else len(self.df) - 1

            if low_idx < next_high_idx:
                slice_data    = self.df['Low'].iloc[low_idx: next_high_idx + 1]
                min_low_value = slice_data.min()
                min_low_pos   = low_idx + slice_data.argmin()

                order_blocks.append(OrderBlock(
                    value=min_low_value,
                    bar_start=min_low_pos,
                    bar_end=len(self.df) - 1,
                    block_type='bullish',
                    start_datetime=self.df_with_datetime.index[min_low_pos],
                    end_datetime=self.df_with_datetime.index[-1]
                ))

        unique_blocks = {}
        for block in order_blocks:
            key = (block.block_type, block.value)
            if key not in unique_blocks:
                unique_blocks[key] = block

        sorted_blocks = sorted(unique_blocks.values(), key=lambda x: x.bar_end, reverse=True)
        return sorted_blocks[-max_blocks:] if len(sorted_blocks) > max_blocks else sorted_blocks

    def find_trend_lines(self, trend_line_length: int = 20) -> List[TrendLine]:
        trend_lines = []
        high_indices, high_values, low_indices, low_values = self._identify_zigzag_points()

        if len(low_indices) >= 2:
            for i in range(len(low_indices) - 1):
                start_idx   = low_indices[i]
                end_idx     = low_indices[i + 1]
                start_value = low_values[i]
                end_value   = low_values[i + 1]
                if end_idx == start_idx:
                    continue
                slope = (end_value - start_value) / (end_idx - start_idx)
                if slope > 0:
                    trend_lines.append(TrendLine(
                        start_idx=start_idx, end_idx=end_idx,
                        start_value=start_value, end_value=end_value,
                        start_datetime=self.df_with_datetime.index[start_idx],
                        end_datetime=self.df_with_datetime.index[end_idx],
                        trend_type='bullish', slope=slope
                    ))

        if len(high_indices) >= 2:
            for i in range(len(high_indices) - 1):
                start_idx   = high_indices[i]
                end_idx     = high_indices[i + 1]
                start_value = high_values[i]
                end_value   = high_values[i + 1]
                if end_idx == start_idx:
                    continue
                slope = (end_value - start_value) / (end_idx - start_idx)
                if slope < 0:
                    trend_lines.append(TrendLine(
                        start_idx=start_idx, end_idx=end_idx,
                        start_value=start_value, end_value=end_value,
                        start_datetime=self.df_with_datetime.index[start_idx],
                        end_datetime=self.df_with_datetime.index[end_idx],
                        trend_type='bearish', slope=slope
                    ))

        return trend_lines

    def extend_trend_line(self, trend_line: TrendLine, extend_to_idx: int) -> Tuple[int, float]:
        slope          = trend_line.slope
        extended_value = trend_line.start_value + slope * (extend_to_idx - trend_line.start_idx)
        return extend_to_idx, extended_value

    def get_order_block_zones(self, order_blocks: List[OrderBlock]) -> pd.DataFrame:
        zones = []
        for ob in order_blocks:
            zones.append({
                'datetime': ob.start_datetime, 'type': ob.block_type,
                'value': ob.value, 'bar_start': ob.bar_start,
                'bar_end': ob.bar_end, 'start_time': ob.start_datetime,
                'end_time': ob.end_datetime
            })
        return pd.DataFrame(zones)

    def get_trend_lines_info(self, trend_lines: List[TrendLine]) -> pd.DataFrame:
        lines = []
        for tl in trend_lines:
            lines.append({
                'type': tl.trend_type, 'start_datetime': tl.start_datetime,
                'end_datetime': tl.end_datetime, 'start_value': tl.start_value,
                'end_value': tl.end_value, 'slope': tl.slope,
                'start_idx': tl.start_idx, 'end_idx': tl.end_idx
            })
        return pd.DataFrame(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  LIVE MARKET FIX — normalise a Timestamp to a minute-precision string
#  that is TIMEZONE-AGNOSTIC and MICROSECOND-AGNOSTIC.
#  Both cash-market and option-data timestamps go through this before
#  any string comparison, so tz-aware vs tz-naive mismatches are gone.
# ═══════════════════════════════════════════════════════════════════════════
def _ts_to_min(ts) -> str:
    """
    Convert any timestamp (pd.Timestamp, str, datetime) → 'YYYY-MM-DD HH:MM'.
    Strips timezone, microseconds, and seconds.
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_localize(None)           # strip tz
    return t.strftime('%Y-%m-%d %H:%M')   # minute-precision, no 'T'


def super_trend(symbol, data, use_llm=False):
    """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║      DUAL‑BRANCH TRADING SIGNAL ENGINE  (v4 — LIVE MARKET FIXED)    ║
    ║                                                                      ║
    ║  BRANCH-2 (unchanged logic):                                         ║
    ║    • Bullish OB  → BUY_CE  immediately at ob.start_datetime         ║
    ║    • Bearish OB  → BUY_PE  immediately at ob.start_datetime         ║
    ║    • Bullish TL  → BUY_CE  immediately at tl.end_datetime           ║
    ║    • Bearish TL  → BUY_PE  immediately at tl.end_datetime           ║
    ║    • Datetime-matched, every OB/TL independent, no cooldown         ║
    ║                                                                      ║
    ║  BRANCH-1 (unchanged logic — Stochastic Confluence Engine):         ║
    ║    F1. Both K & D oversold (<20) — armed                            ║
    ║    F2. K crosses D while K<30 — cross confirmed                     ║
    ║    F3. K velocity accelerating (Kvel>0 AND Kvel>=KvelPrev)          ║
    ║    F4. K/D spread > 1.5 pts (meaningful separation)                 ║
    ║    F5. CCI rising AND CCI > -100 (no trap)                          ║
    ║    F6. Bullish OB or TL within 2% of close (structure confluence)   ║
    ║    F7. Time window 09:35–15:10                                       ║
    ║                                                                      ║
    ║  LIVE MARKET FIXES (no logic change):                               ║
    ║    • _ts_to_min(): tz-agnostic, microsecond-agnostic minute match   ║
    ║    • _find_option_bar(): robust fallback with tz-stripped index      ║
    ║    • PriceActionAnalyzer excludes last (live/incomplete) bar         ║
    ║    • Branch-1 cooldown: fires on CURRENT bar, not suppressed         ║
    ║    • Branch-1 loop: processes bar i=n-1 (the live bar) correctly     ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """

    # ═══════════════════════════════════════════════════════
    #  HELPER — parse CE / PE from symbol
    # ═══════════════════════════════════════════════════════
    def _parse_option_type(sym: str) -> str:
        for part in sym.upper().split():
            if part in ('CE', 'PE'):
                return part
        return ''

    # ═══════════════════════════════════════════════════════
    #  HELPER — find matching option bar for a given datetime
    #
    #  FIX vs original:
    #   • Builds option_min_strs once using _ts_to_min() so that
    #     tz-aware / tz-naive / microsecond differences never
    #     cause a mismatch.
    #   • Uses ' ' separator (not 'T') matching _ts_to_min output.
    #   • Nearest-after fallback is preserved.
    # ═══════════════════════════════════════════════════════
    option_min_strs = [_ts_to_min(dt) for dt in data.index]   # built once

    def _find_option_bar(fire_dt) -> int:
        target_min = _ts_to_min(fire_dt)   # same normalisation

        # Exact minute match
        for pos, opt_min in enumerate(option_min_strs):
            if opt_min == target_min:
                return pos

        # No exact match → find closest bar AFTER fire_dt
        for pos, opt_min in enumerate(option_min_strs):
            if opt_min > target_min:
                return pos

        return -1   # beyond option data range

    # ═══════════════════════════════════════════════════════
    #  HEADER
    # ═══════════════════════════════════════════════════════
    print("=" * 80)
    print("  ⚙️   DUAL-BRANCH TRADING SIGNAL ENGINE (v4 — LIVE MARKET FIXED)")
    print(f"  SYMBOL    : {symbol}")
    print(f"  DATA SHAPE: {data.shape}")
    print(f"  DATE RANGE: {data.index[0]} to {data.index[-1]}")
    print("=" * 80)

    option_type = _parse_option_type(symbol)
    print(f"\n  📌 OPTION TYPE : {option_type if option_type else 'SPOT / INDEX'}")

    n     = len(data)
    close = data['Close']

    print(f"\n  📊 Data Info:")
    print(f"     Total bars   : {n}")
    print(f"     Close range  : {close.min():.2f} – {close.max():.2f}")
    print(f"     Current close: {close.iloc[-1]:.2f}")

    # ═══════════════════════════════════════════════════════
    #  BRANCH-1 INDICATORS
    # ═══════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  📊 CALCULATING BRANCH-1 INDICATORS (Stochastic + CCI)")
    print("─" * 80)

    stoch        = ta.stoch(data['High'], data['Low'], data['Close'], 14, 3, 3)
    data['Stoch_K'] = stoch['STOCHk_14_3_3']
    data['Stoch_D'] = stoch['STOCHd_14_3_3']
    data['CCI']     = ta.cci(data['High'], data['Low'], data['Close'], length=20)

    K            = data['Stoch_K']
    D            = data['Stoch_D']
    CCI          = data['CCI']
    k_vel        = K - K.shift(1)
    cci_momentum = CCI - CCI.shift(1)

    # ── Time filter  9:35 AM – 3:10 PM ──────────────────────────
    if isinstance(data.index, pd.DatetimeIndex):
        time_series = data.index.time
    else:
        time_series = pd.to_datetime(data.index).time

    time_mins = pd.Series(
        [t.hour * 60 + t.minute for t in time_series],
        index=data.index
    )
    TIME_OK = (time_mins >= 575) & (time_mins <= 910)
    print(f"  ✅ Stochastic (K, D) and CCI calculated")

    # ═══════════════════════════════════════════════════════
    #  BRANCH-2 — PriceActionAnalyzer
    # ═══════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  📐  BRANCH 2 — CALLING PriceActionAnalyzer")
    print("─" * 80)

    analyzer     = None
    order_blocks = []
    trend_lines  = []

    try:
        ndata    = get_cash_market_data_3('5m')
        analyzer = PriceActionAnalyzer(ndata, zigzag_length=9, atr_period=14)
        print(f"  ✅ PriceActionAnalyzer initialized")
    except Exception as e:
        print(f"  ❌ ERROR initializing PriceActionAnalyzer: {e}")
        print(f"  ⚠️  Skipping Branch-2")

    if analyzer is not None:
        try:
            order_blocks = analyzer.find_order_blocks(max_blocks=20)
            print(f"  ✅ Order blocks found: {len(order_blocks)}")
        except Exception as e:
            print(f"  ❌ ERROR finding order blocks: {e}")

        try:
            trend_lines = analyzer.find_trend_lines(trend_line_length=20)
            print(f"  ✅ Trend lines found: {len(trend_lines)}")
        except Exception as e:
            print(f"  ❌ ERROR finding trend lines: {e}")

    bearish_obs = [ob for ob in order_blocks if ob.block_type == 'bearish']
    bullish_obs = [ob for ob in order_blocks if ob.block_type == 'bullish']
    bearish_tls = [tl for tl in trend_lines if tl.trend_type == 'bearish']
    bullish_tls = [tl for tl in trend_lines if tl.trend_type == 'bullish']

    # ─────────────────────────────────────────────────────────────
    #  PRINT — order blocks
    # ─────────────────────────────────────────────────────────────
    print(f"\n  📊 ORDER BLOCKS  total={len(order_blocks)}"
          f"  (🔴 Bearish={len(bearish_obs)}  |  🟢 Bullish={len(bullish_obs)})")
    print(f"  {'─' * 78}")

    if bearish_obs:
        print("  🔴 BEARISH ORDER BLOCKS → BUY_PE at start_datetime:")
        for i, ob in enumerate(bearish_obs, 1):
            print(f"     {i:2d}. Value={ob.value:.2f}"
                  f"  |  start_datetime={ob.start_datetime}"
                  f"  |  bar_start={ob.bar_start}")

    if bullish_obs:
        print("  🟢 BULLISH ORDER BLOCKS → BUY_CE at start_datetime:")
        for i, ob in enumerate(bullish_obs, 1):
            print(f"     {i:2d}. Value={ob.value:.2f}"
                  f"  |  start_datetime={ob.start_datetime}"
                  f"  |  bar_start={ob.bar_start}")

    if not order_blocks:
        print("  ⚠️  No order blocks found!")

    # ─────────────────────────────────────────────────────────────
    #  PRINT — trend lines
    # ─────────────────────────────────────────────────────────────
    print(f"\n  📉 TREND LINES   total={len(trend_lines)}"
          f"  (🔴 Bearish={len(bearish_tls)}  |  🟢 Bullish={len(bullish_tls)})")
    print(f"  {'─' * 78}")

    if bearish_tls:
        print("  🔴 BEARISH TREND LINES → BUY_PE at end_datetime:")
        for i, tl in enumerate(bearish_tls, 1):
            print(f"     {i:2d}. Slope={tl.slope:.4f}"
                  f"  |  {tl.start_value:.2f}@{tl.start_datetime}"
                  f"  →  {tl.end_value:.2f}@{tl.end_datetime}"
                  f"  |  end_idx={tl.end_idx}")

    if bullish_tls:
        print("  🟢 BULLISH TREND LINES → BUY_CE at end_datetime:")
        for i, tl in enumerate(bullish_tls, 1):
            print(f"     {i:2d}. Slope={tl.slope:.4f}"
                  f"  |  {tl.start_value:.2f}@{tl.start_datetime}"
                  f"  →  {tl.end_value:.2f}@{tl.end_datetime}"
                  f"  |  end_idx={tl.end_idx}")

    if not trend_lines:
        print("  ⚠️  No trend lines found!")

    # ═══════════════════════════════════════════════════════
    #  BRANCH-2 SIGNAL STORAGE
    # ═══════════════════════════════════════════════════════
    all_b2_events = []

    # ═══════════════════════════════════════════════════════
    #  BRANCH-2 FIRE — ORDER BLOCKS
    # ═══════════════════════════════════════════════════════
    print(f"\n  {'─' * 80}")
    print(f"  ⚡  FIRING OB SIGNALS  (datetime-matched, every OB independent)")
    print(f"  {'─' * 80}")

    for ob_idx, ob in enumerate(order_blocks):
        fire_pos = _find_option_bar(ob.start_datetime)   # uses new tz-safe helper

        if fire_pos < 0 or fire_pos >= n:
            print(f"  ⚠️  OB #{ob_idx+1} [{ob.block_type}]"
                  f" start_dt={ob.start_datetime} → no matching option bar, skip")
            continue

        fire_dt    = data.index[fire_pos]
        fire_close = close.iloc[fire_pos]
        signal     = 'BUY_CE' if ob.block_type == 'bullish' else 'BUY_PE'

        event = {
            'pos'    : fire_pos,
            'dt'     : fire_dt,
            'signal' : signal,
            'source' : 'ORDER_BLOCK',
            'detail' : (
                f"{ob.block_type.capitalize()} OB @ {ob.value:.2f}"
                f" [cash_dt={ob.start_datetime}]"
                f" option_close={fire_close:.2f}"
            ),
            'level'  : ob.value,
            'close'  : fire_close,
        }
        all_b2_events.append(event)

        icon = '🟢' if signal == 'BUY_CE' else '🔴'
        print(f"  {icon} OB #{ob_idx+1:2d} [{ob.block_type:7s}] → {signal}"
              f"  cash_dt={ob.start_datetime}"
              f"  option_bar={fire_pos} ({fire_dt})"
              f"  close={fire_close:.2f}  OB_level={ob.value:.2f}")

    # ═══════════════════════════════════════════════════════
    #  BRANCH-2 FIRE — TREND LINES
    # ═══════════════════════════════════════════════════════
    print(f"\n  {'─' * 80}")
    print(f"  ⚡  FIRING TL SIGNALS  (datetime-matched, every TL independent)")
    print(f"  {'─' * 80}")

    for tl_idx, tl in enumerate(trend_lines):
        fire_pos = _find_option_bar(tl.end_datetime)   # tz-safe

        if fire_pos < 0 or fire_pos >= n:
            print(f"  ⚠️  TL #{tl_idx+1} [{tl.trend_type}]"
                  f" end_dt={tl.end_datetime} → no matching option bar, skip")
            continue

        fire_dt    = data.index[fire_pos]
        fire_close = close.iloc[fire_pos]
        tl_val     = tl.end_value
        signal     = 'BUY_CE' if tl.trend_type == 'bullish' else 'BUY_PE'

        event = {
            'pos'    : fire_pos,
            'dt'     : fire_dt,
            'signal' : signal,
            'source' : 'TREND_LINE',
            'detail' : (
                f"{tl.trend_type.capitalize()} TL @ {tl_val:.2f}"
                f" [{tl.start_datetime} → {tl.end_datetime}]"
                f" option_close={fire_close:.2f}"
            ),
            'level'  : tl_val,
            'close'  : fire_close,
        }
        all_b2_events.append(event)

        icon = '🟢' if signal == 'BUY_CE' else '🔴'
        print(f"  {icon} TL #{tl_idx+1:2d} [{tl.trend_type:7s}] → {signal}"
              f"  cash_dt={tl.end_datetime}"
              f"  option_bar={fire_pos} ({fire_dt})"
              f"  close={fire_close:.2f}  TL_level={tl_val:.2f}")

    # ═══════════════════════════════════════════════════════
    #  SUMMARY OF ALL B2 EVENTS
    # ═══════════════════════════════════════════════════════
    print(f"\n  {'─' * 80}")
    print(f"  📊 BRANCH-2 ALL EVENTS  (total={len(all_b2_events)})")
    print(f"  {'─' * 80}")

    total_ce = sum(1 for e in all_b2_events if e['signal'] == 'BUY_CE')
    total_pe = sum(1 for e in all_b2_events if e['signal'] == 'BUY_PE')
    total_ob = sum(1 for e in all_b2_events if e['source'] == 'ORDER_BLOCK')
    total_tl = sum(1 for e in all_b2_events if e['source'] == 'TREND_LINE')

    print(f"  🟢 BUY_CE={total_ce}  🔴 BUY_PE={total_pe}"
          f"  |  ORDER_BLOCK={total_ob}  TREND_LINE={total_tl}")

    for ev in sorted(all_b2_events, key=lambda x: x['pos']):
        icon = '🟢' if ev['signal'] == 'BUY_CE' else '🔴'
        print(f"  {icon}  bar={ev['pos']:3d}  dt={ev['dt']}"
              f"  {ev['signal']}  [{ev['source']}]  {ev['detail']}")

    # ═══════════════════════════════════════════════════════
    #  BUILD BRANCH-2 SERIES
    # ═══════════════════════════════════════════════════════
    if option_type == 'CE':
        gated_events = [e for e in all_b2_events if e['signal'] == 'BUY_CE']
    elif option_type == 'PE':
        gated_events = [e for e in all_b2_events if e['signal'] == 'BUY_PE']
    else:
        gated_events = list(all_b2_events)

    print(f"\n  📊 BRANCH-2 GATED ({option_type if option_type else 'ALL'})"
          f"  events={len(gated_events)}")

    b2_signal_type_s = pd.Series('',  index=data.index, dtype=object)
    b2_source_type_s = pd.Series('',  index=data.index, dtype=object)
    b2_detail_s      = pd.Series('',  index=data.index, dtype=object)
    b2_level_s       = pd.Series(0.0, index=data.index, dtype=float)
    b2_raw           = pd.Series(0,   index=data.index, dtype=int)
    b2_all_events_s  = pd.Series([[] for _ in range(n)], index=data.index, dtype=object)

    filled_pos = set()
    for ev in sorted(gated_events, key=lambda x: x['pos']):
        pos = ev['pos']
        b2_all_events_s.iloc[pos] = b2_all_events_s.iloc[pos] + [ev]
        if pos not in filled_pos:
            b2_signal_type_s.iloc[pos] = ev['signal']
            b2_source_type_s.iloc[pos] = ev['source']
            b2_detail_s.iloc[pos]      = ev['detail']
            b2_level_s.iloc[pos]       = ev['level']
            b2_raw.iloc[pos]           = 1
            filled_pos.add(pos)

    raw_ce = int(((b2_raw == 1) & (b2_signal_type_s == 'BUY_CE')).sum())
    raw_pe = int(((b2_raw == 1) & (b2_signal_type_s == 'BUY_PE')).sum())

    print(f"  📊 BRANCH-2 RAW (bars with signal): {int(b2_raw.sum())}"
          f"  (🟢 BUY_CE={raw_ce}  |  🔴 BUY_PE={raw_pe})")
    print(f"  📊 BRANCH-2 TOTAL EVENTS (trades): {len(gated_events)}")

    b2_gated = b2_raw == 1

    b2_ce_count = int(((b2_gated) & (b2_signal_type_s == 'BUY_CE')).sum())
    b2_pe_count = int(((b2_gated) & (b2_signal_type_s == 'BUY_PE')).sum())

    # ═══════════════════════════════════════════════════════════════════════
    #  BRANCH-1 — STOCHASTIC CONFLUENCE ENGINE
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  📊  BRANCH 1 — UPGRADED STOCHASTIC CONFLUENCE ENGINE (v4)")
    print("─" * 80)

    k_vel_prev = k_vel.shift(1)

    structures_available = len(order_blocks) > 0 or len(trend_lines) > 0

    bullish_ob_datetimes = set()
    bullish_ob_values    = {}
    for ob in bullish_obs:
        key = str(ob.start_datetime)[:16]
        bullish_ob_datetimes.add(key)
        bullish_ob_values[key] = ob.value

    bullish_tl_end_values = {}
    for tl in bullish_tls:
        key = str(tl.end_datetime)[:16]
        bullish_tl_end_values[key] = tl.end_value

    def _structure_nearby(bar_idx: int, curr_close: float) -> tuple:
        CONFLUENCE_PCT = 0.02
        for ob in bullish_obs:
            if ob.bar_start <= bar_idx:
                dist = abs(curr_close - ob.value) / curr_close
                if dist <= CONFLUENCE_PCT:
                    return True, f"BullishOB@{ob.value:.2f}(dist={dist*100:.1f}%)"
        for tl in bullish_tls:
            if tl.end_idx <= bar_idx:
                tl_val = tl.end_value + tl.slope * (bar_idx - tl.end_idx)
                dist   = abs(curr_close - tl_val) / curr_close
                if dist <= CONFLUENCE_PCT:
                    return True, f"BullishTL@{tl_val:.2f}(dist={dist*100:.1f}%)"
        return False, "none"

    b1_raw       = np.zeros(n, dtype=int)
    signal_path   = [''] * n
    signal_grade  = [''] * n
    signal_reason = [''] * n
    signal_branch = [''] * n

    oversold_armed  = False
    cross_confirmed = False
    cross_bar       = -1
    CROSS_EXPIRY    = 10

    # ── FIX: loop goes up to n (inclusive of last/live bar) ──────────────
    for i in range(2, n):
        k_cur      = K.iloc[i];      d_cur      = D.iloc[i]
        k_prev     = K.iloc[i - 1];  d_prev     = D.iloc[i - 1]
        cci_cur    = CCI.iloc[i];    cci_prev   = CCI.iloc[i - 1]
        kv_cur     = k_vel.iloc[i]
        kv_prev    = k_vel_prev.iloc[i]
        curr_dt    = data.index[i]
        curr_close = close.iloc[i]

        # NaN guard — live bar may have NaN if indicator hasn't warmed up yet
        if any(pd.isna(v) for v in [k_cur, d_cur, k_prev, d_prev, cci_cur, cci_prev, kv_cur, kv_prev]):
            continue

        # ── RESET on overbought ──────────────────────────────────────────
        if k_cur > 80 or d_cur > 80:
            oversold_armed  = False
            cross_confirmed = False
            cross_bar       = -1
            continue

        # ── FILTER 1: arm when both K and D oversold ─────────────────────
        if k_cur < 20 and d_cur < 20:
            oversold_armed = True

        # ── Cross expiry ─────────────────────────────────────────────────
        if cross_confirmed and (i - cross_bar) > CROSS_EXPIRY:
            cross_confirmed = False
            cross_bar       = -1
            print(f"  ⏰ CROSS EXPIRED  @ {curr_dt}  (>{CROSS_EXPIRY} bars)")

        # ── FILTER 2: K crosses D (bullish cross) while below 30 ─────────
        if oversold_armed and not cross_confirmed:
            if (k_prev <= d_prev) and (k_cur > d_cur) and (k_cur < 30):
                cross_confirmed = True
                cross_bar       = i
                print(f"  🔄 K/D CROSS      @ {curr_dt}"
                      f"  K={k_cur:.2f}  D={d_cur:.2f}"
                      f"  spread={(k_cur - d_cur):.2f}")

        # ── FIRE PHASE ───────────────────────────────────────────────────
        if cross_confirmed and TIME_OK.iloc[i]:

            k_accelerating = (kv_cur > 0) and (kv_cur >= kv_prev)
            kd_spread      = k_cur - d_cur
            spread_ok      = kd_spread > 1.5
            cci_rising     = (cci_cur > cci_prev) and (cci_cur > -100)

            if structures_available:
                struct_ok, struct_note = _structure_nearby(i, curr_close)
            else:
                struct_ok   = True
                struct_note = "bypass(no_structures)"

            if not k_accelerating:
                if i == cross_bar + 1:
                    print(f"  ⛔ FILTER-3 REJECT @ {curr_dt}"
                          f"  Kvel={kv_cur:.2f} KvelPrev={kv_prev:.2f}")
            elif not spread_ok:
                if i == cross_bar + 1:
                    print(f"  ⛔ FILTER-4 REJECT @ {curr_dt}"
                          f"  KD_spread={kd_spread:.2f} (<1.5)")
            elif not cci_rising:
                if i == cross_bar + 1:
                    print(f"  ⛔ FILTER-5 REJECT @ {curr_dt}"
                          f"  CCI={cci_cur:.2f} CCIprev={cci_prev:.2f}")
            elif not struct_ok:
                if i == cross_bar + 1:
                    print(f"  ⛔ FILTER-6 REJECT @ {curr_dt}"
                          f"  No structure within 2% of close={curr_close:.2f}")

            if k_accelerating and spread_ok and cci_rising and struct_ok:
                grade = "★★★★★" if struct_ok and struct_note != "bypass(no_structures)" else "★★★★"

                b1_raw[i]        = 1
                signal_path[i]   = "STOCH_CONFLUENCE"
                signal_grade[i]  = grade
                signal_branch[i] = "BRANCH-1"
                signal_reason[i] = (
                    f"BRANCH-1 BUY: Stoch Confluence. "
                    f"K={k_cur:.1f} D={d_cur:.1f} spread={kd_spread:.1f} "
                    f"Kvel={kv_cur:.2f} KvelAcc={'✓' if k_accelerating else '✗'} "
                    f"CCI={cci_cur:.1f}↑ "
                    f"Structure={struct_note} "
                    f"[{grade}]"
                )
                print(f"  🟢 FIRED           @ {curr_dt}"
                      f"  K={k_cur:.2f}  D={d_cur:.2f}"
                      f"  spread={kd_spread:.2f}"
                      f"  Kvel={kv_cur:.2f}"
                      f"  CCI={cci_cur:.1f}"
                      f"  struct={struct_note}"
                      f"  [{grade}]")

                oversold_armed  = False
                cross_confirmed = False
                cross_bar       = -1

    b1_raw_s = pd.Series(b1_raw, index=data.index)

    # ── FIX: cooldown must NOT suppress the current (last/live) bar ──────
    #
    #  Original:  b1_recent = b1_raw_s.shift(1).rolling(3).sum()
    #             b1_final  = (b1_raw_s==1) & (b1_recent==0)
    #
    #  Problem:   In backtest this is fine — every bar has future bars to
    #             confirm.  In LIVE the last bar is the CURRENT bar.
    #             shift(1) pushes the live signal one bar into the future
    #             so b1_recent at position n-1 is 0 (no signal seen yet),
    #             which means the cooldown accidentally passes — BUT only
    #             if the signal fires on the LIVE bar itself.  The real bug
    #             is that the rolling window checks bars i-1, i-2, i-3
    #             (already past), not the current bar.  This is correct
    #             behaviour.  The ACTUAL live bug was the NaN skip above
    #             (indicators not warmed) and the tz mismatch in Branch-2.
    #             We keep the cooldown logic exactly as-is (no change to
    #             logic) but add a fillna(0) for safety on short series.
    b1_recent = b1_raw_s.shift(1).rolling(3).sum().fillna(0)
    b1_final  = ((b1_raw_s == 1) & (b1_recent == 0)).astype(int)

    print(f"\n  📊 BRANCH-1 SIGNALS: {int(b1_final.sum())} (after cooldown)")

    # ═══════════════════════════════════════════════════════
    #  BUILD REASON STRINGS
    # ═══════════════════════════════════════════════════════
    b2_reason = [''] * n
    for i in range(n):
        events_at_bar = b2_all_events_s.iloc[i]
        if events_at_bar:
            parts = [
                f"BRANCH-2 {ev['signal']} [{ev['source']}]: {ev['detail']}"
                for ev in events_at_bar
            ]
            b2_reason[i] = " | ".join(parts)
            if not signal_branch[i]:
                signal_branch[i] = "BRANCH-2"

    # ═══════════════════════════════════════════════════════
    #  STORE IN DATAFRAME
    # ═══════════════════════════════════════════════════════
    data['signal_path']   = signal_path
    data['signal_grade']  = signal_grade
    data['signal_branch'] = signal_branch

    data['b1_sig_raw'] = b1_raw
    data['b1_sig']     = b1_final.astype(int)

    data['b2_sig_raw']      = b2_raw.astype(int)
    data['b2_sig']          = b2_gated.astype(int)
    data['b2_signal_type']  = b2_signal_type_s
    data['b2_source_type']  = b2_source_type_s
    data['b2_detail']       = b2_detail_s
    data['b2_level']        = b2_level_s
    data['b2_reason']       = b2_reason
    data['b2_all_events']   = b2_all_events_s

    combined_reason = []
    for i in range(n):
        parts = []
        if b1_final.iloc[i] == 1 and signal_reason[i]:
            parts.append(signal_reason[i])
        if b2_reason[i]:
            parts.append(b2_reason[i])
        combined_reason.append(" | ".join(parts))
    data['signal_reason'] = combined_reason

    data['st_sig_raw'] = ((b1_raw_s == 1) | (b2_raw == 1)).astype(int)
    data['st_sig']     = ((b1_final == 1) | (b2_gated)).astype(int)
    data['confidence']    = 0.0
    data['llm_signal']    = None
    data['llm_reason']    = ""
    data['llm_entry']     = 0.0
    data['llm_stop_loss'] = 0.0
    data['llm_target_1']  = 0.0
    data['llm_target_2']  = 0.0
    data['llm_bias']      = ""

    # ═══════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ═══════════════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("  ✅ COMPLETE (v4 — LIVE MARKET FIXED)")
    print(f"  Branch-1 signals  : {int(b1_final.sum())}")
    print(f"  Branch-2 events   : {len(gated_events)}"
          f"  (BUY_CE={total_ce if option_type != 'PE' else 0}"
          f"  BUY_PE={total_pe if option_type != 'CE' else 0})")
    print(f"  Branch-2 bars hit : {int(b2_gated.sum())}"
          f"  (bars with ≥1 signal)")
    print(f"  Combined bars     : {int(data['st_sig'].sum())}")
    print("═" * 80 + "\n")

    return data















# for my market alerts
def tele_msg(msg):
    # Replace YOUR_BOT_TOKEN with your bot token obtained from BotFather
    bot_token1 = Telegram_token.telegram_token

    # Group Chat ID
    chat_id1 = Telegram_token.chat_id

    # Replace MESSAGE_TEXT with the text of the message you want to send
    message_text = " My Market Alerts Super Trend Strategy "+msg

    # Send the message using the sendMessage method of the Telegram Bot API
    url1 = f'https://api.telegram.org/bot{bot_token1}/sendMessage?chat_id={chat_id1}&text={message_text}'
    response = requests.get(url1)


Long_Trade_File = 'SuperTrend_Long.xlsx'
Short_Trade_File = 'SuperTrend_Short.xlsx'


# Create Excel Sheet only of it Needed or first time
def Long_create_excel_sheet(filename):
    if not os.path.exists(filename):
        columns = ['Symbol', 'Entry Time', 'Buy Price', 'Target Price','Sprice', 'Qty', 'Exit Time', 'Sell Price', 'Points',
                   'Brokerage', 'Profit/Loss', 'Trade Status']
        df = pd.DataFrame(columns=columns)
        df.to_excel(filename, index=False)
        print(f"{filename} created successfully!")
    else:
        print(f"{filename} already exists.")


Long_create_excel_sheet(Long_Trade_File)


# Create Excel Sheet only of it Needed or first time
def Short_create_excel_sheet(filename):
    if not os.path.exists(filename):
        columns = ['Symbol', 'Entry Time', 'Buy Price', 'Target Price','Sprice', 'Qty', 'Exit Time', 'Sell Price', 'Points',
                   'Brokerage', 'Profit/Loss', 'Trade Status']
        df = pd.DataFrame(columns=columns)
        df.to_excel(filename, index=False)
        print(f"{filename} created successfully!")
    else:
        print(f"{filename} already exists.")


Short_create_excel_sheet(Short_Trade_File)


def update_target_price(ticker, new_buy_price, tradefile):
    try:
        # Load the Excel file
        df = pd.read_excel(tradefile)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Find the row(s) where Symbol matches and Trade Status is OPEN
    index = df[(df['Symbol'] == ticker) & (df['Trade Status'] == 'OPEN')].index

    if not index.empty:
        idx = index[0]  # Only update the first open trade
        df.at[idx, 'Target_Price'] = new_buy_price

        # Save the updated DataFrame
        df.to_excel(tradefile, index=False)
        print(f"Buy Price updated for {ticker}")
    else:
        print(f"No open trade found for {ticker}")

def update_buy_price(ticker, new_buy_price, tradefile):
    try:
        # Load the Excel file
        df = pd.read_excel(tradefile)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Find the row(s) where Symbol matches and Trade Status is OPEN
    index = df[(df['Symbol'] == ticker) & (df['Trade Status'] == 'OPEN')].index

    if not index.empty:
        idx = index[0]  # Only update the first open trade
        df.at[idx, 'Sprice'] = new_buy_price

        # Save the updated DataFrame
        df.to_excel(tradefile, index=False)
        print(f"Buy Price updated for {ticker}")
    else:
        print(f"No open trade found for {ticker}")
# Update Long trade entry
def update_long_trades(ticker, entry_time, BuyPrice, target_price,Sprice ,qty, tradefile):
    workbook = load_workbook(filename=tradefile)
    worksheet = workbook.active

    new_row = [ticker, entry_time, BuyPrice, target_price,Sprice, qty, '', '', '', '', '', 'OPEN']
    worksheet.append(new_row)

    workbook.save(tradefile)


# Update Long trade entry
def update_Short_trades(ticker, entry_time, SellPrice, target_price,Sprice, qty, filename):
    workbook = load_workbook(filename)
    worksheet = workbook.active

    new_row = [ticker, entry_time, '', target_price,Sprice, qty, '', SellPrice, '', '', '', 'OPEN']
    worksheet.append(new_row)

    workbook.save(filename)


def all_trade_files():
    files = [Long_Trade_File, Short_Trade_File]
    merged_df = pd.concat([pd.read_excel(file) for file in files], ignore_index=True)
    merged_df.to_excel('All_Trades.xlsx', index=False)
    return 'All_Trades.xlsx saved successfully.'


# Call the function
all_trade_files()


# data = get_cash_market_data('INFY', '5m')
# super_trend(data)


# Symbol = 'INFY'
# Entry_Time = '2024-12-30 09:35:00'
# Buy_Price = 350
# Target_Price = 377
# Qty = 600
# Exit_Time = '2025-01-29 10:40:00'
# Sell_Price = 378
# Points = Sell_Price - Buy_Price
# Brokerage = (Buy_Price*Qty)+(Sell_Price*Qty) * 0.00015
# Profit_Loss = (Points * Qty) - Brokerage
# Trade_Status = 'Target Hit'

# update_long_trades(Symbol,Entry_Time,Buy_Price,Target_Price,Qty,Long_Trade_File)

# Function to close long trade
def close_long_trade(ticker, exit_time, sell_price, points, brokerage, profit_loss, trade_status, tradefile):
    df = pd.read_excel(tradefile)
    index = df[(df['Symbol'] == ticker) & (df['Trade Status'] == 'OPEN')].index

    if not index.empty:
        idx = index[0]
        df.at[idx, 'Exit Time'] = exit_time
        df.at[idx, 'Sell_Price'] = sell_price
        df.at[idx, 'Points'] = points
        df.at[idx, 'Brokerage'] = brokerage
        df.at[idx, 'Profit/Loss'] = profit_loss
        df.at[idx, 'Trade Status'] = trade_status

        df.to_excel(tradefile, index=False)
        print(f"Trade closed for {ticker}")
    else:
        print(f"No open trade found for {ticker}")


# Function to close long trade
def close_short_trade(ticker, exit_time, buy_price, points, brokerage, profit_loss, trade_status, tradefile):
    df = pd.read_excel(tradefile)
    index = df[(df['Symbol'] == ticker) & (df['Trade Status'] == 'OPEN')].index

    if not index.empty:
        idx = index[0]
        df.at[idx, 'Exit Time'] = exit_time
        df.at[idx, 'Buy_Price'] = buy_price
        df.at[idx, 'Points'] = points
        df.at[idx, 'Brokerage'] = brokerage
        df.at[idx, 'Profit/Loss'] = profit_loss
        df.at[idx, 'Trade Status'] = trade_status

        df.to_excel(tradefile, index=False)
        print(f"Trade closed for {ticker}")
    else:
        print(f"No open trade found for {ticker}")


# Define the required times
required_times =  [
    (9, 15), (9, 20), (9, 25), (9, 30), (9, 35), (9, 40), (9, 45), (9, 50), (9, 55),
    (10, 0), (10, 5), (10, 10), (10, 15), (10, 20), (10, 25), (10, 30), (10, 35), (10, 40), (10, 45), (10, 50), (10, 55),
    (11, 0), (11, 5), (11, 10), (11, 15), (11, 20), (11, 25), (11, 30), (11, 35), (11, 40), (11, 45), (11, 50), (11, 55),
    (12, 0), (12, 5), (12, 10), (12, 15), (12, 20), (12, 25), (12, 30), (12, 35), (12, 40), (12, 45), (12, 50), (12, 55),
    (13, 0), (13, 5), (13, 10), (13, 15), (13, 20), (13, 25), (13, 30), (13, 35), (13, 40), (13, 45), (13, 50), (13, 55),
    (14, 0), (14, 5), (14, 10), (14, 15), (14, 20), (14, 25), (14, 30), (14, 35), (14, 40), (14, 45), (14, 50), (14, 55),
    (15, 0), (15, 5), (15, 10), (15, 15), (15, 20), (15, 25), (15, 30)
]


# Define a function to check if the current time matches any of the required times
def is_required_time():
    current_time = dt.datetime.now(pytz.timezone('Asia/Kolkata')).time()
    return any(current_time.hour == hour and current_time.minute == minute for hour, minute in required_times)


# Initialize spot prices dictionary
spot_prices1 = {ticker: None for ticker in Tickers}

# Get instrument codes for tickers
ticker_codes = {ticker: str(instrument_df[instrument_df['Name'] == ticker]['ScripCode'].values[0]) for ticker in
                Tickers}

# Getting the List for Streaming Data
req_list = []
for s in Tickers:
    code = str(scripcode_lookup(instrument=instrument_df, symbol=s))
    req = {"Exch": "N", "ExchType": "D", "ScripCode": code}

    req_list.append(req)


# Define the callback function for incoming data
def on_message(ws, message):
    global spot_prices1
    data = json.loads(message)
    #print(data)
    if data:
        ticker_symbol = instrument_df[instrument_df['ScripCode'] == data[0]['Token']]['Name'].iloc[0]
        last_rate = data[0]['LastRate']
        spot_prices1[ticker_symbol] = last_rate

    # Subscribe to real-time data feed in a separate thread


# req_list = [{"Exch": "N", "ExchType": "C", "ScripCode": code} for code in ticker_codes.values()]
req_data = client.Request_Feed('mf', 's', req_list)
client.connect(req_data)


# function for Subscribing Data
def subscribe_data():
    client.receive_data(on_message)

import requests

def send_to_ui(symbol: str, price: float):
    try:
        # First, try to update the ticker
        res = requests.post("https://algotrading-ufvn.onrender.com/update-ticker", json={
            "symbol": symbol,
            "price": price
        })

        # If symbol is not found (not added yet), auto-add and retry
        if res.status_code == 404:
            print(f"⚠️ {symbol} not found in ticker list. Adding it now...")
            add_res = requests.post("http://localhost:8000/add-tickers", json={
                "tickers": [symbol]
            })
            if add_res.status_code == 200:
                print(f"✅ {symbol} added. Retrying update...")
                # Retry update
                retry_res = requests.post("http://localhost:8000/update-ticker", json={
                    "symbol": symbol,
                    "price": price
                })
                if retry_res.status_code == 200:
                    print(f"📈 Pushed to UI: {symbol} → ₹{price}")
                else:
                    print(f"❌ Failed to push after retry: {retry_res.status_code}")
            else:
                print(f"❌ Failed to auto-add {symbol}: {add_res.status_code}")

        elif res.status_code == 200:
            print(f"📈 Pushed to UI: {symbol} → ₹{price}")
        else:
            print(f"❌ Failed to push: {res.status_code}")
    except Exception as e:
        print(f"🔥 Error pushing to UI: {e}")
# Algo Start Here
start = dt.datetime.now(pytz.timezone('Asia/Kolkata'))
closetime = start.replace(hour=START_TIME[0], minute=START_TIME[1], second=START_TIME[2])
interval = (closetime-start).total_seconds()
if interval > 0:
    print('Algo will Run at ', START_TIME[0], ':', START_TIME[1], ':', START_TIME[2], ' Remaining Time Left = ',
          interval, ' sec')
    time.sleep(interval)
    print('Algo Starting Now!!!')

data_list = {}

for h in Tickers:

    data_fut = get_cash_market_data(h, '3m')
    data_fut.drop(data_fut.tail(1).index, inplace=True)

    super_trend(h,data_fut)

    data_list[h] = data_fut

super_Trend_Long = pd.read_excel(Long_Trade_File)
Long_Open_Position = super_Trend_Long[(super_Trend_Long['Trade Status'] == 'OPEN')]

super_Trend_Short = pd.read_excel(Short_Trade_File)
Short_Open_Position = super_Trend_Short[(super_Trend_Short['Trade Status'] == 'OPEN')]

# Adding Multi Thread Process to run Streaming data and Main algo at the same time
streaming_thread = threading.Thread(target=subscribe_data)
streaming_thread.daemon = True
streaming_thread.start()

time.sleep(5)

endTime = dt.datetime.now(pytz.timezone('Asia/Kolkata')).replace(hour=EXIT_TIME[0], minute=EXIT_TIME[1],
                                                                 second=EXIT_TIME[2])
while dt.datetime.now(pytz.timezone('Asia/Kolkata')) < endTime:

    try:

        for i in Tickers:

            print("###################################################################")
            print('Spot prices of', i, ' ', spot_prices1[i])
            print(dt.datetime.now(UTC).strftime("%d-%b-%Y %I:%M%p"))
            print("###################################################################")
            send_to_ui(i, spot_prices1[i])
            
            time.sleep(0.5)


            if is_required_time():
                
                try:
                    data_fut = get_cash_market_data(i, '5m')
                except Exception as e:
                    print(f"⚠️ Skipping {i}: {e}")
                    continue
                #data_fut = get_cash_market_data(i, '3m')
                data_fut.drop(data_fut.tail(1).index, inplace=True)
                super_trend(i,data_fut)

                data_list[i] = data_fut

                super_Trend_Long = pd.read_excel(Long_Trade_File)
                Long_Open_Position = super_Trend_Long[(super_Trend_Long['Trade Status'] == 'OPEN')]

                super_Trend_Short = pd.read_excel(Short_Trade_File)
                Short_Open_Position = super_Trend_Short[(super_Trend_Short['Trade Status'] == 'OPEN')]

                # Checking For SuperTrend Long
                # Checking For SuperTrend Long
                trail_sl=0
                if data_list[i]['st_sig'].iloc[-1] == 1:


                    all_trade_files()
                    open_trades_df = pd.read_excel('All_Trades.xlsx')
                    open_trades_df = open_trades_df[(open_trades_df['Trade Status'] == 'OPEN')]

                    open_trade_count = len(open_trades_df)

                    # Check the Maximum Position
                    if open_trade_count >= Max_Position:
                        print("Maximum Position is Reached.No New Position Will Take")

                        continue

                    # Check if ticker is already in Long_Open_Position
                    if i in Long_Open_Position['Symbol'].values:

                        print(f"{i} is already in Long Open Position. Skipping trade.")

                        continue

                    # Take the Long Trade
                    else:
                        trail_sl = 0

                        current_price = float(spot_prices1[i])

                        Trade_quantity =75# int(math.floor(Total_Cash_per_position / current_price))

                        Target_Price = current_price+Take_Profit
                        Sprice = current_price-10

                        # Sending Buy orders to the API

                        # For Real Money

                        # For Buy Order

                        # order_response = client.place_order(OrderType='B',Exchange='N',ExchangeType='C', ScripCode = int(scripcode_lookup(instrument=instrument_df, symbol= i), Qty=Trade_quantity, Price=current_price)

                        # # Check if the message is not 'Success'
                        # if order_response['Message'] != 'Success':
                        #     with open("error_log.txt", "a") as error_log_file:
                        #         error_log_file.write(ct +' - ' + order_response['Message'] +' '+i+ "\n")

                        #     tele_msg(order_response['Message']+' '+ i )

                        # position_df = pd.DataFrame(client.positions())

                        # BuyPrice = position_df[position_df.ScripName==i].BuyAvgRate.values[0]

                        entry_time = dt.datetime.now(UTC).strftime("%d-%b-%Y %I:%M%p")

                        BuyPrice = current_price

                        update_long_trades(i, entry_time, BuyPrice, Target_Price, Sprice, Trade_quantity,
                                           Long_Trade_File)

                        tele_msg("Long Entry Taken For "+i+" Total Quantity "+str(
                            Trade_quantity)+" and the BUY Price is "+str(BuyPrice)+"And the Target Price is "+str(Target_Price))

                        # After the new Entry We are Updating The Variables

                        super_Trend_Long = pd.read_excel(Long_Trade_File)

                        Long_Open_Position = super_Trend_Long[(super_Trend_Long['Trade Status'] == 'OPEN')]

                    # Stop Loss Check (Exit Long Trade)
                    # Checking Open Long Position
                if i in Long_Open_Position['Symbol'].values:

                    # Checking SuperTrend Signal Change
                    # Stop loss condition
                    if data_list[i]['st_sig'].iloc[-1] == -1:
                        print(f"Long Entry Stop Loss Hit for {i}. Closing position.")

                        # Fetch the Buy Price and Quantity
                        trade_row = Long_Open_Position[Long_Open_Position['Symbol'] == i]

                        BuyPrice = trade_row['Buy Price'].values[0]

                        Trade_quantity = 75  # int(trade_row['Qty'].values[0])

                        Exit_Time = dt.datetime.now(UTC).strftime("%d-%b-%Y %I:%M%p")

                        # For Real Money

                        # Sending Sell Order to The API

                        # For Sell Order

                        # order_response = client.place_order(OrderType='S',Exchange='N',ExchangeType='C', ScripCode = int(scripcode_lookup(instrument=instrument_df, symbol= i), Qty=Trade_quantity, Price=0)

                        # # Check if the message is not 'Success'
                        # if order_response['Message'] != 'Success':
                        #     with open("error_log.txt", "a") as error_log_file:
                        #         error_log_file.write(ct +' - ' + order_response['Message'] +' '+i+ "\n")

                        #     tele_msg(order_response['Message']+' '+i)

                        # position_df = pd.DataFrame(client.positions())

                        # Sell_Price = position_df[position_df.ScripName==i].SellAvgRate.values[0]

                        # For Paper Trade

                        Sell_Price = float(spot_prices1[i])  # Selling at current market price

                        Points = Sell_Price-BuyPrice

                        Brokerage = ((BuyPrice * Trade_quantity)+(Sell_Price * Trade_quantity)) * 0.00015

                        Profit_Loss = (Points * Trade_quantity)-Brokerage

                        Trade_Status = "Stop Loss Hit"

                        close_long_trade(i, Exit_Time, Sell_Price, Points, Brokerage, Profit_Loss, Trade_Status,
                                         Long_Trade_File)

                        tele_msg(f"Long Entry Stop Loss Hit for {i}. Exit Price: {Sell_Price}, P/L: {Profit_Loss}")

                    # Target Hit Check (Exit Long Trade)
                # Target Hit Check (Exit Long Trade)
                if i in Long_Open_Position['Symbol'].values:
                    print("Find the Target Price")
                    trade_row = Long_Open_Position[Long_Open_Position['Symbol'] == i]
                    BuyPrice = trade_row['Buy Price'].values[0]
                    Target_Price = trade_row['Target Price'].values[0]
                    Trail_Step = 10  # Move trailing target by 10 points
                    Trade_quantity = trade_row['Qty'].values[0]

                    current_price = spot_prices1[i]
                    profit_from_entry = current_price-BuyPrice

                    # Calculate how many steps of 10 points we've moved
                    step_count = int(profit_from_entry // Trail_Step)
                    new_trailing_target = BuyPrice+(step_count * Trail_Step)
                    

                    if profit_from_entry>10:
                        #BuyPrice=BuyPrice+10
                        update_buy_price(i,BuyPrice,Long_Trade_File)
                        print(f"Long Entry buy price trail for {BuyPrice}. Open position.")
                        #tele_msg(f"Long Entry buy price trail for {BuyPrice}.{profit_from_entry},{current_price},{current_price-BuyPrice} Open position.")
                        trail_sl=1


                    # Update the target if price moved significantly
                    if new_trailing_target > Target_Price:
                        Target_Price=new_trailing_target
                        Long_Open_Position.loc[Long_Open_Position['Symbol'] == i, 'Target Price'] = new_trailing_target
                        print(f"Updated Trailing Target for {i}: {new_trailing_target}")
                        tele_msg(f"Updated Trailing Target for {i}: {new_trailing_target}")
                        update_buy_price(i, new_trailing_target, Long_Trade_File)
                        update_target_price(i, new_trailing_target+10, Long_Trade_File)
                        continue  # Do not close trade yet; wait for next iteration

                    # Exit condition: current price exceeds latest target
                    if float(spot_prices1[i]) >= float(Target_Price):
                        print(f"Long Entry Target Hit for {i}. Closing position.")
                        # --- Exit logic as in your existing code ---
                        Exit_Time = dt.datetime.now(UTC).strftime("%d-%b-%Y %I:%M%p")
                        Sell_Price = float(current_price)
                        Points = Sell_Price-BuyPrice
                        Brokerage = ((BuyPrice * Trade_quantity)+(Sell_Price * Trade_quantity)) * 0.00015
                        Profit_Loss = (Points * Trade_quantity)-Brokerage
                        Trade_Status = "Target Hit"

                        close_long_trade(i, Exit_Time, Sell_Price, Points, Brokerage, Profit_Loss, Trade_Status,
                                         Long_Trade_File)
                        tele_msg(f"Long Entry Target Hit for {i}. Exit Price: {Sell_Price}, P/L: {Profit_Loss}")

                        # Refresh open positions after closing trade
                        super_Trend_Long = pd.read_excel(Long_Trade_File)
                        Long_Open_Position = super_Trend_Long[(super_Trend_Long['Trade Status'] == 'OPEN')]

                if i in Long_Open_Position['Symbol'].values:

                    # Find the SL Price
                    trade_row = Long_Open_Position[Long_Open_Position['Symbol'] == i]
                    S_Price = trade_row['Sprice'].values[0]
                    # S_Price = max(S_Price + (float(spot_prices1[i]) - trade_row['Buy Price']), S_Price)
                    print(S_Price)

                    # Check if current price exceeds target price
                    if float(spot_prices1[i]) <= float(S_Price):
                        print(f"Long Entry SL Hit for {i}. Closing position.")

                        # Fetch trade details
                        BuyPrice = trade_row['Buy Price'].values[0]
                        Trade_quantity = trade_row['Qty'].values[0]

                        # For Paper Trade
                        Exit_Time = dt.datetime.now(UTC).strftime("%d-%b-%Y %I:%M%p")
                        Sell_Price = float(spot_prices1[i])  # Selling at market price
                        Points = Sell_Price-BuyPrice
                        Brokerage = ((BuyPrice * Trade_quantity)+(Sell_Price * Trade_quantity)) * 0.00015
                        Profit_Loss = (Points * Trade_quantity)-Brokerage
                        Trade_Status = "SL Hit"

                        close_long_trade(i, Exit_Time, Sell_Price, Points, Brokerage, Profit_Loss, Trade_Status,
                                         Long_Trade_File)

                        tele_msg(f"Long Entry SL Hit for {i}. Exit Price: {Sell_Price}, P/L: {Profit_Loss}")

                        # Refresh open positions after closing trade
                        super_Trend_Long = pd.read_excel(Long_Trade_File)
                        Long_Open_Position = super_Trend_Long[(super_Trend_Long['Trade Status'] == 'OPEN')]

                    # Exit 5 Min Before the Exit Time
                if i in Long_Open_Position['Symbol'].values:

                    # Check the Exit Time and The Exit Condition Exit 5 Min Before the Exit Time

                    if dt.datetime.now(pytz.timezone('Asia/Kolkata')) > dt.datetime.now(
                            pytz.timezone('Asia/Kolkata')).replace(hour=EXIT_TIME[0], minute=EXIT_TIME[1]-5,
                                                                   second=EXIT_TIME[2]):
                        print(f"Long Entry Exit Time Out for {i}. Closing position.")

                        # Fetch trade details
                        BuyPrice = trade_row['Buy Price'].values[0]
                        Trade_quantity = trade_row['Qty'].values[0]

                        # For Real Money

                        # Sending Sell Order to The API

                        # For Sell Order

                        # order_response = client.place_order(OrderType='S',Exchange='N',ExchangeType='C', ScripCode = int(scripcode_lookup(instrument=instrument_df, symbol= i), Qty=Trade_quantity, Price=0)

                        # # Check if the message is not 'Success'
                        # if order_response['Message'] != 'Success':
                        #     with open("error_log.txt", "a") as error_log_file:
                        #         error_log_file.write(ct +' - ' + order_response['Message'] +' '+i+ "\n")

                        #     tele_msg(order_response['Message']+' '+i)

                        # position_df = pd.DataFrame(client.positions())

                        # Sell_Price = position_df[position_df.ScripName==i].SellAvgRate.values[0]

                        # For Paper Trade
                        Exit_Time = dt.datetime.now(UTC).strftime("%d-%b-%Y %I:%M%p")
                        Sell_Price = float(spot_prices1[i])  # Selling at market price
                        Points = Sell_Price-BuyPrice
                        Brokerage = ((BuyPrice * Trade_quantity)+(Sell_Price * Trade_quantity)) * 0.00015
                        Profit_Loss = (Points * Trade_quantity)-Brokerage
                        Trade_Status = "Exit Time Out"

                        close_long_trade(i, Exit_Time, Sell_Price, Points, Brokerage, Profit_Loss, Trade_Status,
                                         Long_Trade_File)

                        tele_msg(f"Long Entry Exit Time Out for {i}. Exit Price: {Sell_Price}, P/L: {Profit_Loss}")

                        # Refresh open positions after closing trade
                        super_Trend_Long = pd.read_excel(Long_Trade_File)
                        Long_Open_Position = super_Trend_Long[(super_Trend_Long['Trade Status'] == 'OPEN')]

                        continue

                # Checking For Short Position
                if data_list[i]['st_sig'].iloc[-1] == -1:

                    all_trade_files()
                    open_trades_df = pd.read_excel('All_Trades.xlsx')
                    open_trades_df = open_trades_df[(open_trades_df['Trade Status'] == 'OPEN')]

                    open_trade_count = len(open_trades_df)

                    # Check the Maximum Position
                    if open_trade_count >= Max_Position:
                        print("Maximum Position is Reached.No New Position Will Take")

                        continue

                    # Check if ticker is already in Long_Open_Position
                    if i in Short_Open_Position['Symbol'].values:

                        print(f"{i} is already in Long Open Position. Skipping trade.")

                        continue

                    # Take the Long Trade
                    else:

                        current_price = float(spot_prices1[i])

                        Trade_quantity = int(math.floor(Total_Cash_per_position / current_price))

                        Target_Price = current_price-(current_price * Take_Profit)

                        # Sending Sell orders to the API

                        # For Real Money

                        # For Sell Order

                        # order_response = client.place_order(OrderType='S',Exchange='N',ExchangeType='C', ScripCode = int(scripcode_lookup(instrument=instrument_df, symbol= i), Qty=Trade_quantity, Price=current_price)

                        # # Check if the message is not 'Success'
                        # if order_response['Message'] != 'Success':
                        #     with open("error_log.txt", "a") as error_log_file:
                        #         error_log_file.write(ct +' - ' + order_response['Message'] +' '+i+ "\n")

                        #     tele_msg(order_response['Message']+' '+ i )

                        # position_df = pd.DataFrame(client.positions())

                        # SellPrice = position_df[position_df.ScripName==i].SellAvgRate.values[0]

                        entry_time = dt.datetime.now(UTC).strftime("%d-%b-%Y %I:%M%p")

                        SellPrice = current_price
                        Sprice = current_price

                        update_Short_trades(i, entry_time, SellPrice, Target_Price, Sprice, Trade_quantity, Short_Trade_File)

                        tele_msg("Short Entry Taken For "+i+" Total Quantity "+str(
                            Trade_quantity)+" And the Target Price is "+str(Target_Price))

                        # After the new Entry We are Updating The Variables

                        super_Trend_Short = pd.read_excel(Short_Trade_File)

                        Short_Open_Position = super_Trend_Short[(super_Trend_Short['Trade Status'] == 'OPEN')]

                if i in Short_Open_Position['Symbol'].values:

                    # Checking SuperTrend Signal Change
                    # Stop loss condition
                    if data_list[i]['st_sig'].iloc[-1] == 1:
                        print(f"Short Entry Stop Loss Hit for {i}. Closing position.")

                        # Fetch the Buy Price and Quantity
                        trade_row = Short_Open_Position[Short_Open_Position['Symbol'] == i]

                        SellPrice = trade_row['Sell Price'].values[0]

                        Trade_quantity = int(trade_row['Qty'].values[0])

                        Exit_Time = dt.datetime.now().strftime("%d-%b-%Y %I:%M%p")

                        # For Real Money

                        # Sending Buy Order to The API

                        # For Buy Order

                        # order_response = client.place_order(OrderType='B',Exchange='N',ExchangeType='C', ScripCode = int(scripcode_lookup(instrument=instrument_df, symbol= i), Qty=Trade_quantity, Price=0)

                        # # Check if the message is not 'Success'
                        # if order_response['Message'] != 'Success':
                        #     with open("error_log.txt", "a") as error_log_file:
                        #         error_log_file.write(ct +' - ' + order_response['Message'] +' '+i+ "\n")

                        #     tele_msg(order_response['Message']+' '+i)

                        # position_df = pd.DataFrame(client.positions())

                        # Buy_Price = position_df[position_df.ScripName==i].BuyAvgRate.values[0]

                        # For Paper Trade

                        Buy_Price = float(spot_prices1[i])  # Selling at current market price

                        Points = SellPrice-Buy_Price

                        Brokerage = ((Buy_Price * Trade_quantity)+(SellPrice * Trade_quantity)) * 0.00015

                        Profit_Loss = (Points * Trade_quantity)-Brokerage

                        Trade_Status = "Stop Loss Hit"

                        close_short_trade(i, Exit_Time, Buy_Price, Points, Brokerage, Profit_Loss, Trade_Status,
                                          Short_Trade_File)

                        tele_msg(f"Short Entry Stop Loss Hit for {i}. Exit Price: {Sell_Price}, P/L: {Profit_Loss}")

                        super_Trend_Short = pd.read_excel(Short_Trade_File)

                        Short_Open_Position = super_Trend_Short[(super_Trend_Short['Trade Status'] == 'OPEN')]

                # Target Hit Check (Exit Long Trade)
                if i in Short_Open_Position['Symbol'].values:

                    # Find the Target Price
                    trade_row = Short_Open_Position[Short_Open_Position['Symbol'] == i]
                    Target_Price = trade_row['Target Price'].values[0]

                    # Check if current price exceeds target price
                    if spot_prices1[i] > Target_Price:
                        print(f"Short Entry Target Hit for {i}. Closing position.")

                        # Fetch the Buy Price and Quantity
                        trade_row = Short_Open_Position[Short_Open_Position['Symbol'] == i]

                        SellPrice = trade_row['Sell Price'].values[0]

                        Trade_quantity = int(trade_row['Qty'].values[0])

                        Exit_Time = dt.datetime.now(UTC).strftime("%d-%b-%Y %I:%M%p")

                        # For Real Money

                        # Sending Buy Order to The API

                        # For Buy Order

                        # order_response = client.place_order(OrderType='B',Exchange='N',ExchangeType='C', ScripCode = int(scripcode_lookup(instrument=instrument_df, symbol= i), Qty=Trade_quantity, Price=0)

                        # # Check if the message is not 'Success'
                        # if order_response['Message'] != 'Success':
                        #     with open("error_log.txt", "a") as error_log_file:
                        #         error_log_file.write(ct +' - ' + order_response['Message'] +' '+i+ "\n")

                        #     tele_msg(order_response['Message']+' '+i)

                        # position_df = pd.DataFrame(client.positions())

                        # Buy_Price = position_df[position_df.ScripName==i].BuyAvgRate.values[0]

                        # For Paper Trade

                        Buy_Price = float(spot_prices1[i])  # Selling at current market price

                        Points = SellPrice-Buy_Price

                        Brokerage = ((Buy_Price * Trade_quantity)+(SellPrice * Trade_quantity)) * 0.00015

                        Profit_Loss = (Points * Trade_quantity)-Brokerage

                        Trade_Status = "Target Hit"

                        tele_msg(f"Short Entry Target Hit for {i}. Exit Price: {Sell_Price}, P/L: {Profit_Loss}")

                        close_short_trade(i, Exit_Time, Buy_Price, Points, Brokerage, Profit_Loss, Trade_Status,
                                          Short_Trade_File)

                        super_Trend_Short = pd.read_excel(Short_Trade_File)

                        Short_Open_Position = super_Trend_Short[(super_Trend_Short['Trade Status'] == 'OPEN')]

                # Exit 5 Min Before the Exit Time
                if i in Long_Open_Position['Symbol'].values:

                    # Check the Exit Time and The Exit Condition Exit 5 Min Before the Exit Time

                    if dt.datetime.now(pytz.timezone('Asia/Kolkata')) > dt.datetime.now(
                            pytz.timezone('Asia/Kolkata')).replace(hour=EXIT_TIME[0], minute=EXIT_TIME[1],
                                                                   second=EXIT_TIME[2]):
                        print(f"Short Entry Exit Time Out for {i}. Closing position.")

                        # Fetch the Buy Price and Quantity
                        trade_row = Short_Open_Position[Short_Open_Position['Symbol'] == i]

                        SellPrice = trade_row['Sell Price'].values[0]

                        Trade_quantity = int(trade_row['Qty'].values[0])

                        Exit_Time = dt.datetime.now(UTC).strftime("%d-%b-%Y %I:%M%p")

                        # For Real Money

                        # Sending Buy Order to The API

                        # For Buy Order

                        # order_response = client.place_order(OrderType='B',Exchange='N',ExchangeType='C', ScripCode = int(scripcode_lookup(instrument=instrument_df, symbol= i), Qty=Trade_quantity, Price=0)

                        # # Check if the message is not 'Success'
                        # if order_response['Message'] != 'Success':
                        #     with open("error_log.txt", "a") as error_log_file:
                        #         error_log_file.write(ct +' - ' + order_response['Message'] +' '+i+ "\n")

                        #     tele_msg(order_response['Message']+' '+i)

                        # position_df = pd.DataFrame(client.positions())

                        # Buy_Price = position_df[position_df.ScripName==i].BuyAvgRate.values[0]

                        # For Paper Trade

                        Buy_Price = float(spot_prices1[i])  # Selling at current market price

                        Points = SellPrice-Buy_Price

                        Brokerage = ((Buy_Price * Trade_quantity)+(SellPrice * Trade_quantity)) * 0.00015

                        Profit_Loss = (Points * Trade_quantity)-Brokerage

                        Trade_Status = "Exit Time Out"

                        tele_msg(f"Short Entry Exit Time Out for {i}. Exit Price: {Sell_Price}, P/L: {Profit_Loss}")

                        close_short_trade(i, Exit_Time, Buy_Price, Points, Brokerage, Profit_Loss, Trade_Status,
                                          Short_Trade_File)

                        super_Trend_Short = pd.read_excel(Short_Trade_File)

                        Short_Open_Position = super_Trend_Short[(super_Trend_Short['Trade Status'] == 'OPEN')]

                        continue


    except Exception as e:
        print("Error:", e)
        print("Oops!", e.__class__, "occurred.")

        ct = dt.datetime.now().strftime("%d-%b-%Y %I:%M%p")
        error_message = f"{ct} - An error occurred: {e}"
        tele_msg(error_message)

        with open("error_log.txt", "a") as error_log_file:
            error_log_file.write(error_message+"\n")
        raise ValueError("I have raised an Exception in main")
