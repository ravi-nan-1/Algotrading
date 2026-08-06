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

from collections import deque, Counter

try:
    import pandas_ta as ta
except ImportError as e:
    raise ImportError(
        "pandas_ta is required for the multi-indicator divergence engine "
        "(pip install pandas_ta)"
    ) from e

try:
    from scipy.signal import argrelextrema
except ImportError as e:
    raise ImportError(
        "scipy is required for the standalone find_swing_levels() utility "
        "(pip install scipy)"
    ) from e


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

def super_trend(symbol: str, data: pd.DataFrame, confirm_max_wait: int = 4,
                index_data: pd.DataFrame = None,
                multi_div_pivot_period: int = 5,
                multi_div_min_count: int = 1,
                multi_div_maxpp: int = 10,
                multi_div_maxbars: int = 100,
                multi_div_search: str = "Regular",
                multi_div_showlast: bool = False,
                multi_div_align_tolerance: "pd.Timedelta|None" = None,
                multi_div_start_time: dt.time = dt.time(10, 0),
                multi_div_stoch_oversold: float = 20.0,
                multi_div_stoch_overbought: float = 80.0,
                multi_div_level_lookback: int = 8,
                multi_div_level_maxwait: int = 8,
                multi_div_swing_order: int = 5,
                multi_div_level_tolerance: float = 8.0,
                multi_div_breakout_buffer_atr_mult: float = 0.15,
                multi_div_breakout_buffer_points: float = 0.0,
                multi_div_enable_base_breakout: bool = True,
                multi_div_base_min_candles: int = 3,
                multi_div_base_max_candles: int = 8,
                multi_div_base_range_pct: float = 0.0035,
                multi_div_base_ema_lag: int = 2,
                multi_div_min_pivot_atr_mult: float = 0.0,
                multi_div_level_min_move_points: float = 0.0,
                multi_div_level_min_move_pct: float = 0.0,
                continuation_level_min_move_points: float = 0.0,
                continuation_level_min_move_pct: float = 0.0,
                multi_div_level_max_distance: float = 25.0) -> pd.DataFrame:
    

    # ------------------------------------------------------------------
    # Nested helpers
    # ------------------------------------------------------------------
    def _md_calculate_indicators(df):
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        df['rsi'] = ta.rsi(close, length=14)

        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        df['macd'] = macd_df.iloc[:, 0]
        df['deltamacd'] = macd_df.iloc[:, 1]

        df['moment'] = close.diff(10)
        df['cci'] = ta.cci(high, low, close, length=10)
        df['obv'] = ta.obv(close, volume)

        stoch_df = ta.stoch(high, low, close, k=14, d=3, smooth_k=1)
        df['stk'] = stoch_df.iloc[:, 0].rolling(3).mean()

        vwma_fast = (close * volume).rolling(12).sum() / volume.rolling(12).sum()
        vwma_slow = (close * volume).rolling(26).sum() / volume.rolling(26).sum()
        df['vwmacd'] = vwma_fast - vwma_slow

        hl = (high - low).replace(0, np.nan)
        cmfm = ((close - low) - (high - close)) / hl
        cmfv = cmfm * volume
        df['cmf'] = cmfv.rolling(21).sum() / volume.rolling(21).sum()

        df['mfi'] = ta.mfi(high, low, close, volume, length=14)

        return df

    def _md_pine_pivothigh(arr, i, prd):
        if i < 2 * prd:
            return None
        center = i - prd
        window = arr[i - 2 * prd: i + 1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) == 0:
            return None
        center_val = arr[center]
        if np.isnan(center_val):
            return None
        if center_val == np.max(valid_window):
            return center_val
        return None

    def _md_pine_pivotlow(arr, i, prd):
        if i < 2 * prd:
            return None
        center = i - prd
        window = arr[i - 2 * prd: i + 1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) == 0:
            return None
        center_val = arr[center]
        if np.isnan(center_val):
            return None
        if center_val == np.min(valid_window):
            return center_val
        return None

    def _md_nz(val, default=0.0):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return val

    def _md_get_back(arr, i, n):
        idx = i - n
        if idx < 0 or idx >= len(arr):
            return np.nan
        return arr[idx]

    def _md_positive_regular_positive_hidden_divergence(
            i, src, close, prsc, pl_positions, pl_vals,
            prd, maxpp, maxbars, dontconfirm, cond
    ):
        divlen = 0

        src_0 = _md_get_back(src, i, 0)
        src_1 = _md_get_back(src, i, 1)
        close_0 = _md_get_back(close, i, 0)
        close_1 = _md_get_back(close, i, 1)

        gate = (
                dontconfirm or
                (not np.isnan(src_0) and not np.isnan(src_1) and src_0 > src_1) or
                (not np.isnan(close_0) and not np.isnan(close_1) and close_0 > close_1)
        )
        if not gate:
            return divlen

        startpoint = 0 if dontconfirm else 1
        n_pivots = len(pl_positions)

        for x in range(maxpp):
            if x >= n_pivots:
                break
            pl_pos = pl_positions[x]
            pl_val = pl_vals[x]
            if pl_pos == 0:
                break

            length = i - pl_pos + prd
            if length > maxbars:
                break
            if length <= 5:
                continue

            src_sp = _md_get_back(src, i, startpoint)
            src_len = _md_get_back(src, i, length)
            prsc_sp = _md_get_back(prsc, i, startpoint)

            if np.isnan(src_sp) or np.isnan(src_len) or np.isnan(prsc_sp):
                continue

            cond1_check = (cond == 1 and src_sp > src_len and prsc_sp < _md_nz(pl_val))
            cond2_check = (cond == 2 and src_sp < src_len and prsc_sp > _md_nz(pl_val))
            if not (cond1_check or cond2_check):
                continue

            close_sp = _md_get_back(close, i, startpoint)
            close_len = _md_get_back(close, i, length)
            if np.isnan(close_sp):
                continue

            span = length - startpoint
            if span == 0:
                continue

            slope1 = (src_sp - src_len) / span
            virtual_line1 = src_sp - slope1
            slope2 = (close_sp - _md_nz(close_len)) / span
            virtual_line2 = close_sp - slope2

            arrived = True
            for y in range(1 + startpoint, length):
                src_y = _md_get_back(src, i, y)
                close_y = _md_get_back(close, i, y)
                if np.isnan(src_y):
                    arrived = False
                    break
                close_y_nz = _md_nz(close_y)
                if src_y < virtual_line1 or close_y_nz < virtual_line2:
                    arrived = False
                    break
                virtual_line1 -= slope1
                virtual_line2 -= slope2

            if arrived:
                divlen = length
                break

        return divlen

    def _md_negative_regular_negative_hidden_divergence(
            i, src, close, prsc, ph_positions, ph_vals,
            prd, maxpp, maxbars, dontconfirm, cond
    ):
        divlen = 0

        src_0 = _md_get_back(src, i, 0)
        src_1 = _md_get_back(src, i, 1)
        close_0 = _md_get_back(close, i, 0)
        close_1 = _md_get_back(close, i, 1)

        gate = (
                dontconfirm or
                (not np.isnan(src_0) and not np.isnan(src_1) and src_0 < src_1) or
                (not np.isnan(close_0) and not np.isnan(close_1) and close_0 < close_1)
        )
        if not gate:
            return divlen

        startpoint = 0 if dontconfirm else 1
        n_pivots = len(ph_positions)

        for x in range(maxpp):
            if x >= n_pivots:
                break
            ph_pos = ph_positions[x]
            ph_val = ph_vals[x]
            if ph_pos == 0:
                break

            length = i - ph_pos + prd
            if length > maxbars:
                break
            if length <= 5:
                continue

            src_sp = _md_get_back(src, i, startpoint)
            src_len = _md_get_back(src, i, length)
            prsc_sp = _md_get_back(prsc, i, startpoint)

            if np.isnan(src_sp) or np.isnan(src_len) or np.isnan(prsc_sp):
                continue

            cond1_check = (cond == 1 and src_sp < src_len and prsc_sp > _md_nz(ph_val))
            cond2_check = (cond == 2 and src_sp > src_len and prsc_sp < _md_nz(ph_val))
            if not (cond1_check or cond2_check):
                continue

            close_sp = _md_get_back(close, i, startpoint)
            close_len = _md_get_back(close, i, length)
            if np.isnan(close_sp):
                continue

            span = length - startpoint
            if span == 0:
                continue

            slope1 = (src_sp - src_len) / span
            virtual_line1 = src_sp - slope1
            slope2 = (close_sp - _md_nz(close_len)) / span
            virtual_line2 = close_sp - slope2

            arrived = True
            for y in range(1 + startpoint, length):
                src_y = _md_get_back(src, i, y)
                close_y = _md_get_back(close, i, y)
                if np.isnan(src_y):
                    arrived = False
                    break
                close_y_nz = _md_nz(close_y)
                if src_y > virtual_line1 or close_y_nz > virtual_line2:
                    arrived = False
                    break
                virtual_line1 -= slope1
                virtual_line2 -= slope2

            if arrived:
                divlen = length
                break

        return divlen

    def calculate_index_divergences(
            index_df, prd=5, source="Close", searchdiv="Regular", showlimit=3,
            maxpp=10, maxbars=100, dontconfirm=False, showlast=False,
            calcmacd=True, calcmacda=True, calcrsi=True, calcstoc=True,
            calccci=True, calcmom=True, calcobv=True, calcvwmacd=True,
            calccmf=True, calcmfi=True, verbose=True,
            min_pivot_atr_mult: float = 0.0,
    ):
        df = index_df.copy()
        if 'Datetime' in df.columns:
            df = df.set_index('Datetime')
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        df = _md_calculate_indicators(df)

        close = df['Close'].values.astype(float)
        high = df['High'].values.astype(float)
        low = df['Low'].values.astype(float)

        _hl = df['High'] - df['Low']
        _hc = (df['High'] - df['Close'].shift()).abs()
        _lc = (df['Low'] - df['Close'].shift()).abs()
        _atr_arr = pd.concat([_hl, _hc, _lc], axis=1).max(axis=1).rolling(14).mean().values

        def _md_pivot_is_significant(val, opp_positions, opp_vals):
            if min_pivot_atr_mult <= 0:
                return True
            if len(opp_positions) == 0 or opp_positions[0] == 0:
                return True
            atr_now = _atr_arr[i] if i < len(_atr_arr) else np.nan
            if np.isnan(atr_now) or atr_now == 0:
                return True
            amplitude = abs(val - opp_vals[0])
            return amplitude >= (atr_now * min_pivot_atr_mult)

        ph_source = close if source == "Close" else high
        pl_source = close if source == "Close" else low
        prsc_pos = close if source == "Close" else low
        prsc_neg = close if source == "Close" else high

        ind_list = [
            ("MACD", calcmacd, df['macd'].values.astype(float)),
            ("Hist", calcmacda, df['deltamacd'].values.astype(float)),
            ("RSI", calcrsi, df['rsi'].values.astype(float)),
            ("Stoch", calcstoc, df['stk'].values.astype(float)),
            ("CCI", calccci, df['cci'].values.astype(float)),
            ("MOM", calcmom, df['moment'].values.astype(float)),
            ("OBV", calcobv, df['obv'].values.astype(float)),
            ("VWMACD", calcvwmacd, df['vwmacd'].values.astype(float)),
            ("CMF", calccmf, df['cmf'].values.astype(float)),
            ("MFI", calcmfi, df['mfi'].values.astype(float)),
        ]

        do_regular = searchdiv in ("Regular", "Regular/Hidden")
        do_hidden = searchdiv in ("Hidden", "Regular/Hidden")

        MAXARR = 20
        ph_positions = deque([0] * MAXARR, maxlen=MAXARR)
        ph_vals = deque([0.0] * MAXARR, maxlen=MAXARR)
        pl_positions = deque([0] * MAXARR, maxlen=MAXARR)
        pl_vals = deque([0.0] * MAXARR, maxlen=MAXARR)

        n_ = len(df)

        remove_last_pos_divs = False
        remove_last_neg_divs = False

        pos_label_history = []
        neg_label_history = []

        for i in range(n_):
            ph = _md_pine_pivothigh(ph_source, i, prd)
            pl = _md_pine_pivotlow(pl_source, i, prd)

            if ph is not None:
                ph_positions.appendleft(i)
                ph_vals.appendleft(ph)
            if pl is not None:
                pl_positions.appendleft(i)
                pl_vals.appendleft(pl)

            if pl is not None:
                remove_last_pos_divs = False
            if ph is not None:
                remove_last_neg_divs = False

            if i < 2 * prd:
                continue

            pos_reg = neg_reg = pos_hid = neg_hid = 0
            active_names_pos = []
            active_names_neg = []

            for name, enabled, indicator in ind_list:
                if not enabled:
                    continue

                d = [0, 0, 0, 0]

                if do_regular:
                    d[0] = _md_positive_regular_positive_hidden_divergence(
                        i, indicator, close, prsc_pos, pl_positions, pl_vals,
                        prd, maxpp, maxbars, dontconfirm, cond=1
                    )
                    d[1] = _md_negative_regular_negative_hidden_divergence(
                        i, indicator, close, prsc_neg, ph_positions, ph_vals,
                        prd, maxpp, maxbars, dontconfirm, cond=1
                    )
                if do_hidden:
                    d[2] = _md_positive_regular_positive_hidden_divergence(
                        i, indicator, close, prsc_pos, pl_positions, pl_vals,
                        prd, maxpp, maxbars, dontconfirm, cond=2
                    )
                    d[3] = _md_negative_regular_negative_hidden_divergence(
                        i, indicator, close, prsc_neg, ph_positions, ph_vals,
                        prd, maxpp, maxbars, dontconfirm, cond=2
                    )

                if d[0]:
                    pos_reg += 1
                    active_names_pos.append(name)
                if d[1]:
                    neg_reg += 1
                    active_names_neg.append(name)
                if d[2]:
                    pos_hid += 1
                    if name not in active_names_pos:
                        active_names_pos.append(name)
                if d[3]:
                    neg_hid += 1
                    if name not in active_names_neg:
                        active_names_neg.append(name)

            total = pos_reg + neg_reg + pos_hid + neg_hid
            if total < showlimit:
                continue

            has_pos = (pos_reg > 0 or pos_hid > 0)
            has_neg = (neg_reg > 0 or neg_hid > 0)

            if has_pos:
                if showlast:
                    pos_label_history.clear()
                else:
                    if remove_last_pos_divs and pos_label_history:
                        pos_label_history.pop()

                pos_label_history.append({
                    'bar': i, 'Datetime': df.index[i], 'Close': df['Close'].iloc[i],
                    'pos_reg_div': pos_reg, 'pos_hid_div': pos_hid,
                    'neg_reg_div': 0, 'neg_hid_div': 0,
                    'indicators': ', '.join(active_names_pos), 'type': 'bottom'
                })
                remove_last_pos_divs = True

            if has_neg:
                if showlast:
                    neg_label_history.clear()
                else:
                    if remove_last_neg_divs and neg_label_history:
                        neg_label_history.pop()

                neg_label_history.append({
                    'bar': i, 'Datetime': df.index[i], 'Close': df['Close'].iloc[i],
                    'pos_reg_div': 0, 'pos_hid_div': 0,
                    'neg_reg_div': neg_reg, 'neg_hid_div': neg_hid,
                    'indicators': ', '.join(active_names_neg), 'type': 'top'
                })
                remove_last_neg_divs = True

        all_final = pos_label_history + neg_label_history
        all_final.sort(key=lambda x: x['bar'])

        if verbose:
            print("-" * 100)
            print(f"🔎 MULTI-INDICATOR DIVERGENCE ENGINE (INDEX data) -- surviving labels: {len(all_final)}")
            print(f"   Bullish (bottom) labels : {len(pos_label_history)}")
            print(f"   Bearish (top) labels    : {len(neg_label_history)}")
            for entry in all_final:
                side = "BULLISH" if entry['type'] == 'bottom' else "BEARISH"
                cnt = entry['pos_reg_div'] + entry['pos_hid_div'] if entry['type'] == 'bottom' \
                    else entry['neg_reg_div'] + entry['neg_hid_div']
                print(f"   [{side:7s}] {entry['Datetime']}  close={entry['Close']:.2f}  "
                      f"agree={cnt}  indicators=({entry['indicators']})")
            print("-" * 100)

        if not all_final:
            return pd.DataFrame(columns=['Close', 'pos_reg_div', 'neg_reg_div', 'pos_hid_div',
                                         'neg_hid_div', 'total_divergences',
                                         'divergence_indicators', 'label_type'])

        out_rows = []
        for entry in all_final:
            out_rows.append({
                'Datetime': entry['Datetime'],
                'Close': entry['Close'],
                'pos_reg_div': entry['pos_reg_div'],
                'neg_reg_div': entry['neg_reg_div'],
                'pos_hid_div': entry['pos_hid_div'],
                'neg_hid_div': entry['neg_hid_div'],
                'total_divergences': (entry['pos_reg_div'] + entry['neg_reg_div'] +
                                      entry['pos_hid_div'] + entry['neg_hid_div']),
                'divergence_indicators': entry['indicators'],
                'label_type': entry['type'],
            })

        out = pd.DataFrame(out_rows).set_index('Datetime')
        return out

    def _md_cluster_levels(prices, tol):
        """Groups nearby swing prices into horizontal levels."""
        levels = []
        for price in sorted(prices):
            found = False
            for i, (level, count, touches) in enumerate(levels):
                if abs(price - level) <= tol:
                    levels[i] = (level, count + 1, touches + [price])
                    found = True
                    break
            if not found:
                levels.append((price, 1, [price]))
        result = []
        for level, count, touches in levels:
            avg_level = round(sum(touches) / len(touches), 2)
            result.append({
                'Level': avg_level,
                'Touches': count,
                'Min': min(touches),
                'Max': max(touches)
            })
        if not result:
            return pd.DataFrame(columns=['Level', 'Touches', 'Min', 'Max'])
        return pd.DataFrame(result).sort_values('Level').reset_index(drop=True)

    def _zigzag_moves(swing_highs_pos, swing_lows_pos):
        pivots = [(pos, 'H', val) for pos, val in swing_highs_pos.items()]
        pivots += [(pos, 'L', val) for pos, val in swing_lows_pos.items()]
        pivots.sort(key=lambda x: x[0])

        move_at_high = {}
        move_at_low = {}
        last_opposite = {}

        for pos, kind, val in pivots:
            if kind == 'H':
                prev_low = last_opposite.get('L')
                if prev_low is not None:
                    move_at_high[pos] = val - prev_low
                last_opposite['H'] = val
            else:
                prev_high = last_opposite.get('H')
                if prev_high is not None:
                    move_at_low[pos] = prev_high - val
                last_opposite['L'] = val

        return move_at_high, move_at_low

    def _md_cluster_levels_with_moves(prices_with_moves, tol):
        levels = []
        for price, move in sorted(prices_with_moves, key=lambda t: t[0]):
            found = False
            for i, (level, count, touches, moves) in enumerate(levels):
                if abs(price - level) <= tol:
                    levels[i] = (level, count + 1, touches + [price], moves + [move])
                    found = True
                    break
            if not found:
                levels.append((price, 1, [price], [move]))

        result = []
        for level, count, touches, moves in levels:
            avg_level = round(sum(touches) / len(touches), 2)
            valid_moves = [m for m in moves if m is not None]
            result.append({
                'Level': avg_level,
                'Touches': count,
                'Min': min(touches),
                'Max': max(touches),
                'MaxMove': round(max(valid_moves), 2) if valid_moves else np.nan,
                'AvgMove': round(sum(valid_moves) / len(valid_moves), 2) if valid_moves else np.nan,
                'MinMove': round(min(valid_moves), 2) if valid_moves else np.nan,
            })
        if not result:
            return pd.DataFrame(columns=['Level', 'Touches', 'Min', 'Max', 'MaxMove', 'AvgMove', 'MinMove'])
        return pd.DataFrame(result).sort_values('Level').reset_index(drop=True)

    def find_swing_levels(df_, order=5, tolerance=8.0, silent=False,
                          min_move_points: float = 0.0,
                          min_move_pct: float = 0.0):
        data_ = df_.copy()
        if not isinstance(data_.index, pd.DatetimeIndex):
            data_.index = pd.to_datetime(data_.index)
        data_ = data_[~data_.index.duplicated(keep='last')].sort_index()

        high_idx = argrelextrema(data_['High'].values, np.greater_equal, order=order)[0]
        low_idx = argrelextrema(data_['Low'].values, np.less_equal, order=order)[0]
        swing_highs = data_.iloc[high_idx]['High'].copy()
        swing_lows = data_.iloc[low_idx]['Low'].copy()

        swing_highs_pos = pd.Series(swing_highs.values, index=high_idx)
        swing_lows_pos = pd.Series(swing_lows.values, index=low_idx)

        move_at_high, move_at_low = _zigzag_moves(swing_highs_pos, swing_lows_pos)

        def _passes_move_filter(pos, val, move_map):
            if min_move_points <= 0 and min_move_pct <= 0:
                return True
            mv = move_map.get(pos)
            if mv is None:
                return False
            if min_move_points > 0 and mv < min_move_points:
                return False
            if min_move_pct > 0 and val > 0 and (mv / val * 100) < min_move_pct:
                return False
            return True

        high_prices_with_moves = [
            (val, move_at_high.get(pos))
            for pos, val in swing_highs_pos.items()
            if _passes_move_filter(pos, val, move_at_high)
        ]
        low_prices_with_moves = [
            (val, move_at_low.get(pos))
            for pos, val in swing_lows_pos.items()
            if _passes_move_filter(pos, val, move_at_low)
        ]

        resistance = _md_cluster_levels_with_moves(high_prices_with_moves, tolerance)
        support = _md_cluster_levels_with_moves(low_prices_with_moves, tolerance)

        return resistance, support, swing_highs, swing_lows

    # ------------------------------------------------------------------
    # Standard setup
    # ------------------------------------------------------------------
    if 'Datetime' in data.columns:
        data = data.set_index('Datetime')
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    if data.index.tz is not None:
        print(f"⚠️  Option data Datetime index is tz-aware ({data.index.tz}) -- stripping tz "
              f"to avoid silent merge_asof mismatches against index_data.")
        data.index = data.index.tz_localize(None)
    data.index.name = 'Datetime'

    data = data.copy()
    n = len(data)

    _symbol_tokens = symbol.upper().split()
    if 'CE' in _symbol_tokens and 'PE' not in _symbol_tokens:
        multi_div_side = 'CE'
    elif 'PE' in _symbol_tokens and 'CE' not in _symbol_tokens:
        multi_div_side = 'PE'
    else:
        multi_div_side = 'BOTH'
        print(f"⚠️  Could not determine CE/PE from symbol '{symbol}' -- multi-indicator "
              f"divergence side-filtering is DISABLED for this run (both bullish and "
              f"bearish labels will be queued, same as before).")

    print("=" * 100)
    print(f"🚀 ENHANCED RANGE-BOUND STRATEGY → {symbol} | Data Points: {n}")
    print("=" * 100)

    data['st_sig'] = 0
    data['condition'] = ''
    data['quality'] = ''
    data['entry_price'] = np.nan
    data['stop_loss'] = np.nan
    data['take_profit'] = np.nan
    data['risk_reward'] = np.nan
    data['confidence'] = 0.0
    data['reason'] = ''
    data['signal_type'] = ''
    data['rating_score'] = np.nan

    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(14).mean()
    data['ATR_MA'] = data['ATR'].rolling(20).mean()

    data['EMA5'] = data['Close'].ewm(span=6, adjust=False).mean()
    data['EMA10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA15'] = data['Close'].ewm(span=15, adjust=False).mean()
    data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA30'] = data['Close'].ewm(span=30, adjust=False).mean()
    data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()

    data['range'] = data['High'] - data['Low']
    data['body'] = (data['Close'] - data['Open']).abs()
    data['body_pct'] = data['body'] / (data['range'].replace(0, np.nan))
    data['close_pos'] = (data['Close'] - data['Low']) / (data['range'].replace(0, np.nan))
    data['is_bullish'] = data['Close'] > data['Open']

    data['vol_ma20'] = data['Volume'].rolling(20).mean()
    data['vol_ratio'] = data['Volume'] / (data['vol_ma20'] + 1e-9)

    data['hlc3'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['ema_trend'] = np.where(data['hlc3'] >= data['EMA5'], 1, -1)

    def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    data['RSI'] = _wilder_rsi(data['Close'], 14)

    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_hist'] = data['MACD'] - data['MACD_signal']

    def _stochastic_k(high, low, close, k_period=14, smooth_k=3):
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        return raw_k.rolling(smooth_k).mean()

    data['STOCH_K'] = _stochastic_k(data['High'], data['Low'], data['Close'], k_period=14, smooth_k=3)

    def psar(high, low, close, iaf=0.02, maxaf=0.2):
        length = len(high)
        sar = np.zeros(length)
        trend = np.zeros(length, dtype=int)
        af = np.full(length, iaf)
        ep = np.zeros(length)
        sar[0] = low[0]
        trend[0] = 1
        ep[0] = high[0]
        for i in range(1, length):
            if trend[i - 1] == 1:
                sar[i] = min(sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1]), low[i - 1])
                if low[i] < sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i - 1]
                    ep[i] = low[i]
                    af[i] = iaf
                else:
                    trend[i] = 1
                    if high[i] > ep[i - 1]:
                        ep[i] = high[i]
                        af[i] = min(af[i - 1] + iaf, maxaf)
                    else:
                        ep[i] = ep[i - 1]
                        af[i] = af[i - 1]
            else:
                sar[i] = max(sar[i - 1] - af[i - 1] * (sar[i - 1] - ep[i - 1]), high[i - 1])
                if high[i] > sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i - 1]
                    ep[i] = high[i]
                    af[i] = iaf
                else:
                    trend[i] = -1
                    if low[i] < ep[i - 1]:
                        ep[i] = low[i]
                        af[i] = min(af[i - 1] + iaf, maxaf)
                    else:
                        ep[i] = ep[i - 1]
                        af[i] = af[i - 1]
        return sar, trend

    data['PSAR'], data['PSAR_trend'] = psar(
        data['High'].values, data['Low'].values, data['Close'].values
    )

    def is_range_bound(idx, lookback=12, threshold_pct=0.85):
        if idx < max(lookback, 20):
            return False
        window = data.iloc[idx - lookback:idx + 1]
        c = data.iloc[idx]
        prev = data.iloc[idx - 1] if idx > 0 else c
        price_range_pct = (window['High'].max() - window['Low'].min()) / window['Close'].iloc[-1] * 100
        psar_window = data['PSAR_trend'].iloc[idx - lookback:idx + 1]
        psar_flips = (psar_window.diff() != 0).sum()
        no_breakout = c['High'] <= prev['High'] * 1.002
        recent_bodies = data['body_pct'].iloc[max(0, idx - 7):idx]
        small_body_condition = (recent_bodies < 0.48).mean() >= 0.65
        is_range = (
                price_range_pct < threshold_pct or
                (psar_flips >= 4 and no_breakout and small_body_condition)
        )
        return is_range

    def is_choppy_market(idx, lookback=10):
        if idx < 25:
            return False
        c = data.iloc[idx]
        atr_now = data['ATR'].iloc[idx]
        if pd.isna(atr_now) or atr_now == 0:
            return False
        window = data.iloc[idx - lookback + 1:idx + 1]
        checks = []
        ema_slope = abs(data['EMA5'].iloc[idx] - data['EMA5'].iloc[idx - 5])
        checks.append(ema_slope < 0.08 * atr_now)
        range_points = window['High'].max() - window['Low'].min()
        checks.append(range_points < 1.8 * atr_now)
        overlap_count = 0
        for j in range(idx - lookback + 1, idx + 1):
            cj = data.iloc[j]
            pj = data.iloc[j - 1]
            if cj['High'] <= pj['High'] and cj['Low'] >= pj['Low']:
                overlap_count += 1
        checks.append((overlap_count / lookback) >= 0.70)
        atr_ma_now = data['ATR_MA'].iloc[idx]
        checks.append((not pd.isna(atr_ma_now)) and atr_now < atr_ma_now)
        psar_distance = abs(c['Close'] - c['PSAR'])
        checks.append(psar_distance < 0.3 * atr_now)
        recent_high = data['High'].iloc[idx - lookback:idx].max()
        checks.append(c['Close'] < recent_high * 1.002)
        return sum(checks) >= 3

    data['consolidation'] = False
    for i in range(20, n):
        win = data.iloc[i - 20:i]
        rng = (win['High'].max() - win['Low'].min()) / win['Close'].iloc[-1]
        data.loc[data.index[i], 'consolidation'] = rng < 0.019

    base_morning_dates_fired = set()

    # ====================== INDEX-DRIVEN MULTI-INDICATOR DIVERGENCE ======================
    use_multi_div = index_data is not None
    div_labels = None
    if use_multi_div:
        div_labels = calculate_index_divergences(
            index_data,
            prd=multi_div_pivot_period,
            searchdiv=multi_div_search,
            showlimit=multi_div_min_count,
            maxpp=multi_div_maxpp,
            maxbars=multi_div_maxbars,
            showlast=multi_div_showlast,
            min_pivot_atr_mult=multi_div_min_pivot_atr_mult,
            verbose=False,
        )
    else:
        print("⚠️  index_data not supplied -- MULTI_INDICATOR_DIVERGENCE branch is DISABLED "
              "for this run. Every other branch still runs normally on the option data.")

    # ---- Align index candles onto the option-data timeline ----
    use_index_filter = index_data is not None
    idx_df_raw = None

    if use_index_filter:
        idx_df = index_data.copy()
        if 'Datetime' in idx_df.columns:
            idx_df = idx_df.set_index('Datetime')
        if not isinstance(idx_df.index, pd.DatetimeIndex):
            idx_df.index = pd.to_datetime(idx_df.index)
        if idx_df.index.tz is not None:
            print(f"⚠️  index_data Datetime index is tz-aware ({idx_df.index.tz}) -- stripping tz "
                  f"to avoid silent merge_asof mismatches against option data.")
            idx_df.index = idx_df.index.tz_localize(None)
        idx_df.index.name = 'Datetime'
        idx_df = idx_df.sort_index()

        idx_df_raw = idx_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

        idx_df['STOCH_K'] = _stochastic_k(
            idx_df['High'], idx_df['Low'], idx_df['Close'], k_period=14, smooth_k=3
        )

        _idx_hl = idx_df['High'] - idx_df['Low']
        _idx_hc = np.abs(idx_df['High'] - idx_df['Close'].shift())
        _idx_lc = np.abs(idx_df['Low'] - idx_df['Close'].shift())
        idx_df['ATR'] = pd.concat([_idx_hl, _idx_hc, _idx_lc], axis=1).max(axis=1).rolling(14).mean()

        idx_df['EMA5'] = idx_df['Close'].ewm(span=6, adjust=False).mean()
        idx_df['bar_pos'] = np.arange(len(idx_df))

        idx_df = idx_df.sort_index()[['Open', 'High', 'Low', 'Close', 'Volume',
                                       'STOCH_K', 'ATR', 'EMA5', 'bar_pos']].add_prefix('idx_')

        if len(idx_df) > 1:
            median_interval = pd.Series(idx_df.index).diff().dropna().median()
        else:
            median_interval = pd.Timedelta(minutes=1)
        if pd.isna(median_interval) or median_interval <= pd.Timedelta(0):
            median_interval = pd.Timedelta(minutes=1)
        align_tol = multi_div_align_tolerance if multi_div_align_tolerance is not None \
            else 2 * median_interval

        opt_start, opt_end = data.index.min(), data.index.max()
        idx_start, idx_end = idx_df.index.min(), idx_df.index.max()
        overlap_start = max(opt_start, idx_start)
        overlap_end = min(opt_end, idx_end)
        print(f"🔗 Aligning index_data onto option timeline | option range: {opt_start} → {opt_end} | "
              f"index range: {idx_start} → {idx_end} | overlap: "
              f"{overlap_start} → "
              f"{overlap_end if overlap_start <= overlap_end else 'NONE (no date overlap!)'} "
              f"| tolerance: {align_tol}")

        data = data.sort_index()
        data_reset = data.reset_index()
        idx_reset = idx_df.reset_index()
        data_reset = pd.merge_asof(data_reset, idx_reset, on='Datetime', direction='backward')
        matched_ohlc = data_reset['idx_Close'].notna().sum()
        data = data_reset.set_index('Datetime')
        print(f"   -> {matched_ohlc}/{len(data)} option candles matched to an index OHLC row.")
    else:
        print("⚠️  index_data not supplied -- MULTI_INDICATOR_DIVERGENCE branch is DISABLED "
              "for this run. Every other branch still runs normally on the option data.")
        for col in ['idx_STOCH_K', 'idx_ATR', 'idx_EMA5', 'idx_Close',
                    'idx_High', 'idx_Low', 'idx_Open', 'idx_bar_pos']:
            data[col] = np.nan

    def _nearest_level_above(levels_df, price):
        if levels_df is None or levels_df.empty:
            return None
        above = levels_df[levels_df['Level'] > price].sort_values('Level')
        if above.empty:
            return None
        return above.iloc[0]

    def _nearest_level_below(levels_df, price):
        if levels_df is None or levels_df.empty:
            return None
        below = levels_df[levels_df['Level'] < price].sort_values('Level', ascending=False)
        if below.empty:
            return None
        return below.iloc[0]

    def _swing_levels_asof(current_bar_pos):
        """
        Return (resistance_df, support_df) using only index bars up to
        current_bar_pos. Non-repainting -- no future candles leak in.
        """
        empty = pd.DataFrame(columns=['Level', 'Touches', 'Min', 'Max'])
        if idx_df_raw is None or pd.isna(current_bar_pos):
            return empty, empty
        cp = int(current_bar_pos)
        min_bars = 2 * multi_div_swing_order + 1
        if cp + 1 < min_bars:
            return empty, empty
        safe_cp = cp - multi_div_swing_order
        if safe_cp < 2 * multi_div_swing_order:
            return empty, empty
        causal_slice = idx_df_raw.iloc[:safe_cp + 1]
        resistance_df, support_df, _, _ = find_swing_levels(
            causal_slice,
            order=multi_div_swing_order,
            tolerance=multi_div_level_tolerance,
            silent=True,
            min_move_points=multi_div_level_min_move_points,
            min_move_pct=multi_div_level_min_move_pct,
        )
        return resistance_df, support_df

    def _continuation_swing_levels_asof(idx, order=5, tolerance=8.0):
        """Returns (resistance_df, support_df, reference_price, source_tag)."""
        empty = pd.DataFrame(columns=['Level', 'Touches', 'Min', 'Max'])

        if use_index_filter and idx_df_raw is not None:
            current_bar_pos = data['idx_bar_pos'].iloc[idx] if 'idx_bar_pos' in data.columns else np.nan
            ref_price = data['idx_Close'].iloc[idx] if 'idx_Close' in data.columns else np.nan
            if pd.isna(current_bar_pos) or pd.isna(ref_price):
                return empty, empty, np.nan, 'index'
            cp = int(current_bar_pos)
            safe_cp = cp - order
            if safe_cp < 2 * order:
                return empty, empty, np.nan, 'index'
            causal_slice = idx_df_raw.iloc[:safe_cp + 1]
            resistance_df, support_df, _, _ = find_swing_levels(
                causal_slice, order=order, tolerance=tolerance, silent=True,
                min_move_points=continuation_level_min_move_points,
                min_move_pct=continuation_level_min_move_pct,
            )
            return resistance_df, support_df, ref_price, 'index'
        else:
            safe_cp = idx - order
            ref_price = data['Close'].iloc[idx]
            if safe_cp < 2 * order:
                return empty, empty, ref_price, 'option'
            causal_slice = data[['Open', 'High', 'Low', 'Close']].iloc[:safe_cp + 1]
            resistance_df, support_df, _, _ = find_swing_levels(
                causal_slice, order=order, tolerance=tolerance, silent=True,
                min_move_points=continuation_level_min_move_points,
                min_move_pct=continuation_level_min_move_pct,
            )
            return resistance_df, support_df, ref_price, 'option'

    # ------------------------------------------------------------------
    # Divergence label alignment
    # ------------------------------------------------------------------
    if div_labels is not None and not div_labels.empty and use_index_filter:
        dl = div_labels.sort_index().reset_index()
        dl = dl.rename(columns={'Datetime': '_div_label_time'})
        dl = dl[['_div_label_time', 'label_type', 'divergence_indicators',
                 'pos_reg_div', 'neg_reg_div', 'pos_hid_div', 'neg_hid_div']]
        dl_sorted = dl.sort_values('_div_label_time')
        data_reset = data.reset_index()
        data_reset = pd.merge_asof(
            data_reset.sort_values('Datetime'), dl_sorted,
            left_on='Datetime', right_on='_div_label_time',
            direction='backward', tolerance=align_tol,
        )
        data = data_reset.set_index('Datetime')
        matched_labels = data['label_type'].notna().sum()
        print(f"   -> {matched_labels}/{len(div_labels)} divergence labels matched onto an "
              f"option candle within tolerance.")
        if matched_labels == 0 and len(div_labels) > 0:
            print("   ⚠️  ZERO labels matched despite the engine finding "
                  f"{len(div_labels)} of them. Likely causes: (a) index_data and option "
                  "data timestamps don't actually overlap in date range -- check the "
                  "alignment line printed above, (b) the two feeds' timestamps are offset "
                  "by more than the tolerance shown above -- pass a wider "
                  "multi_div_align_tolerance, or (c) a residual tz/format mismatch. This is "
                  "not a divergence-math bug -- the engine's own label count above is correct.")
    else:
        data['label_type'] = np.nan
        data['divergence_indicators'] = np.nan
        data['_div_label_time'] = pd.NaT

    # ====================== DETECTORS ======================

    def detect_v_shape(idx):
        if idx < 14 or is_range_bound(idx, lookback=14, threshold_pct=0.95) or is_choppy_market(idx):
            return False, {}

        lookback = 9
        window = data.iloc[idx - lookback:idx + 1]
        low_idx_rel = window['Low'].values.argmin()
        pivot_idx = idx - lookback + low_idx_rel
        bars_since_pivot = idx - pivot_idx

        MAX_BARS_SINCE_PIVOT = 2
        if bars_since_pivot < 0 or bars_since_pivot > MAX_BARS_SINCE_PIVOT:
            return False, {}

        pivot_low = data['Low'].iloc[pivot_idx]

        local_min = data['Low'].iloc[max(0, pivot_idx - 6):pivot_idx + 7].min()
        if pivot_low > local_min:
            return False, {}

        left_start = max(0, pivot_idx - 8)
        left_window = data.iloc[left_start:pivot_idx + 1]
        swing_high = left_window['High'].max()
        decline_pct = (swing_high - pivot_low) / swing_high * 100 if swing_high > 0 else 0
        points_drop = swing_high - pivot_low

        if not (points_drop >= 25 and decline_pct >= 1.2):
            return False, {}

        def is_ladder_down(start_idx, end_idx):
            streak = 0
            prev_low = None
            prev_high = None
            for j in range(start_idx, end_idx + 1):
                cj = data.iloc[j]
                if cj['is_bullish']:
                    prev_low = None
                    prev_high = None
                    streak = 0
                    continue
                if cj['body_pct'] < 0.35:
                    prev_low = None
                    prev_high = None
                    streak = 0
                    continue
                if prev_low is not None:
                    if cj['High'] > prev_low + 0.10:
                        prev_low = cj['Low']
                        prev_high = cj['High']
                        streak = 1
                        continue
                prev_low = cj['Low']
                prev_high = cj['High']
                streak += 1
            return streak

        ladder_streak = is_ladder_down(left_start, pivot_idx)
        if ladder_streak < 2:
            return False, {}

        if 'EMA5' in data.columns:
            ema5_below_found = any(
                data.iloc[j]['Low'] < data['EMA5'].iloc[j]
                for j in range(left_start, pivot_idx + 1)
                if not pd.isna(data['EMA5'].iloc[j])
            )
            if not ema5_below_found:
                return False, {}

        c = data.iloc[idx]
        prev = data.iloc[idx - 1]
        candle_range = c['High'] - c['Low']
        if candle_range == 0:
            return False, {}

        lower_wick = min(c['Open'], c['Close']) - c['Low']
        lower_wick_pct = lower_wick / candle_range
        upper_wick_pct = (c['High'] - max(c['Open'], c['Close'])) / candle_range

        def is_ladder_up(start_idx, end_idx):
            streak = 0
            prev_high = None
            for j in range(start_idx, end_idx + 1):
                cj = data.iloc[j]
                if not cj['is_bullish']:
                    prev_high = None
                    streak = 0
                    continue
                if cj['body_pct'] < 0.35:
                    prev_high = None
                    streak = 0
                    continue
                if prev_high is not None:
                    if cj['Low'] < prev_high - 0.10:
                        prev_high = cj['High']
                        streak = 1
                        continue
                prev_high = cj['High']
                streak += 1
            return streak

        right_start = pivot_idx + 1
        if right_start <= idx:
            right_ladder_streak = is_ladder_up(right_start, idx)
        else:
            right_ladder_streak = 0

        if right_ladder_streak < 1:
            return False, {}

        is_prev_hammer_red = False
        if idx > 0:
            p_range = prev['High'] - prev['Low']
            if p_range > 0 and not prev['is_bullish']:
                p_lower = min(prev['Open'], prev['Close']) - prev['Low']
                if p_lower / p_range >= 0.55:
                    is_prev_hammer_red = True

        pattern_type = None

        if (c['is_bullish'] and
                c['body_pct'] >= 0.60 and
                c['close_pos'] >= 0.65 and
                c['Close'] > prev['High'] and
                c['Low'] >= prev['Low'] - 0.10):
            pattern_type = 'GREEN'

        elif (lower_wick_pct >= 0.60 and
              upper_wick_pct <= 0.20 and
              c['close_pos'] >= 0.40 and
              c['High'] <= prev['High'] + 0.10):
            pattern_type = 'WICK'

        elif (is_prev_hammer_red and
              c['is_bullish'] and
              c['body_pct'] >= 0.50 and
              c['Close'] > prev['Close'] and
              c['Low'] >= prev['Low'] - 0.10):
            pattern_type = 'GREEN_AFTER_WICK'

        if pattern_type is None and bars_since_pivot >= 2:
            box_start = pivot_idx + 1
            box_end = idx - 1
            if box_end - box_start + 1 >= 2:
                box_window = data.iloc[box_start:box_end + 1]
                box_high = box_window['High'].max()
                box_low = box_window['Low'].min()
                box_range = box_high - box_low
                decline_size = swing_high - pivot_low

                box_is_tight = decline_size > 0 and box_range <= 0.35 * decline_size
                no_downside_overlap = c['Low'] >= box_low - 0.10

                box_breakout_confirmed = (
                        c['Close'] > box_high and
                        c['is_bullish'] and
                        c['body_pct'] >= 0.50 and
                        no_downside_overlap
                )
                if box_is_tight and box_breakout_confirmed:
                    pattern_type = 'BOX_BREAKOUT'

        if pattern_type is None:
            return False, {}

        recovery_pct = (c['Close'] - pivot_low) / (swing_high - pivot_low + 1e-9)

        min_recovery = 0.18 if 'WICK' in pattern_type else 0.28
        if recovery_pct < min_recovery:
            return False, {}

        recent_high = (swing_high if pivot_idx == idx
                       else data['High'].iloc[pivot_idx:idx].max())

        breakout_price = c['High'] if 'WICK' in pattern_type else c['Close']

        if breakout_price < recent_high * 0.990:
            return False, {}

        if c['vol_ratio'] < 1.10:
            return False, {}

        if 'EMA5' in data.columns:
            ema5_now = data['EMA5'].iloc[idx]
            ema5_prev = data['EMA5'].iloc[idx - 1]
            if not pd.isna(ema5_now) and not pd.isna(ema5_prev):
                if not (ema5_now >= ema5_prev or c['Close'] > ema5_now):
                    return False, {}

        print(
            f"[VSHAPE] {data.index[idx]} >>> SIGNAL: pattern={pattern_type} | "
            f"left_ladder={ladder_streak} | right_ladder={right_ladder_streak} | "
            f"decline_pct={decline_pct:.2f} | recovery_pct={recovery_pct * 100:.2f} | "
            f"vol={c['vol_ratio'] * 100:.1f} | bars_since_pivot={bars_since_pivot}"
        )

        return True, {
            'pattern_type': pattern_type,
            'decline_pct': round(decline_pct, 2),
            'recovery_pct': round(recovery_pct * 100, 2),
            'lower_wick_pct': round(lower_wick_pct * 100, 2),
            'left_ladder_streak': ladder_streak,
            'right_ladder_streak': right_ladder_streak,
            'bars_since_pivot': bars_since_pivot,
            'vol': round(c['vol_ratio'] * 100, 1),
        }

    def detect_base(idx):
        if idx < 10:
            return False, {}

        if is_range_bound(idx, lookback=10, threshold_pct=0.90) or is_choppy_market(idx):
            return False, {}

        tm = data.index[idx].time()

        if not (dt.time(9, 30) <= tm <= dt.time(10, 0)):
            return False, {}

        current_date = data.index[idx].date()

        if current_date in base_morning_dates_fired:
            return False, {}

        if idx < 2:
            return False, {}

        red = data.iloc[idx - 2]
        green1 = data.iloc[idx - 1]
        green2 = data.iloc[idx]

        ema5_red = data["EMA5"].iloc[idx - 2]

        is_red = red["Close"] < red["Open"]
        red_below_ema = (red["Open"] < ema5_red) and (red["Close"] < ema5_red)

        if not (is_red and red_below_ema):
            return False, {}

        is_green1 = green1["Close"] > green1["Open"]
        if not is_green1:
            return False, {}

        is_green2 = green2["Close"] > green2["Open"]
        if not is_green2:
            return False, {}

        if green2["Close"] > red["Open"]:
            return True, {
                "pattern_type": "BASE_MORNING",
                "strength": "HIGH",
                "red_open": round(red["Open"], 2),
                "green2_close": round(green2["Close"], 2),
            }
        else:
            return False, {}

    continuation_state = {}
    CONTINUATION_LEVEL_COOLDOWN = 1

    def detect_continuation(idx, swing_order=5, level_tolerance=8.0,
                            max_level_distance_atr_mult=2.5,
                            level_cooldown=CONTINUATION_LEVEL_COOLDOWN):
        if idx < 30 or is_choppy_market(idx):
            return False, {}

        c = data.iloc[idx]
        prev = data.iloc[idx - 1]
        ts = data.index[idx]

        # SETUP 1: EMA5 fall + reversal
        if idx >= 2 and multi_div_side in ('CE', 'BOTH'):
            g1, g2 = prev, c
            if g1['is_bullish'] and g2['is_bullish']:
                red = data.iloc[idx - 2]
                ema5_red = data['EMA5'].iloc[idx - 2]
                red_below_ema5 = (
                        (not red['is_bullish']) and not pd.isna(ema5_red) and
                        red['Open'] < ema5_red and red['Close'] < ema5_red
                )
                if red_below_ema5:
                    broke_red_high = g2['Close'] > red['High']
                    vol_ok = g2['vol_ratio'] >= 1.0

                    if broke_red_high and vol_ok:

                        return True, {
                            'side': 'CE',
                            'pattern_type': 'CONTINUATION_EMA5_FALL_REVERSAL',
                            'level': round(red['High'], 2),
                            'level_source': 'red_candle_high',
                            'red_candle_time': str(data.index[idx - 2]),
                            'vol': round(g2['vol_ratio'] * 100, 1),
                        }

        # SETUP 2: swing-level breakout (CE only)
        resistance_df, _support_unused, ref_price, source = _continuation_swing_levels_asof(
            idx, order=swing_order, tolerance=level_tolerance
        )
        if resistance_df.empty or pd.isna(ref_price):
            return False, {}

        nearest_res = _nearest_level_above(resistance_df, ref_price)
        if nearest_res is None:
            return False, {}

        level = float(nearest_res['Level'])
        touches = int(nearest_res['Touches'])
        level_bucket = round(level / level_tolerance) * level_tolerance

        atr_now = (data['idx_ATR'].iloc[idx] if source == 'index' and 'idx_ATR' in data.columns
                   else data['ATR'].iloc[idx])
        if pd.isna(atr_now) or atr_now == 0:
            return False, {}
        max_dist = atr_now * max_level_distance_atr_mult

        ema15_now = data['EMA15'].iloc[idx]
        ema15_prev = data['EMA15'].iloc[idx - 1]

        if multi_div_side in ('CE', 'BOTH'):
            dist_before = level - ref_price
            if 0 < dist_before <= max_dist:
                two_green_above_ema15 = (
                        c['is_bullish'] and prev['is_bullish'] and
                        c['Close'] > ema15_now and prev['Close'] > ema15_prev
                )
                broke_level = ref_price > level
                vol_ok = c['vol_ratio'] >= 1.0

                if two_green_above_ema15 and broke_level and vol_ok:

                    continuation_state.setdefault('fired_levels_ce', {})[level_bucket] = idx
                    return True, {
                        'side': 'CE',
                        'pattern_type': 'CONTINUATION_SWING_BREAKOUT',
                        'level': round(level, 2),
                        'touches': touches,
                        'level_source': source,
                        'dist_before_break': round(dist_before, 2),
                        'vol': round(c['vol_ratio'] * 100, 1),
                    }

        return False, {}

    def detect_box_consolidation_breakout(idx, box_lookback=12, min_candles_in_box=8):
        if idx < box_lookback + 5 or is_choppy_market(idx):
            return False, {}
        atr_now = data['ATR'].iloc[idx]
        if pd.isna(atr_now) or atr_now == 0:
            return False, {}
        box_window = data.iloc[idx - box_lookback:idx]
        box_high = box_window['High'].max()
        box_low = box_window['Low'].min()
        box_range = box_high - box_low
        pre_box_start = idx - box_lookback
        pre_box_lookback = box_lookback * 2
        pre_start_idx = max(0, pre_box_start - pre_box_lookback)
        pre_window = data.iloc[pre_start_idx:pre_box_start]
        if len(pre_window) < 3:
            return False, {}
        pre_high = pre_window['High'].max()
        pre_low = pre_window['Low'].min()
        prior_fall = pre_high - pre_low
        high_loc = pre_window['High'].idxmax()
        low_loc = pre_window['Low'].idxmin()
        if not (high_loc < low_loc and prior_fall > 30):
            return False, {}
        if box_range > 2.5 * atr_now:
            return False, {}
        small_body_ratio = (box_window['body_pct'] < 0.30).mean()
        if small_body_ratio < 0.30:
            return False, {}
        overlap_count = 0
        for j in range(idx - box_lookback + 1, idx):
            cj = data.iloc[j]
            pj = data.iloc[j - 1]
            if cj['High'] <= pj['High'] * 1.0015 and cj['Low'] >= pj['Low'] * 0.9985:
                overlap_count += 1
        if (overlap_count / (box_lookback - 1)) < 0.25:
            return False, {}
        c = data.iloc[idx]
        breakout_confirmed = (c['Close'] > box_high and c['is_bullish']
                              and c['body_pct'] >= 0.20 and c['close_pos'] >= 0.25)
        if not breakout_confirmed:
            return False, {}
        if c['vol_ratio'] < 0.65:
            return False, {}
        return True, {
            'pattern_type': 'CONSOLIDATION_BOX_BREAKOUT',
            'box_high': round(box_high, 2), 'box_low': round(box_low, 2),
            'box_range': round(box_range, 2), 'prior_fall': round(prior_fall, 2),
            'vol': round(c['vol_ratio'] * 100, 1)
        }

    def detect_fall_box_breakout(idx, min_red_streak=4, max_wait=8):
        if idx < min_red_streak + 6 or is_choppy_market(idx):
            return False, {}
        c = data.iloc[idx]
        if not c['is_bullish']:
            return False, {}
        for gap in range(1, max_wait + 1):
            end_idx = idx - gap
            if end_idx < min_red_streak + 1:
                continue
            if data.iloc[end_idx]['is_bullish']:
                continue
            red_streak = 0
            j = end_idx
            while j >= 0 and not data.iloc[j]['is_bullish']:
                red_streak += 1
                j -= 1
            streak_start_idx = j + 1
            if red_streak <= min_red_streak:
                continue
            if end_idx - 1 < 0:
                continue
            last_red_idx = end_idx
            second_last_red_idx = end_idx - 1
            red1 = data.iloc[last_red_idx]
            red2 = data.iloc[second_last_red_idx]
            box_high = max(red1['Open'], red1['Close'], red2['Open'], red2['Close'])
            box_low = min(red1['Open'], red1['Close'], red2['Open'], red2['Close'])
            streak_high = data['High'].iloc[streak_start_idx:last_red_idx + 1].max()
            streak_low = data['Low'].iloc[streak_start_idx:last_red_idx + 1].min()
            total_fall = streak_high - streak_low
            if total_fall <= 40:
                continue
            atr_now = data['ATR'].iloc[idx]
            if pd.isna(atr_now) or atr_now == 0:
                continue
            if (streak_high - streak_low) < 1.5 * atr_now:
                continue
            hammer_idx = None
            for h in range(last_red_idx + 1, idx):
                hc = data.iloc[h]
                hc_range = hc['High'] - hc['Low']
                if hc_range <= 0:
                    continue
                hc_body_pct = abs(hc['Close'] - hc['Open']) / hc_range
                hc_upper_pct = (hc['High'] - max(hc['Open'], hc['Close'])) / hc_range
                hc_lower_pct = (min(hc['Open'], hc['Close']) - hc['Low']) / hc_range
                if hc_lower_pct >= 0.50 and hc_body_pct <= 0.30 and hc_upper_pct <= 0.10:
                    hammer_idx = h
                    break
            if hammer_idx is not None and hammer_idx == idx - 1:
                if c['vol_ratio'] < 0.55:
                    continue
                return True, {
                    'pattern_type': 'FALL_HAMMER_REVERSAL', 'red_streak': red_streak,
                    'total_fall': round(total_fall, 2), 'hammer_idx': int(hammer_idx),
                    'vol': round(c['vol_ratio'] * 100, 1)
                }
            already_broken = False
            for k in range(last_red_idx + 1, idx):
                ck = data.iloc[k]
                if ck['Close'] > box_high and ck['is_bullish'] and ck['body_pct'] >= 0.20:
                    already_broken = True
                    break
            if already_broken:
                continue
            if not (c['Close'] > box_high and c['body_pct'] >= 0.20 and c['close_pos'] >= 0.25):
                continue
            if c['vol_ratio'] < 0.55:
                continue
            return True, {
                'pattern_type': 'FALL_BOX_BREAKOUT', 'red_streak': red_streak,
                'total_fall': round(total_fall, 2), 'box_high': round(box_high, 2),
                'box_low': round(box_low, 2), 'vol': round(c['vol_ratio'] * 100, 1)
            }
        return False, {}

    # ====================== LEVELS, QUALITY, RATING ======================
    def calculate_levels(idx, condition):
        entry = data['Close'].iloc[idx]
        FIXED_SL_POINTS = 11
        FIXED_TARGET_POINTS = 30
        stop = entry - FIXED_SL_POINTS
        target = entry + FIXED_TARGET_POINTS
        rr = abs(target - entry) / (abs(entry - stop) + 1e-9)
        return entry, stop, target, rr

    def calculate_levels_bear(idx, condition):
        entry = data['Close'].iloc[idx]
        FIXED_SL_POINTS = 11
        FIXED_TARGET_POINTS = 30
        stop = entry + FIXED_SL_POINTS
        target = entry - FIXED_TARGET_POINTS
        rr = abs(entry - target) / (abs(stop - entry) + 1e-9)
        return entry, stop, target, rr

    def get_quality(rr):
        if rr < 1.25:
            return "REJECT"
        return "PREMIUM" if rr >= 2.2 else "HIGH" if rr >= 1.8 else "MEDIUM"

    _QUALITY_WEIGHT = {"PREMIUM": 3, "HIGH": 2, "MEDIUM": 1, "REJECT": -999}

    def _rating_score(confidence, quality, priority):
        return confidence * 0.6 + _QUALITY_WEIGHT[quality] * 20 - priority * 1.0

    def _write_signal(idx, side, condition, sig_data, confidence, entry, stop, target, rr,
                      quality, score, wait_note=""):
        data.loc[data.index[idx], 'st_sig'] = 1
        data.loc[data.index[idx], 'condition'] = condition
        data.loc[data.index[idx], 'quality'] = quality
        data.loc[data.index[idx], 'entry_price'] = entry
        data.loc[data.index[idx], 'stop_loss'] = stop
        data.loc[data.index[idx], 'take_profit'] = target
        data.loc[data.index[idx], 'risk_reward'] = rr
        data.loc[data.index[idx], 'confidence'] = confidence
        data.loc[data.index[idx], 'signal_type'] = side
        data.loc[data.index[idx], 'rating_score'] = score
        reason = f"{condition.replace('_', ' ')} | {side} | RR={rr:.2f}x | score={score:.1f} | "
        for k, v in sig_data.items():
            if isinstance(v, (int, float)):
                reason += f"{k}={v:.1f} "
        if wait_note:
            reason += wait_note
        data.loc[data.index[idx], 'reason'] = reason[:220]
        return {
            'datetime': data.index[idx], 'close': data['Close'].iloc[idx],
            'condition': condition, 'quality': quality, 'entry': entry, 'stop': stop,
            'target': target, 'rr': rr, 'confidence': confidence, 'signal_type': side,
            'rating_score': score
        }

    # ====================== MAIN SIGNAL LOOP ======================
    signals_list = []
    COOLDOWN_CANDLES = 1
    last_signal_idx = -(10 ** 9)

    pending_signals = []
    pending_signals_bear = []

    multi_div_bull_seen = 0
    multi_div_bear_seen = 0
    multi_div_bull_fired = 0
    multi_div_bear_fired = 0
    last_multi_div_bull_time = None
    last_multi_div_bear_time = None

    for idx in range(18, n):

        candidates_this_idx = []

        idx_close_now = data['idx_Close'].iloc[idx] if 'idx_Close' in data.columns else np.nan

        # ---- Pending CE confirmation ----
        still_pending = []
        for p in pending_signals:
            age = idx - p['detect_idx']
            if age < 0:
                still_pending.append(p)
                continue
            if age > confirm_max_wait:
                continue
            trend_now = data['ema_trend'].iloc[idx]
            if trend_now == 1:
                entry, stop, target, rr = calculate_levels(idx, p['condition'])
                quality = get_quality(rr)
                if quality != "REJECT":
                    score = _rating_score(p['confidence'], quality, p['priority'])
                    candidates_this_idx.append({
                        'side': 'CE', 'condition': p['condition'], 'quality': quality,
                        'entry': entry, 'stop': stop, 'target': target, 'rr': rr,
                        'confidence': p['confidence'], 'priority': p['priority'],
                        'data': p['data'], 'score': score,
                        'wait_note': f"(confirmed after {age} candle(s) via ema_trend)"
                    })
                continue
            else:
                still_pending.append(p)
        pending_signals = still_pending

        # ---- Pending PE confirmation ----
        still_pending_bear = []
        for p in pending_signals_bear:
            age = idx - p['detect_idx']
            if age < 0:
                still_pending_bear.append(p)
                continue
            if age > confirm_max_wait:
                continue
            trend_now = data['ema_trend'].iloc[idx]
            if trend_now == -1:
                entry, stop, target, rr = calculate_levels_bear(idx, p['condition'])
                quality = get_quality(rr)
                if quality != "REJECT":
                    score = _rating_score(p['confidence'], quality, p['priority'])
                    candidates_this_idx.append({
                        'side': 'PE', 'condition': p['condition'], 'quality': quality,
                        'entry': entry, 'stop': stop, 'target': target, 'rr': rr,
                        'confidence': p['confidence'], 'priority': p['priority'],
                        'data': p['data'], 'score': score,
                        'wait_note': f"(confirmed after {age} candle(s) via ema_trend)"
                    })
                continue
            else:
                still_pending_bear.append(p)
        pending_signals_bear = still_pending_bear

        # ---- Fire best candidate from pending confirmations ----
        if candidates_this_idx and (idx - last_signal_idx) >= COOLDOWN_CANDLES:
            best = max(candidates_this_idx, key=lambda x: x['score'])
            result = _write_signal(
                idx, best['side'], best['condition'], best['data'], best['confidence'],
                best['entry'], best['stop'], best['target'], best['rr'], best['quality'],
                best['score'], best['wait_note']
            )
            signals_list.append(result)
            last_signal_idx = idx
            continue

        if idx - last_signal_idx < COOLDOWN_CANDLES:
            continue

        # ---- Fresh pattern detection ----
        b_det, b_data = detect_base(idx)
        if b_det:
            base_morning_dates_fired.add(data.index[idx].date())
            pending_signals.append({'detect_idx': idx, 'condition': 'BASE_MORNING',
                                    'data': b_data, 'confidence': 82, 'priority': 1})

        v_det, v_data = detect_v_shape(idx)
        if v_det:
            pending_signals.append({'detect_idx': idx, 'condition': 'V_SHAPE_REVERSAL',
                                    'data': v_data, 'confidence': 76, 'priority': 2})

        c_det, c_data = detect_continuation(idx)
        if c_det:
            if c_data.get('side') == 'PE':
                pending_signals_bear.append({'detect_idx': idx, 'condition': c_data['pattern_type'],
                                             'data': c_data, 'confidence': 66, 'priority': 3})
            else:
                pending_signals.append({'detect_idx': idx, 'condition': c_data['pattern_type'],
                                        'data': c_data, 'confidence': 66, 'priority': 3})

        box_det, box_data = detect_box_consolidation_breakout(idx)
        if box_det:
            pending_signals.append({'detect_idx': idx, 'condition': 'CONSOLIDATION_BOX_BREAKOUT',
                                    'data': box_data, 'confidence': 74, 'priority': 4})

        fb_det, fb_data = detect_fall_box_breakout(idx)
        if fb_det:
            pending_signals.append({'detect_idx': idx, 'condition': fb_data['pattern_type'],
                                    'data': fb_data, 'confidence': 80, 'priority': 2})

        # ================================================================
        # MULTI-INDICATOR DIVERGENCE -- fires signal IMMEDIATELY on label
        # ================================================================
        if use_multi_div:
            row_label = data['label_type'].iloc[idx] if 'label_type' in data.columns else None
            if isinstance(row_label, str):
                indicators_str = data['divergence_indicators'].iloc[idx]
                pos_reg = data['pos_reg_div'].iloc[idx] if 'pos_reg_div' in data.columns else 0
                pos_hid = data['pos_hid_div'].iloc[idx] if 'pos_hid_div' in data.columns else 0
                neg_reg = data['neg_reg_div'].iloc[idx] if 'neg_reg_div' in data.columns else 0
                neg_hid = data['neg_hid_div'].iloc[idx] if 'neg_hid_div' in data.columns else 0

                src_time = (data['_div_label_time'].iloc[idx]
                            if '_div_label_time' in data.columns else data.index[idx])
                tm_now = data.index[idx].time()
                after_start_time = tm_now >= multi_div_start_time
                stoch_now = data['idx_STOCH_K'].iloc[idx]
                stoch_known = not pd.isna(stoch_now)
                stoch_oversold = stoch_known and stoch_now <= multi_div_stoch_oversold
                stoch_overbought = stoch_known and stoch_now >= multi_div_stoch_overbought
                stoch_str = f"{stoch_now:.1f}" if stoch_known else "n/a"

                # --------------------------------------------------------
                # BULLISH label -- fire CE signal immediately
                # --------------------------------------------------------
                if row_label == 'bottom' and src_time != last_multi_div_bull_time:
                    last_multi_div_bull_time = src_time
                    multi_div_bull_seen += 1
                    agree_count = int(pos_reg + pos_hid)

                    if not after_start_time:
                        print(f"[MULTI-DIV] BULLISH #{multi_div_bull_seen} @ {data.index[idx]} "
                              f"| agree={agree_count} | indicators=({indicators_str}) -> skipped "
                              f"(before {multi_div_start_time.strftime('%H:%M')} cutoff)")

                    elif not stoch_oversold:
                        print(f"[MULTI-DIV] BULLISH #{multi_div_bull_seen} @ {data.index[idx]} "
                              f"| agree={agree_count} | indicators=({indicators_str}) -> skipped "
                              f"(index Stoch %K={stoch_str}, not oversold "
                              f"<= {multi_div_stoch_oversold})")

                    elif multi_div_side in ('CE', 'BOTH'):
                        entry, stop, target, rr = calculate_levels(idx, 'MULTI_INDICATOR_BULLISH_DIVERGENCE')
                        quality = get_quality(rr)
                        if quality == "REJECT":
                            print(f"[MULTI-DIV] BULLISH #{multi_div_bull_seen} @ {data.index[idx]} "
                                  f"| agree={agree_count} | indicators=({indicators_str}) -> skipped "
                                  f"(RR={rr:.2f} rejected by quality gate)")
                        elif (idx - last_signal_idx) < COOLDOWN_CANDLES:
                            print(f"[MULTI-DIV] BULLISH #{multi_div_bull_seen} @ {data.index[idx]} "
                                  f"| agree={agree_count} | indicators=({indicators_str}) -> skipped "
                                  f"(cooldown: {COOLDOWN_CANDLES - (idx - last_signal_idx)} candle(s) remaining)")
                        else:
                            score = _rating_score(78, quality, 2)
                            sig_data = {
                                'agree_count': agree_count,
                                'pos_reg_div': pos_reg,
                                'pos_hid_div': pos_hid,
                                'stoch_k': stoch_now,
                            }
                            print(f"[MULTI-DIV] BULLISH #{multi_div_bull_seen} @ {data.index[idx]} "
                                  f"| agree={agree_count} | indicators=({indicators_str}) | "
                                  f"index Stoch %K={stoch_str} (oversold) -> FIRING CE signal immediately "
                                  f"| entry={entry:.2f} | quality={quality} | RR={rr:.2f}")
                            result = _write_signal(
                                idx, 'CE', 'MULTI_INDICATOR_BULLISH_DIVERGENCE', sig_data,
                                78, entry, stop, target, rr, quality, score,
                                wait_note="(immediate -- no queue)"
                            )
                            signals_list.append(result)
                            last_signal_idx = idx
                            multi_div_bull_fired += 1

                    else:
                        print(f"[MULTI-DIV] BULLISH #{multi_div_bull_seen} @ {data.index[idx]} "
                              f"| agree={agree_count} | indicators=({indicators_str}) -> skipped "
                              f"(this run is {multi_div_side}-side only)")

                # --------------------------------------------------------
                # BEARISH label -- fire PE signal immediately
                # --------------------------------------------------------
                elif row_label == 'top' and src_time != last_multi_div_bear_time:
                    last_multi_div_bear_time = src_time
                    multi_div_bear_seen += 1
                    agree_count = int(neg_reg + neg_hid)

                    if not after_start_time:
                        print(f"[MULTI-DIV] BEARISH #{multi_div_bear_seen} @ {data.index[idx]} "
                              f"| agree={agree_count} | indicators=({indicators_str}) -> skipped "
                              f"(before {multi_div_start_time.strftime('%H:%M')} cutoff)")

                    elif not stoch_overbought:
                        print(f"[MULTI-DIV] BEARISH #{multi_div_bear_seen} @ {data.index[idx]} "
                              f"| agree={agree_count} | indicators=({indicators_str}) -> skipped "
                              f"(index Stoch %K={stoch_str}, not overbought "
                              f">= {multi_div_stoch_overbought})")

                    elif multi_div_side in ('PE', 'BOTH'):
                        entry, stop, target, rr = calculate_levels_bear(idx, 'MULTI_INDICATOR_BEARISH_DIVERGENCE')
                        quality = get_quality(rr)
                        if quality == "REJECT":
                            print(f"[MULTI-DIV] BEARISH #{multi_div_bear_seen} @ {data.index[idx]} "
                                  f"| agree={agree_count} | indicators=({indicators_str}) -> skipped "
                                  f"(RR={rr:.2f} rejected by quality gate)")
                        elif (idx - last_signal_idx) < COOLDOWN_CANDLES:
                            print(f"[MULTI-DIV] BEARISH #{multi_div_bear_seen} @ {data.index[idx]} "
                                  f"| agree={agree_count} | indicators=({indicators_str}) -> skipped "
                                  f"(cooldown: {COOLDOWN_CANDLES - (idx - last_signal_idx)} candle(s) remaining)")
                        else:
                            score = _rating_score(78, quality, 2)
                            sig_data = {
                                'agree_count': agree_count,
                                'neg_reg_div': neg_reg,
                                'neg_hid_div': neg_hid,
                                'stoch_k': stoch_now,
                            }
                            print(f"[MULTI-DIV] BEARISH #{multi_div_bear_seen} @ {data.index[idx]} "
                                  f"| agree={agree_count} | indicators=({indicators_str}) | "
                                  f"index Stoch %K={stoch_str} (overbought) -> FIRING PE signal immediately "
                                  f"| entry={entry:.2f} | quality={quality} | RR={rr:.2f}")
                            result = _write_signal(
                                idx, 'PE', 'MULTI_INDICATOR_BEARISH_DIVERGENCE', sig_data,
                                78, entry, stop, target, rr, quality, score,
                                wait_note="(immediate -- no queue)"
                            )
                            signals_list.append(result)
                            last_signal_idx = idx
                            multi_div_bear_fired += 1

                    else:
                        print(f"[MULTI-DIV] BEARISH #{multi_div_bear_seen} @ {data.index[idx]} "
                              f"| agree={agree_count} | indicators=({indicators_str}) -> skipped "
                              f"(this run is {multi_div_side}-side only)")

    # ====================== SUMMARY ======================
    print(f"\n📊 SIGNALS GENERATED: {len(signals_list)}\n")
    if use_multi_div:
        md_fired = [s for s in signals_list if 'MULTI_INDICATOR' in s['condition']]
        md_fired_bull = sum(1 for s in md_fired if s['signal_type'] == 'CE')
        md_fired_bear = sum(1 for s in md_fired if s['signal_type'] == 'PE')
        print(f"   MULTI-DIV seen (bull/bear)            : {multi_div_bull_seen} / {multi_div_bear_seen}")
        print(f"   Actually FIRED as trades              : {len(md_fired)}  "
              f"(CE: {md_fired_bull}, PE: {md_fired_bear})")
        print("   [Signals not fired were dropped by: time cutoff, stoch filter, "
              "side filter, RR quality gate, or cooldown.]\n")

    if signals_list:
        print(f"{'DateTime':<26} {'Close':<10} {'Side':<5} {'Condition':<32} "
              f"{'Quality':<10} {'RR':<7} {'Score':<7} {'Conf':<6}")
        print("-" * 115)
        for sig in signals_list[-25:]:
            print(f"{str(sig['datetime']):<26} {sig['close']:<10.2f} "
                  f"{sig.get('signal_type', 'CE'):<5} "
                  f"{sig['condition']:<32} {sig['quality']:<10} {sig['rr']:<7.2f} "
                  f"{sig['rating_score']:<7.1f} {sig['confidence']:<6.1f}")

        conds = Counter(s['condition'] for s in signals_list)
        print("\n📈 CONDITION BREAKDOWN:")
        for c in sorted(conds):
            print(f"   {c}: {conds[c]}")

        sides = Counter(s.get('signal_type', 'CE') for s in signals_list)
        print("\n🎯 SIDE BREAKDOWN:")
        for side in ['CE', 'PE']:
            if side in sides:
                print(f"   {side}: {sides[side]}")

        quals = Counter(s['quality'] for s in signals_list)
        print("\n💎 QUALITY BREAKDOWN:")
        for q in ['PREMIUM', 'HIGH', 'MEDIUM']:
            if q in quals:
                print(f"   {q}: {quals[q]}")

        print(f"\n💰 AVG RR: {np.mean([s['rr'] for s in signals_list]):.2f}x")
        print(f"📊 AVG CONF: {np.mean([s['confidence'] for s in signals_list]):.1f}%")
        print(f"⭐ AVG RATING SCORE: {np.mean([s['rating_score'] for s in signals_list]):.1f}")

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
    (9, 15), (9, 18), (9, 21), (9, 24), (9, 27), (9, 30), (9, 33), (9, 36), (9, 39), (9, 42), (9, 45), (9, 48), (9, 51), (9, 54), (9, 57),
    (10, 0), (10, 3), (10, 6), (10, 9), (10, 12), (10, 15), (10, 18), (10, 21), (10, 24), (10, 27), (10, 30), (10, 33), (10, 36), (10, 39), (10, 42), (10, 45), (10, 48), (10, 51), (10, 54), (10, 57),
    (11, 0), (11, 3), (11, 6), (11, 9), (11, 12), (11, 15), (11, 18), (11, 21), (11, 24), (11, 27), (11, 30), (11, 33), (11, 36), (11, 39), (11, 42), (11, 45), (11, 48), (11, 51), (11, 54), (11, 57),
    (12, 0), (12, 3), (12, 6), (12, 9), (12, 12), (12, 15), (12, 18), (12, 21), (12, 24), (12, 27), (12, 30), (12, 33), (12, 36), (12, 39), (12, 42), (12, 45), (12, 48), (12, 51), (12, 54), (12, 57),
    (13, 0), (13, 3), (13, 6), (13, 9), (13, 12), (13, 15), (13, 18), (13, 21), (13, 24), (13, 27), (13, 30), (13, 33), (13, 36), (13, 39), (13, 42), (13, 45), (13, 48), (13, 51), (13, 54), (13, 57),
    (14, 0), (14, 3), (14, 6), (14, 9), (14, 12), (14, 15), (14, 18), (14, 21), (14, 24), (14, 27), (14, 30), (14, 33), (14, 36), (14, 39), (14, 42), (14, 45), (14, 48), (14, 51), (14, 54), (14, 57),
    (15, 0), (15, 3), (15, 6), (15, 9), (15, 12), (15, 15), (15, 18), (15, 21), (15, 24), (15, 27), (15, 30)
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
    df = get_cash_market_data_3('3m')
    data_fut =super_trend(h,data_fut,4,df)

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
# ── track processed candle minutes to avoid double-fire ──────
last_processed_minute = {}  # {ticker: (hour, minute)}



while dt.datetime.now(pytz.timezone('Asia/Kolkata')) < endTime:

    try:

        for i in Tickers:

            now_ist = dt.datetime.now(pytz.timezone('Asia/Kolkata'))
            current_minute = (now_ist.hour, now_ist.minute)

            

            if spot_prices1[i] is None:
                time.sleep(0.5)
                continue

            send_to_ui(i, spot_prices1[i])
            time.sleep(0.5)

            # ════════════════════════════════════════════════════
            # BLOCK A — SIGNAL + ENTRY
            # Runs only once per 5-min candle per ticker
            # ════════════════════════════════════════════════════
            if is_required_time() and last_processed_minute.get(i) != current_minute:

                last_processed_minute[i] = current_minute  # mark this candle done

                try:
                    data_fut = get_cash_market_data(i, '3m')
                    print("###################################################################")
                    print('Spot prices of', i, ' ', spot_prices1[i])
                    print(now_ist.strftime("%d-%b-%Y %I:%M:%S%p"))
                    print("###################################################################")
                except Exception as e:
                    print(f"⚠️ Skipping {i}: {e}")
                    continue

                data_fut.drop(data_fut.tail(1).index, inplace=True)
                df = get_cash_market_data_3('3m')
                data_fut =super_trend(i, data_fut,4,df)
                data_list[i] = data_fut

                super_Trend_Long = pd.read_excel(Long_Trade_File)
                Long_Open_Position = super_Trend_Long[super_Trend_Long['Trade Status'] == 'OPEN']
                super_Trend_Short = pd.read_excel(Short_Trade_File)
                Short_Open_Position = super_Trend_Short[super_Trend_Short['Trade Status'] == 'OPEN']

                # ── LONG ENTRY ────────────────────────────────
                if (data_list[i]["st_sig"].tail(3) == 1).any():

                    all_trade_files()
                    open_trades_df = pd.read_excel('All_Trades.xlsx')
                    open_trade_count = len(open_trades_df[open_trades_df['Trade Status'] == 'OPEN'])

                    if open_trade_count >= Max_Position:
                        print("Maximum Position Reached. No New Position.")
                        continue

                    if i in Long_Open_Position['Symbol'].values:
                        print(f"{i} already in Long Open Position. Skipping.")
                        continue

                    current_price  = float(spot_prices1[i])
                    Trade_quantity = 65
                    Target_Price   = current_price + Take_Profit
                    Sprice         = current_price - 10
                    entry_time     = dt.datetime.now(UTC).strftime("%d-%b-%Y %I:%M%p")
                    BuyPrice       = current_price

                    update_long_trades(i, entry_time, BuyPrice, Target_Price, Sprice, Trade_quantity, Long_Trade_File)
                    tele_msg(
                        f"Long Entry: {i} | Qty: {Trade_quantity} | "
                        f"Buy: {BuyPrice} | Target: {Target_Price} | SL: {Sprice}"
                    )

                    super_Trend_Long   = pd.read_excel(Long_Trade_File)
                    Long_Open_Position = super_Trend_Long[super_Trend_Long['Trade Status'] == 'OPEN']

            # ════════════════════════════════════════════════════
            # BLOCK B — SL / TARGET / TRAILING / TIMEOUT
            # Runs EVERY loop iteration (~every 0.5s)
            # NOT inside is_required_time — fires in real time
            # ════════════════════════════════════════════════════

            # Re-read positions fresh for exit checks
            super_Trend_Long   = pd.read_excel(Long_Trade_File)
            Long_Open_Position = super_Trend_Long[super_Trend_Long['Trade Status'] == 'OPEN']

            if i not in Long_Open_Position['Symbol'].values:
                continue  # nothing open, skip all exit checks

            trade_row      = Long_Open_Position[Long_Open_Position['Symbol'] == i]
            BuyPrice       = float(trade_row['Buy Price'].values[0])
            Target_Price   = float(trade_row['Target Price'].values[0])
            S_Price        = float(trade_row['Sprice'].values[0])
            Trade_quantity = int(trade_row['Qty'].values[0])
            current_price  = float(spot_prices1[i])

            profit_from_entry = current_price - BuyPrice

            print(f"  📊 {i} | Buy:{BuyPrice} | Now:{current_price} | "
                  f"Target:{Target_Price} | SL:{S_Price} | PnL:{round(profit_from_entry,2)}")

            # ── B1: TRAILING SL + STAGED TARGET ───────────────
            # Logic:
            #   Target steps up in 10s: 10 → 20 → 30 → 40 → 50 (capped at 50)
            #   SL trails in 5-point steps, always one step behind price,
            #   and never moves backward — locks in profit already gained
            Target_Step = 10
            SL_Step     = 5
            Max_Target_Offset = 50   # target stops climbing after entry+50

            if profit_from_entry >= SL_Step:

                # ── Target: steps in 10s, capped at Max_Target_Offset ──
                target_steps = int(profit_from_entry // Target_Step)
                new_target   = BuyPrice + min((target_steps + 1) * Target_Step, Max_Target_Offset)
                new_target   = max(new_target, Target_Price)   # never move Target down

                # ── SL: trails in 5-point steps, one step behind price ──
                sl_steps = int(profit_from_entry // SL_Step)
                new_sl   = BuyPrice + ((sl_steps - 2) * SL_Step)
                new_sl   = max(new_sl, S_Price)   # never move SL down — locks in profit already gained

                if new_sl > S_Price or new_target > Target_Price:
                    print(f"  📈 TRAILING SL: {S_Price} → {new_sl}  |  TARGET: {Target_Price} → {new_target}  [{i}]")
                    tele_msg(f"📈 Trail Update: {i} | SL: {S_Price}→{new_sl} | Target: {Target_Price}→{new_target}")

                    # Update Sprice (SL column) in Excel
                    update_buy_price(i, new_sl, Long_Trade_File)      # updates Sprice column
                    # Update Target Price column in Excel
                    update_target_price(i, new_target, Long_Trade_File)

                    # Refresh local variables after update
                    super_Trend_Long   = pd.read_excel(Long_Trade_File)
                    Long_Open_Position = super_Trend_Long[super_Trend_Long['Trade Status'] == 'OPEN']
                    trade_row          = Long_Open_Position[Long_Open_Position['Symbol'] == i]
                    S_Price            = float(trade_row['Sprice'].values[0])
                    Target_Price       = float(trade_row['Target Price'].values[0])

                

            # ── B2: HARD SL HIT ──────────────────────────────
            # Check SL BEFORE target so trailing SL fires first
            if current_price <= S_Price:
                print(f"  🔴 SL HIT → {i} | Price:{current_price} | SL:{S_Price}")
                Exit_Time   = dt.datetime.now(UTC).strftime("%d-%b-%Y %I:%M%p")
                Sell_Price  = current_price
                Points      = Sell_Price - BuyPrice
                Brokerage   = ((BuyPrice * Trade_quantity) + (Sell_Price * Trade_quantity)) * 0.00015
                Profit_Loss = (Points * Trade_quantity) - Brokerage
                Trade_Status = "SL Hit"

                close_long_trade(i, Exit_Time, Sell_Price, Points, Brokerage, Profit_Loss, Trade_Status, Long_Trade_File)
                tele_msg(f"🔴 SL Hit: {i} | Exit: {Sell_Price} | P/L: {Profit_Loss}")

                super_Trend_Long   = pd.read_excel(Long_Trade_File)
                Long_Open_Position = super_Trend_Long[super_Trend_Long['Trade Status'] == 'OPEN']
                continue

            # ── B3: TARGET HIT ────────────────────────────────
            if current_price >= Target_Price:
                print(f"  🎯 TARGET HIT → {i} | Price:{current_price} | Target:{Target_Price}")
                Exit_Time   = dt.datetime.now(UTC).strftime("%d-%b-%Y %I:%M%p")
                Sell_Price  = current_price
                Points      = Sell_Price - BuyPrice
                Brokerage   = ((BuyPrice * Trade_quantity) + (Sell_Price * Trade_quantity)) * 0.00015
                Profit_Loss = (Points * Trade_quantity) - Brokerage
                Trade_Status = "Target Hit"

                close_long_trade(i, Exit_Time, Sell_Price, Points, Brokerage, Profit_Loss, Trade_Status, Long_Trade_File)
                tele_msg(f"🎯 Target Hit: {i} | Exit: {Sell_Price} | P/L: {Profit_Loss}")

                super_Trend_Long   = pd.read_excel(Long_Trade_File)
                Long_Open_Position = super_Trend_Long[super_Trend_Long['Trade Status'] == 'OPEN']
                continue

            # ── B4: EXIT TIME OUT ─────────────────────────────
            # Exit 5 minutes before end time
            exit_cutoff = now_ist.replace(
                hour=EXIT_TIME[0],
                minute=max(EXIT_TIME[1] - 5, 0),
                second=EXIT_TIME[2]
            )
            if now_ist >= exit_cutoff:
                print(f"  ⏰ TIME OUT → {i}")
                Exit_Time   = dt.datetime.now(UTC).strftime("%d-%b-%Y %I:%M%p")
                Sell_Price  = current_price
                Points      = Sell_Price - BuyPrice
                Brokerage   = ((BuyPrice * Trade_quantity) + (Sell_Price * Trade_quantity)) * 0.00015
                Profit_Loss = (Points * Trade_quantity) - Brokerage
                Trade_Status = "Exit Time Out"

                close_long_trade(i, Exit_Time, Sell_Price, Points, Brokerage, Profit_Loss, Trade_Status, Long_Trade_File)
                tele_msg(f"⏰ Time Out: {i} | Exit: {Sell_Price} | P/L: {Profit_Loss}")

                super_Trend_Long   = pd.read_excel(Long_Trade_File)
                Long_Open_Position = super_Trend_Long[super_Trend_Long['Trade Status'] == 'OPEN']
                continue

    except Exception as e:
        ct = dt.datetime.now().strftime("%d-%b-%Y %I:%M%p")
        error_message = f"{ct} - An error occurred: {e}"
        print("Error:", e)
        print("Oops!", e.__class__, "occurred.")
        tele_msg(error_message)
        with open("error_log.txt", "a") as error_log_file:
            error_log_file.write(error_message + "\n")
        # ✅ NO raise — bot stays alive on any error
        time.sleep(2)
           # error_log_file.write(error_message+"\n")
        raise ValueError("I have raised an Exception in main")
