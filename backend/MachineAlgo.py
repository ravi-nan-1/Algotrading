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

Take_Profit = 30

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



def get_cash_market_data_3(timeframe='5m'):
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







def super_trendjj(symbol, data, use_llm=False):
    """
    IMPROVED Reversal Strategy - Buy at Lows with Better Risk Management
    - Cleaner code, fewer bugs, stronger filters, adaptive elements
    - Reduced overfitting risk (fewer magic numbers, better logic)
    """
    if 'Datetime' in data.columns:
        data = data.set_index('Datetime')
    data = data.copy()

    print("=" * 80)
    print(f"🚀 IMPROVED SUPER TREND → {symbol} | Shape: {data.shape}")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════
    # TUNABLE PARAMETERS (more balanced defaults)
    # ═══════════════════════════════════════════════════════
    ATR_PERIOD = 14
    EMA_TREND_PERIOD = 21  # Slightly longer for better trend filter
    STOCH_K = 14
    STOCH_SMOOTH = 3
    STOCH_D = 3

    ENTRY_PROB = 55.0  # Lowered a bit for more opportunities
    EXIT_PROB = 42.0
    MIN_BODY_EFF = 0.20  # Stricter for quality candles
    MIN_VOL_RATIO = .81  # Stronger volume confirmation
    RR_RATIO = 2.0  # More realistic RR (was 2.5)
    MAX_HOLD_BARS = 15  # Shorter hold to reduce exposure

    K_OVERSOLD = 20  # Slightly higher (less extreme)
    K_OVERSOLD_LOOKBACK = 5

    n = len(data)
    if n < 150:  # Increased warmup
        data['st_sig'] = 0
        data['state'] = 0
        return data

    # ═══════════════════════════════════════════════════════
    # INDICATORS (improved & cleaned)
    # ═══════════════════════════════════════════════════════
    data['ema_trend'] = data['Close'].ewm(span=EMA_TREND_PERIOD, adjust=False).mean()

    # True Range based ATR (more accurate than High-Low only)
    high_low = data['High']-data['Low']
    high_close = np.abs(data['High']-data['Close'].shift())
    low_close = np.abs(data['Low']-data['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['atr'] = tr.rolling(ATR_PERIOD).mean()

    # Velocity (smoothed returns)
    data['velocity'] = np.log(data['Close'] / data['Close'].shift(1)).ewm(span=5, adjust=False).mean()

    # Stochastic
    low_min = data['Low'].rolling(STOCH_K).min()
    high_max = data['High'].rolling(STOCH_K).max()
    raw_k = 100 * (data['Close']-low_min) / (high_max-low_min+1e-9)
    data['k'] = raw_k.rolling(STOCH_SMOOTH).mean()
    data['d'] = data['k'].rolling(STOCH_D).mean()

    # Candle strength & Volume
    rng = (data['High']-data['Low']).replace(0, np.nan)
    body = (data['Close']-data['Open']).abs()
    data['body_efficiency'] = (body / rng).fillna(0)
    data['range_exp'] = rng / (rng.rolling(20).mean()+1e-9)
    data['vol_ratio'] = data['Volume'] / (data['Volume'].rolling(20).mean()+1e-9)

    # ═══════════════════════════════════════════════════════
    # IMPROVED PROBABILITY SCORE
    # ═══════════════════════════════════════════════════════
    vel_score = 100 / (1+np.exp(-data['velocity'] * 12))  # Slightly steeper
    mom_score = ((data['k']-25).clip(lower=0) / 75) * 100  # Adjusted baseline
    disp_score = (data['body_efficiency'] * 100+
                  data['range_exp'] * 35+
                  data['vol_ratio'] * 25) / 1.6  # Rebalanced

    # Dynamic weighting based on trend strength (helps in different regimes)
    trend_strength = (data['Close']-data['ema_trend']).abs() / data['atr'].replace(0, np.nan)
    trend_factor = trend_strength.clip(upper=3) / 3.0  # 0-1 normalization

    data['probability_score'] = (
            0.40 * vel_score * (1+0.2 * trend_factor)+  # Velocity more important in trend
            0.35 * mom_score+
            0.25 * disp_score
    ).clip(0, 100)

    # ═══════════════════════════════════════════════════════
    # SIGNAL GENERATION (bug fixes + stronger logic)
    # ═══════════════════════════════════════════════════════
    data['st_sig'] = 0
    data['signal_priority'] = ''
    data['stop_price'] = np.nan
    data['target_price'] = np.nan
    data['state'] = 0
    data['exit_reason'] = ''

    in_position = False
    entry_idx = None
    entry_price = None

    for i in range(150, n):
        current_idx = data.index[i]

        if in_position:
            # Exit logic (improved)
            hit_stop = data['Low'].iloc[i] <= data['stop_price'].iloc[i-1]
            hit_target = data['High'].iloc[i] >= data['target_price'].iloc[i-1]
            momentum_fail = data['velocity'].iloc[i] < -0.0008
            prob_faded = data['probability_score'].iloc[i] < EXIT_PROB
            timeout = (i-entry_idx) > MAX_HOLD_BARS
            # New: trailing stop
            trailing_stop = entry_price * 0.985 if (i-entry_idx) > 5 else data['stop_price'].iloc[i-1]

            if hit_stop or hit_target or momentum_fail or prob_faded or timeout or (
                    data['Low'].iloc[i] <= trailing_stop):
                in_position = False
                data.loc[current_idx, 'state'] = 0
                if hit_target:
                    data.loc[current_idx, 'exit_reason'] = 'TARGET'
                elif hit_stop or (data['Low'].iloc[i] <= trailing_stop):
                    data.loc[current_idx, 'exit_reason'] = 'STOP'
                else:
                    data.loc[current_idx, 'exit_reason'] = 'OTHER'
                continue
            else:
                data.loc[current_idx, 'state'] = 1
            continue

        # Market Filters
        in_uptrend = data['Close'].iloc[i] > data['ema_trend'].iloc[i]
        favorable = (data['range_exp'].iloc[i] > 0.85) and (data['vol_ratio'].iloc[i] > 0.75)
        strong_momentum = data['k'].iloc[i] > data['d'].iloc[i] and data['k'].iloc[i-1] <= data['d'].iloc[i-1]

        # Oversold context
        lookback_start = max(0, i-K_OVERSOLD_LOOKBACK)
        recent_k_min = data['k'].iloc[lookback_start:i+1].min()

        # HIGH PRIORITY (best reversals near lows)
        high_priority = (
                favorable and in_uptrend and strong_momentum and
                recent_k_min <= K_OVERSOLD and
                data['body_efficiency'].iloc[i] > MIN_BODY_EFF and
                data['vol_ratio'].iloc[i] > MIN_VOL_RATIO and
                data['probability_score'].iloc[i] >= ENTRY_PROB-8  # Slightly relaxed for high-prio
        )

        # LOW PRIORITY (still good setups)
        low_priority = (
                favorable and
                data['probability_score'].iloc[i] > ENTRY_PROB and
                data['body_efficiency'].iloc[i] > MIN_BODY_EFF and
                data['vol_ratio'].iloc[i] > MIN_VOL_RATIO
        )

        if high_priority or low_priority:
            data.loc[current_idx, 'st_sig'] = 1
            data.loc[current_idx, 'signal_priority'] = 'HIGH' if high_priority else 'LOW'
            data.loc[current_idx, 'state'] = 1

            atr_val = data['atr'].iloc[i]
            stop_dist = atr_val * 1.65  # Tighter stop
            entry_price = data['Close'].iloc[i]

            data.loc[current_idx, 'stop_price'] = entry_price-stop_dist
            data.loc[current_idx, 'target_price'] = entry_price+(stop_dist * RR_RATIO)

            in_position = True
            entry_idx = i

    # Summary
    buy_signals = data[data['st_sig'] == 1]
    print(f"\n📊 BUY SIGNALS GENERATED: {len(buy_signals)}")
    if not buy_signals.empty:
        print(buy_signals[['Close', 'probability_score', 'signal_priority',
                           'stop_price', 'target_price']].round(4))

    return data



def super_trend(symbol: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    AGGRESSIVE MULTI-BRANCH OPTION BUYING STRATEGY
    FIXED: Morning trades + Wick Reversal + Enhanced Range-Bound Filter (PSAR + Candle Logic)
    """
    if 'Datetime' in data.columns:
        data = data.set_index('Datetime')

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    data = data.copy()
    n = len(data)

    print("=" * 100)
    print(f"🚀 ENHANCED RANGE-BOUND STRATEGY → {symbol} | Data Points: {n}")
    print("=" * 100)

    # Initialize signal columns
    data['st_sig'] = 0
    data['condition'] = ''
    data['quality'] = ''
    data['entry_price'] = np.nan
    data['stop_loss'] = np.nan
    data['take_profit'] = np.nan
    data['risk_reward'] = np.nan
    data['confidence'] = 0.0
    data['reason'] = ''

    # Technical Indicators
    high_low = data['High']-data['Low']
    high_close = np.abs(data['High']-data['Close'].shift())
    low_close = np.abs(data['Low']-data['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(14).mean()

    data['EMA5'] = data['Close'].ewm(span=5, adjust=False).mean()
    data['EMA10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()

    data['range'] = data['High']-data['Low']
    data['body'] = (data['Close']-data['Open']).abs()
    data['body_pct'] = data['body'] / (data['range'].replace(0, np.nan))
    data['close_pos'] = (data['Close']-data['Low']) / (data['range'].replace(0, np.nan))
    data['is_bullish'] = data['Close'] > data['Open']

    data['vol_ma20'] = data['Volume'].rolling(20).mean()
    data['vol_ratio'] = data['Volume'] / (data['vol_ma20']+1e-9)

    # ====================== PSAR ======================
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
            if trend[i-1] == 1:
                sar[i] = min(sar[i-1]+af[i-1] * (ep[i-1]-sar[i-1]), low[i-1])
                if low[i] < sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = low[i]
                    af[i] = iaf
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1]+iaf, maxaf)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            else:
                sar[i] = max(sar[i-1]-af[i-1] * (sar[i-1]-ep[i-1]), high[i-1])
                if high[i] > sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = high[i]
                    af[i] = iaf
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1]+iaf, maxaf)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        return sar, trend

    data['PSAR'], data['PSAR_trend'] = psar(data['High'].values, data['Low'].values, data['Close'].values)

    # ====================== ENHANCED RANGE-BOUND FILTER ======================
    def is_range_boundgh(idx, lookback=12, threshold_pct=0.85):
        """Returns True if market is range-bound (PSAR + Candle Logic)"""
        if idx < max(lookback, 20):
            return False

        window = data.iloc[idx-lookback:idx+1]
        c = data.iloc[idx]
        prev = data.iloc[idx-1] if idx > 0 else c

        # 1. Overall Price Range
        price_range_pct = (window['High'].max()-window['Low'].min()) / window['Close'].iloc[-1] * 100

        # 2. PSAR Flip Count (high flips = ranging market)
        psar_window = data['PSAR_trend'].iloc[idx-lookback:idx+1]
        psar_flips = (psar_window.diff() != 0).sum()

        # 3. Current candle did not break previous high significantly
        no_breakout = c['High'] <= prev['High'] * 1.002

        # 4. Previous candles have small bodies
        recent_bodies = data['body_pct'].iloc[max(0, idx-7):idx]
        small_body_condition = (recent_bodies < 0.48).mean() >= 0.65

        # Final Decision
        is_range = (
                price_range_pct < threshold_pct or
                (psar_flips >= 4 and no_breakout and small_body_condition)
        )
        return is_range

    def is_range_bound(idx, lookback=12, threshold_pct=0.85):
        """Enhanced Range Detection using PSAR + Candle Logic"""
        if idx < max(lookback, 20):
            return False

        window = data.iloc[idx-lookback:idx+1]
        c = data.iloc[idx]
        prev = data.iloc[idx-1] if idx > 0 else c

        # 1. Price Range
        price_range_pct = (window['High'].max()-window['Low'].min()) / window['Close'].iloc[-1] * 100

        # 2. PSAR - Frequent flips indicate range-bound market
        psar_window = data['PSAR_trend'].iloc[idx-lookback:idx+1]
        psar_flips = (psar_window.diff() != 0).sum()

        # 3. Current candle not breaking previous high
        no_breakout = c['High'] <= prev['High'] * 1.002

        # 4. Previous candles body not too big
        recent_bodies = data['body_pct'].iloc[max(0, idx-7):idx]
        small_body_condition = (recent_bodies < 0.48).mean() >= 0.65

        # Final Range Decision
        is_range = (
                price_range_pct < threshold_pct or
                (psar_flips >= 4 and no_breakout and small_body_condition)
        )
        return is_range
    # Consolidation flag (for reference)
    data['consolidation'] = False
    for i in range(20, n):
        win = data.iloc[i-20:i]
        rng = (win['High'].max()-win['Low'].min()) / win['Close'].iloc[-1]
        data.loc[data.index[i], 'consolidation'] = rng < 0.019

    # ====================== DETECTORS ======================
    def detect_base(idx):
        if idx < 20 or is_range_bound(idx, lookback=12, threshold_pct=0.85):
            return False, {}

        tm = data.index[idx].time()
        if not (dt.time(9, 15) <= tm <= dt.time(10, 30)):
            return False, {}

        c = data.iloc[idx]
        uptrend = c['Close'] > c['EMA5'] > c['EMA20']
        strong_body = c['is_bullish'] and c['body_pct'] >= 0.55 and c['close_pos'] >= 0.60
        momentum = (c['Close']-c['Open']) / (c['High']-c['Low']+1e-9) > 0.62
        vol_ok = c['vol_ratio'] > 1.20

        if not (uptrend and strong_body and momentum and vol_ok):
            return False, {}

        if c['Close'] < data['Low'].iloc[idx-8:idx].min() * 0.985:
            return False, {}

        return True, {
            'pattern_type': 'BASE_MORNING',
            'strength': 'HIGH',
            'vol': round(c['vol_ratio'] * 100, 1)
        }

    def detect_v_shape(idx):
        if idx < 14 or is_range_bound(idx, lookback=14, threshold_pct=0.95):
            return False, {}

        lookback = 9
        window = data.iloc[idx-lookback:idx+1]
        low_idx_rel = window['Low'].values.argmin()
        pivot_idx = idx-lookback+low_idx_rel
        bars_since_pivot = idx-pivot_idx

        if bars_since_pivot < 1 or bars_since_pivot > 5:
            return False, {}

        if data['Low'].iloc[pivot_idx] > data['Low'].iloc[max(0, pivot_idx-6):pivot_idx+7].min():
            return False, {}

        left_start = max(0, pivot_idx-8)
        left_window = data.iloc[left_start:pivot_idx+1]
        swing_high = left_window['High'].max()
        pivot_low = data['Low'].iloc[pivot_idx]
        decline_pct = (swing_high-pivot_low) / swing_high * 100 if swing_high > 0 else 0
        points_drop = swing_high-pivot_low

        red_streak = 0
        for j in range(pivot_idx, left_start-1, -1):
            cj = data.iloc[j]
            if cj['is_bullish']:
                break
            if cj['body_pct'] < (0.30 if idx < 30 else 0.35):
                break
            red_streak += 1

        is_sharp_fall = points_drop >= 20 or decline_pct >= 1.0
        is_steep_decline = decline_pct >= 0.75 or points_drop >= 15

        if not (is_sharp_fall or is_steep_decline):
            return False, {}

        ema5_below_found = any(
            (data.iloc[j]['Low'] < data.iloc[j].get('EMA5', np.inf))
            for j in range(left_start, pivot_idx+1)
        )
        if not ema5_below_found and 'EMA5' in data.columns:
            return False, {}

        c = data.iloc[idx]
        candle_range = c['High']-c['Low']
        lower_wick = min(c['Open'], c['Close'])-c['Low']
        lower_wick_pct = lower_wick / candle_range if candle_range > 0 else 0
        upper_wick_pct = (c['High']-max(c['Open'], c['Close'])) / candle_range if candle_range > 0 else 0

        is_prev_long_wick_red = False
        if idx > 0:
            pc = data.iloc[idx-1]
            p_range = pc['High']-pc['Low']
            if p_range > 0 and not pc['is_bullish']:
                p_lower = min(pc['Open'], pc['Close'])-pc['Low']
                if p_lower / p_range >= 0.52:
                    is_prev_long_wick_red = True

        pattern_type = None
        if c['is_bullish'] and c['body_pct'] >= 0.52 and c['close_pos'] >= 0.58:
            pattern_type = 'GREEN'
        elif (not c['is_bullish']) and lower_wick_pct >= 0.48 and c['close_pos'] >= 0.35:
            if c['body_pct'] >= 0.12 or lower_wick_pct >= 0.65:
                pattern_type = 'WICK'
            elif lower_wick_pct >= 0.72 and upper_wick_pct <= 0.22:
                pattern_type = 'WICK_DOJI'
        elif is_prev_long_wick_red and c['is_bullish'] and c['body_pct'] >= 0.45:
            pattern_type = 'GREEN_AFTER_WICK'

        if pattern_type is None:
            return False, {}

        recovery_pct = (c['Close']-pivot_low) / (swing_high-pivot_low+1e-9)
        min_recovery = 0.14 if 'WICK' in pattern_type else 0.35
        if recovery_pct < min_recovery:
            return False, {}

        recent_high = data['High'].iloc[pivot_idx:idx].max()
        breakout_price = c['High'] if 'WICK' in pattern_type else c['Close']
        if breakout_price < recent_high * 0.996:
            return False, {}

        min_vol = 0.85 if 'WICK' in pattern_type else 1.10
        if c['vol_ratio'] < min_vol:
            return False, {}

        return True, {
            'pattern_type': pattern_type,
            'decline_pct': round(decline_pct, 2),
            'recovery_pct': round(recovery_pct * 100, 2),
            'lower_wick_pct': round(lower_wick_pct * 100, 2),
            'vol': round(c['vol_ratio'] * 100, 1)
        }

    def detect_continuation(idx):
        if idx < 20 or is_range_bound(idx, lookback=10, threshold_pct=0.9):
            return False, {}

        tm = data.index[idx].time()
        if dt.time(9, 15) <= tm <= dt.time(9, 45):
            return False, {}

        c = data.iloc[idx]
        ema10 = data['EMA10'].iloc[idx]
        ema20 = data['EMA20'].iloc[idx]

        trend_ok = c['Close'] > ema10 > ema20
        body_ok = c['is_bullish'] and c['body_pct'] >= 0.48 and c['close_pos'] >= 0.55
        vol_ok = c['vol_ratio'] > 1.15
        pullback = data['Close'].iloc[idx-4:idx].min() < ema10 * 0.999
        breakout = c['Close'] > data['High'].iloc[idx-3:idx].max() * 0.999

        if not (trend_ok and body_ok and vol_ok and pullback and breakout):
            return False, {}

        return True, {
            'pattern_type': 'CONTINUATION',
            'strength': 'MEDIUM',
            'vol': round(c['vol_ratio'] * 100, 1)
        }

    # ====================== LEVELS & QUALITY ======================
    def calculate_levels(idx, condition):
        entry = data['Close'].iloc[idx]
        FIXED_SL_POINTS = 11
        FIXED_TARGET_POINTS = 30
        stop = entry-FIXED_SL_POINTS
        target = entry+FIXED_TARGET_POINTS
        rr = abs(target-entry) / (abs(entry-stop)+1e-9)
        return entry, stop, target, rr

    def get_quality(rr):
        if rr < 1.25:
            return "REJECT"
        return "PREMIUM" if rr >= 2.2 else "HIGH" if rr >= 1.8 else "MEDIUM"

    # ====================== MAIN SIGNAL LOOP ======================
    signals_list = []

    for idx in range(18, n):
        candidates = []

        # Base Morning
        b_det, b_data = detect_base(idx)
        if b_det:
            entry, stop, target, rr = calculate_levels(idx, 'BASE')
            quality = get_quality(rr)
            if quality != "REJECT":
                candidates.append({
                    'condition': 'BASE_MORNING', 'quality': quality, 'entry': entry,
                    'stop': stop, 'target': target, 'rr': rr, 'confidence': 82,
                    'data': b_data, 'priority': 1
                })

        # V-Shape Reversal
        v_det, v_data = detect_v_shape(idx)
        if v_det:
            entry, stop, target, rr = calculate_levels(idx, 'V_SHAPE')
            quality = get_quality(rr)
            if quality != "REJECT":
                conf = 88 if quality == "PREMIUM" else 76
                candidates.append({
                    'condition': 'V_SHAPE_REVERSAL', 'quality': quality, 'entry': entry,
                    'stop': stop, 'target': target, 'rr': rr, 'confidence': conf,
                    'data': v_data, 'priority': 2
                })

        # Continuation
        c_det, c_data = detect_continuation(idx)
        if c_det:
            entry, stop, target, rr = calculate_levels(idx, 'CONTINUATION')
            quality = get_quality(rr)
            if quality != "REJECT":
                candidates.append({
                    'condition': 'BULLISH_CONTINUATION', 'quality': quality, 'entry': entry,
                    'stop': stop, 'target': target, 'rr': rr, 'confidence': 66,
                    'data': c_data, 'priority': 3
                })

        if candidates:
            best = max(candidates, key=lambda x: (
                {"PREMIUM": 3, "HIGH": 2, "MEDIUM": 1}.get(x['quality'], 0),
                -x['priority']
            ))

            condition = best['condition']
            quality = best['quality']
            entry = best['entry']
            stop = best['stop']
            target = best['target']
            rr = best['rr']
            confidence = best['confidence']
            sig_data = best['data']

            data.loc[data.index[idx], 'st_sig'] = 1
            data.loc[data.index[idx], 'condition'] = condition
            data.loc[data.index[idx], 'quality'] = quality
            data.loc[data.index[idx], 'entry_price'] = entry
            data.loc[data.index[idx], 'stop_loss'] = stop
            data.loc[data.index[idx], 'take_profit'] = target
            data.loc[data.index[idx], 'risk_reward'] = rr
            data.loc[data.index[idx], 'confidence'] = confidence

            reason = f"{condition.replace('_', ' ')} | RR={rr:.2f}x | "
            for k, v in sig_data.items():
                if isinstance(v, (int, float)):
                    reason += f"{k}={v:.1f} "
            data.loc[data.index[idx], 'reason'] = reason[:200]

            signals_list.append({
                'datetime': data.index[idx], 'close': data['Close'].iloc[idx],
                'condition': condition, 'quality': quality, 'entry': entry,
                'stop': stop, 'target': target, 'rr': rr, 'confidence': confidence
            })

    # ====================== SUMMARY ======================
    print(f"\n📊 SIGNALS GENERATED: {len(signals_list)}\n")
    if signals_list:
        print(f"{'DateTime':<30} {'Close':<10} {'Condition':<25} {'Quality':<10} {'RR':<8} {'Conf':<8}")
        print("-" * 95)
        for sig in signals_list[-25:]:
            print(f"{str(sig['datetime']):<30} {sig['close']:<10.2f} {sig['condition']:<25} "
                  f"{sig['quality']:<10} {sig['rr']:<8.2f} {sig['confidence']:<8.1f}")

        from collections import Counter
        conds = Counter(s['condition'] for s in signals_list)
        print("\n📈 CONDITION BREAKDOWN:")
        for c in sorted(conds):
            print(f"   {c}: {conds[c]}")

        quals = Counter(s['quality'] for s in signals_list)
        print("\n💎 QUALITY BREAKDOWN:")
        for q in ['PREMIUM', 'HIGH', 'MEDIUM']:
            if q in quals:
                print(f"   {q}: {quals[q]}")

        print(f"\n💰 AVG RR: {np.mean([s['rr'] for s in signals_list]):.2f}x")
        print(f"📊 AVG CONF: {np.mean([s['confidence'] for s in signals_list]):.1f}%")

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

    data_fut =super_trend(h,data_fut)

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
                data_fut =super_trend(i, data_fut)
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
                new_sl   = BuyPrice + ((sl_steps - 1) * SL_Step)
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
