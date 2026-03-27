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



def super_trend(symbol, data, use_llm=True):
    """
    ╔══════════════════════════════════════════════════════════════╗
    ║          5-STEP VERIFICATION TRADING MACHINE                ║
    ║          LONG ONLY · STOCHASTIC BASED                       ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  STEP 1: OVERSOLD DETECTION                                  ║
    ║          K < 20 AND D < 20 → System ARMED                   ║
    ║                                                              ║
    ║  STEP 2: REVERSAL CONFIRMATION                               ║
    ║          K crosses above D + Both turning up (V-shape)       ║
    ║                                                              ║
    ║  STEP 3: MOMENTUM & VELOCITY CHECK                           ║
    ║          K/D slow vs Price fast → Strength confirmed         ║
    ║                                                              ║
    ║  STEP 4: MID-ZONE CROSSOVER (Alternative Entry)             ║
    ║          K crosses D between 20-80 + Momentum/Velocity OK   ║
    ║                                                              ║
    ║  STEP 5: BULLISH CANDLE PATTERN CONFIRMATION ★NEW★          ║
    ║          Oversold + K×D + (K-D)>5 + Bullish candle/pattern  ║
    ║          Detects: Engulfing, Hammer, Morning Star,           ║
    ║          Piercing Line, Three White Soldiers, Marubozu,      ║
    ║          Harami, Dragonfly Doji, simple green candle         ║
    ║                                                              ║
    ║  ★ MASTER FILTER (ALL PATHS): ★                             ║
    ║    • Previous 2 candles MUST be GREEN                        ║
    ║    • Current Open > Previous Open                            ║
    ║    • Current High > Previous High                            ║
    ║                                                              ║
    ║  TRADE: Step1+2+3 OR Step4 OR Step5 must pass               ║
    ║         + Master Filter must pass                            ║
    ╚══════════════════════════════════════════════════════════════╝

    PARAMS:
        symbol   : Stock symbol
        data     : DataFrame with OHLCV
        use_llm  : True  → Filter through LLM
                   False → Raw technical signals only
    """

    # ═══════════════════════════════════════════════════════════════
    #  MACHINE INITIALIZATION — COMPUTE ALL INDICATORS
    # ═══════════════════════════════════════════════════════════════
    print("=" * 70)
    print(f"  ⚙️  TRADING MACHINE INITIALIZING — {symbol}")
    print("=" * 70)

    # --- Stochastic Oscillator ---
    stoch = ta.stoch(data['High'], data['Low'], data['Close'], 14, 3, 3)
    data['Stoch_K'] = stoch['STOCHk_14_3_3']
    data['Stoch_D'] = stoch['STOCHd_14_3_3']

    # --- EMAs ---
    data['EMA5']  = ta.ema(data['Close'], length=5)
    data['EMA9']  = ta.ema(data['Close'], length=9)
    data['EMA15'] = ta.ema(data['Close'], length=15)

    # --- Aliases ---
    K      = data['Stoch_K']
    D      = data['Stoch_D']
    close  = data['Close']
    opn    = data['Open']
    high   = data['High']
    low    = data['Low']
    volume = data['Volume']
    ema5   = data['EMA5']
    ema9   = data['EMA9']
    ema15  = data['EMA15']

    # --- Velocities ---
    k_vel       = K - K.shift(1)
    d_vel       = D - D.shift(1)
    k_accel     = k_vel - k_vel.shift(1)
    d_accel     = d_vel - d_vel.shift(1)

    # --- Price velocities ---
    price_change_pct = ((close - close.shift(1)) / close.shift(1) * 100).fillna(0)
    price_vel_abs    = close - close.shift(1)

    # --- Candle basics ---
    green_candle   = close > opn
    candle_body    = abs(close - opn)
    candle_range   = high - low
    body_ratio     = (candle_body / candle_range.replace(0, float('nan'))).fillna(0)

    # --- Volume ---
    vol_rising     = volume > volume.shift(1)
    vol_avg_10     = volume.rolling(10).mean()
    vol_above_avg  = volume > vol_avg_10

    # --- EMA slopes ---
    ema5_rising    = ema5 > ema5.shift(1)
    ema9_rising    = ema9 > ema9.shift(1)
    ema15_rising   = ema15 > ema15.shift(1)

    # ═══════════════════════════════════════════════════════════════
    # ╔═══════════════════════════════════════════════════════════╗
    # ║  ★★★ MASTER FILTER: 2 GREEN CANDLES + OPEN/HIGH ★★★     ║
    # ║                                                           ║
    # ║  This condition MUST pass for ANY trade signal:           ║
    # ║    1. Current candle (i) is GREEN (Close > Open)          ║
    # ║    2. Previous candle (i-1) is GREEN (Close > Open)       ║
    # ║    3. Current Open > Previous Open                        ║
    # ║    4. Current High > Previous High                        ║
    # ╚═══════════════════════════════════════════════════════════╝
    # ═══════════════════════════════════════════════════════════════

    print("\n  🟩  Initializing Master Filter: 2 Green Candles + Open/High Check...")

    # Current candle is green
    curr_green = close > opn

    # Previous candle is green
    prev_candle_green = close.shift(1) > opn.shift(1)

    # Both candles must be green
    two_green_candles = curr_green & prev_candle_green

    # Current candle Open > Previous candle Open
    curr_open_gt_prev_open = opn > opn.shift(1)

    # Current candle High > Previous candle High
    curr_high_gt_prev_high = high > high.shift(1)

    # ── MASTER FILTER VERDICT ──
    MASTER_FILTER = (
        two_green_candles &
        curr_open_gt_prev_open &
        curr_high_gt_prev_high
    )

    data['master_filter'] = MASTER_FILTER.astype(int)

    # ── MASTER FILTER STATUS LOG ──
    last_idx_mf = len(data) - 1
    print("\n  ─── MASTER FILTER: 2 GREEN CANDLES + OPEN/HIGH CHECK ───")
    print(f"    Current candle GREEN:        {'✅' if curr_green.iloc[last_idx_mf] else '❌'}"
          f"  (O={opn.iloc[last_idx_mf]:.2f} C={close.iloc[last_idx_mf]:.2f})")
    print(f"    Previous candle GREEN:       {'✅' if prev_candle_green.iloc[last_idx_mf] else '❌'}"
          f"  (O={opn.shift(1).iloc[last_idx_mf]:.2f} C={close.shift(1).iloc[last_idx_mf]:.2f})" if last_idx_mf > 0 else "")
    print(f"    Curr Open > Prev Open:       {'✅' if curr_open_gt_prev_open.iloc[last_idx_mf] else '❌'}"
          f"  ({opn.iloc[last_idx_mf]:.2f} > {opn.shift(1).iloc[last_idx_mf]:.2f})" if last_idx_mf > 0 else "")
    print(f"    Curr High > Prev High:       {'✅' if curr_high_gt_prev_high.iloc[last_idx_mf] else '❌'}"
          f"  ({high.iloc[last_idx_mf]:.2f} > {high.shift(1).iloc[last_idx_mf]:.2f})" if last_idx_mf > 0 else "")
    print(f"    MASTER FILTER VERDICT:       {'🟢 PASS' if MASTER_FILTER.iloc[last_idx_mf] else '🔴 FAIL'}")

    # ═══════════════════════════════════════════════════════════════
    #  TIME FILTER
    # ═══════════════════════════════════════════════════════════════
    if isinstance(data.index, pd.DatetimeIndex):
        time_series = data.index.time
    else:
        time_series = pd.to_datetime(data.index).time

    data['time_mins'] = [t.hour * 60 + t.minute for t in time_series]
    TIME_OK = (data['time_mins'] >= 575) & (data['time_mins'] <= 910)

    # ═══════════════════════════════════════════════════════════════
    # ╔═══════════════════════════════════════════════════════════╗
    # ║              STEP 1: OVERSOLD DETECTION                  ║
    # ║     "Is the system armed? Are K and D both under 20?"    ║
    # ╚═══════════════════════════════════════════════════════════╝
    # ═══════════════════════════════════════════════════════════════

    k_below_20 = K < 20
    d_below_20 = D < 20
    both_below_20_now = k_below_20 & d_below_20

    OVERSOLD_LOOKBACK = 12

    was_both_below_20 = pd.Series(False, index=data.index)
    for i in range(0, OVERSOLD_LOOKBACK + 1):
        was_both_below_20 |= (K.shift(i) < 20) & (D.shift(i) < 20)

    bars_both_below_20 = pd.Series(0, index=data.index, dtype=float)
    for i in range(0, OVERSOLD_LOOKBACK + 1):
        bars_both_below_20 += ((K.shift(i) < 20) & (D.shift(i) < 20)).astype(int)

    deep_oversold = bars_both_below_20 >= 2

    STEP_1_ARMED = was_both_below_20 & deep_oversold
    data['step_1_armed'] = STEP_1_ARMED.astype(int)

    # ═══════════════════════════════════════════════════════════════
    # ╔═══════════════════════════════════════════════════════════╗
    # ║         STEP 2: REVERSAL CONFIRMATION (V-SHAPE)          ║
    # ║   "K crossed above D AND both K,D are turning up"        ║
    # ╚═══════════════════════════════════════════════════════════╝
    # ═══════════════════════════════════════════════════════════════

    prev_k_below_d  = K.shift(1) < D.shift(1)
    prev_k_equal_d  = K.shift(1) == D.shift(1)
    curr_k_above_d  = K > D
    k_crossover_d   = (prev_k_below_d | prev_k_equal_d) & curr_k_above_d

    recent_crossover = pd.Series(False, index=data.index)
    for i in range(0, 4):
        shifted_cross = k_crossover_d.shift(i).fillna(False)
        recent_crossover |= shifted_cross

    k_turning_up = (K > K.shift(1)) & (K.shift(1) <= K.shift(2))
    d_turning_up = (D > D.shift(1)) & (D.shift(1) <= D.shift(2))

    k_rising = K > K.shift(1)
    d_rising = D > D.shift(1)
    both_rising = k_rising & d_rising

    k_was_falling = K.shift(2) > K.shift(1)
    k_now_rising  = K > K.shift(1)
    v_shape_k     = k_was_falling & k_now_rising

    d_was_falling = D.shift(2) > D.shift(1)
    d_now_rising  = D > D.shift(1)
    v_shape_d     = d_was_falling & d_now_rising

    v_reversal = (v_shape_k | v_shape_d) & both_rising

    k_above_d = K > D

    STEP_2_REVERSAL = recent_crossover & both_rising & k_above_d
    data['step_2_reversal'] = STEP_2_REVERSAL.astype(int)

    # ═══════════════════════════════════════════════════════════════
    # ╔═══════════════════════════════════════════════════════════╗
    # ║       STEP 3: MOMENTUM, STRENGTH & VELOCITY CHECK        ║
    # ╚═══════════════════════════════════════════════════════════╝
    # ═══════════════════════════════════════════════════════════════

    k_speed      = k_vel.abs()
    d_speed      = d_vel.abs()
    price_speed  = price_change_pct.abs()

    k_not_too_fast   = k_speed < 15
    d_not_too_fast   = d_speed < 12
    kd_controlled    = k_not_too_fast & d_not_too_fast

    price_has_pulse  = price_speed > 0.02

    kd_avg_speed     = (k_speed + d_speed) / 2
    velocity_ratio   = price_speed / kd_avg_speed.replace(0, float('nan'))
    velocity_ratio   = velocity_ratio.fillna(0)

    slow_kd_vs_price = (kd_avg_speed < 10) | (velocity_ratio > 0.01)

    price_momentum_3  = close > close.shift(3)
    price_momentum_1  = close > close.shift(1)

    k_momentum        = k_vel > 0
    k_positive_accel  = k_accel >= 0
    d_not_falling     = d_vel >= 0

    is_green = green_candle
    above_ema5 = close > ema5
    vol_ok = vol_rising | vol_above_avg
    decent_body = body_ratio > 0.3

    k_has_room = K < 80
    k_not_dead_low = K > 8

    STEP_3_MOMENTUM = (
        kd_controlled &
        price_has_pulse &
        slow_kd_vs_price &
        price_momentum_1 &
        k_momentum &
        d_not_falling &
        is_green &
        k_has_room &
        k_not_dead_low
    )

    STEP_3_MOMENTUM_RELAXED = (
        kd_controlled &
        price_has_pulse &
        k_momentum &
        is_green &
        k_has_room
    )

    data['step_3_momentum'] = STEP_3_MOMENTUM.astype(int)

    # ═══════════════════════════════════════════════════════════════
    # ╔═══════════════════════════════════════════════════════════╗
    # ║    STEP 4: MID-ZONE CROSSOVER (Alternative Entry Path)   ║
    # ╚═══════════════════════════════════════════════════════════╝
    # ═══════════════════════════════════════════════════════════════

    k_in_mid_zone     = (K > 20) & (K < 80)
    d_in_mid_zone     = (D > 20) & (D < 80)
    mid_zone           = k_in_mid_zone & d_in_mid_zone

    mid_zone_crossover = k_crossover_d & mid_zone

    recent_mid_cross   = pd.Series(False, index=data.index)
    for i in range(0, 3):
        recent_mid_cross |= mid_zone_crossover.shift(i).fillna(False)

    ema_support_count  = ema5_rising.astype(int) + ema9_rising.astype(int) + ema15_rising.astype(int)
    ema_support        = ema_support_count >= 2

    price_above_ema9   = close > ema9

    STEP_4_MID_ZONE = (
        TIME_OK &
        recent_mid_cross &
        k_above_d &
        both_rising &
        STEP_3_MOMENTUM_RELAXED &
        ema_support &
        price_above_ema9 &
        decent_body &
        ~STEP_1_ARMED
    )

    data['step_4_midzone'] = STEP_4_MID_ZONE.astype(int)

    # ═══════════════════════════════════════════════════════════════
    # ╔═══════════════════════════════════════════════════════════╗
    # ║  STEP 5: BULLISH CANDLE PATTERN CONFIRMATION ★NEW★       ║
    # ╚═══════════════════════════════════════════════════════════╝
    # ═══════════════════════════════════════════════════════════════

    print("\n  🕯️  Initializing Bullish Candle Pattern Detection Engine...")

    # ─── HELPER CALCULATIONS FOR PATTERNS ───
    upper_shadow = high - close.where(close >= opn, opn)
    lower_shadow = close.where(close <= opn, opn) - low
    upper_shadow = high - pd.concat([close, opn], axis=1).max(axis=1)
    lower_shadow = pd.concat([close, opn], axis=1).min(axis=1) - low

    candle_body_raw = close - opn
    candle_body_abs = candle_body
    candle_range_safe = candle_range.replace(0, float('nan'))

    # Previous candle info
    prev_close  = close.shift(1)
    prev_open   = opn.shift(1)
    prev_high   = high.shift(1)
    prev_low    = low.shift(1)
    prev_body   = abs(prev_close - prev_open)
    prev_range  = prev_high - prev_low
    prev_green  = prev_close > prev_open
    prev_red    = prev_close < prev_open

    # 2-bars-ago candle info
    prev2_close = close.shift(2)
    prev2_open  = opn.shift(2)
    prev2_high  = high.shift(2)
    prev2_low   = low.shift(2)
    prev2_body  = abs(prev2_close - prev2_open)
    prev2_green = prev2_close > prev2_open
    prev2_red   = prev2_close < prev2_open

    # ─── PATTERN 1: SIMPLE GREEN (BULLISH) CANDLE ───
    simple_green = (
        green_candle &
        (candle_body_abs > candle_range_safe * 0.3)
    )

    # ─── PATTERN 2: BULLISH ENGULFING ───
    bullish_engulfing = (
        green_candle &
        prev_red &
        (opn <= prev_close) &
        (close >= prev_open) &
        (candle_body_abs > prev_body) &
        (candle_body_abs > candle_range_safe * 0.4)
    )

    # ─── PATTERN 3: HAMMER ───
    hammer = (
        (lower_shadow >= candle_body_abs * 2) &
        (upper_shadow <= candle_body_abs * 0.3) &
        (candle_body_abs > 0) &
        (candle_body_abs <= candle_range_safe * 0.35) &
        (lower_shadow >= candle_range_safe * 0.6)
    )

    # ─── PATTERN 4: INVERTED HAMMER ───
    inverted_hammer = (
        (upper_shadow >= candle_body_abs * 2) &
        (lower_shadow <= candle_body_abs * 0.3) &
        (candle_body_abs > 0) &
        (candle_body_abs <= candle_range_safe * 0.35) &
        prev_red
    )

    # ─── PATTERN 5: MORNING STAR ───
    prev2_midpoint = (prev2_open + prev2_close) / 2
    morning_star = (
        prev2_red &
        (prev2_body > prev2_high.sub(prev2_low).replace(0, float('nan')) * 0.5) &
        (prev_body < prev_range.replace(0, float('nan')) * 0.3) &
        green_candle &
        (candle_body_abs > candle_range_safe * 0.4) &
        (close > prev2_midpoint)
    )

    # ─── PATTERN 6: PIERCING LINE ───
    prev_midpoint = (prev_open + prev_close) / 2
    piercing_line = (
        prev_red &
        green_candle &
        (opn < prev_close) &
        (close > prev_midpoint) &
        (close < prev_open) &
        (candle_body_abs > candle_range_safe * 0.4)
    )

    # ─── PATTERN 7: THREE WHITE SOLDIERS ───
    three_white_soldiers = (
        green_candle &
        prev_green &
        prev2_green &
        (close > prev_close) &
        (prev_close > prev2_close) &
        (opn >= prev_open) & (opn <= prev_close) &
        (prev_open >= prev2_open) & (prev_open <= prev2_close) &
        (candle_body_abs > candle_range_safe * 0.5) &
        (prev_body > prev_range.replace(0, float('nan')) * 0.5)
    )

    # ─── PATTERN 8: BULLISH MARUBOZU ───
    bullish_marubozu = (
        green_candle &
        (upper_shadow <= candle_range_safe * 0.05) &
        (lower_shadow <= candle_range_safe * 0.05) &
        (candle_body_abs >= candle_range_safe * 0.9)
    )

    # ─── PATTERN 9: BULLISH HARAMI ───
    bullish_harami = (
        prev_red &
        green_candle &
        (opn > prev_close) &
        (close < prev_open) &
        (candle_body_abs < prev_body * 0.5) &
        (prev_body > prev_range.replace(0, float('nan')) * 0.5)
    )

    # ─── PATTERN 10: DRAGONFLY DOJI ───
    dragonfly_doji = (
        (candle_body_abs <= candle_range_safe * 0.1) &
        (lower_shadow >= candle_range_safe * 0.7) &
        (upper_shadow <= candle_range_safe * 0.1) &
        (candle_range > 0)
    )

    # ─── PATTERN 11: TWEEZER BOTTOM ───
    low_tolerance = candle_range_safe * 0.02
    tweezer_bottom = (
        prev_red &
        green_candle &
        (abs(low - prev_low) <= low_tolerance) &
        (candle_body_abs > candle_range_safe * 0.3)
    )

    # ═══════════════════════════════════════════════════════════════
    #  COMBINE ALL BULLISH PATTERNS
    # ═══════════════════════════════════════════════════════════════

    ANY_BULLISH_PATTERN = (
        simple_green |
        bullish_engulfing |
        hammer |
        inverted_hammer |
        morning_star |
        piercing_line |
        three_white_soldiers |
        bullish_marubozu |
        bullish_harami |
        dragonfly_doji |
        tweezer_bottom
    )

    # ─── PATTERN NAME TRACKING ───
    data['bullish_pattern_name'] = ""
    data.loc[bullish_engulfing,       'bullish_pattern_name'] = "BULLISH_ENGULFING"
    data.loc[hammer,                  'bullish_pattern_name'] = "HAMMER"
    data.loc[inverted_hammer,         'bullish_pattern_name'] = "INVERTED_HAMMER"
    data.loc[morning_star,            'bullish_pattern_name'] = "MORNING_STAR"
    data.loc[piercing_line,           'bullish_pattern_name'] = "PIERCING_LINE"
    data.loc[three_white_soldiers,    'bullish_pattern_name'] = "THREE_WHITE_SOLDIERS"
    data.loc[bullish_marubozu,        'bullish_pattern_name'] = "BULLISH_MARUBOZU"
    data.loc[bullish_harami,          'bullish_pattern_name'] = "BULLISH_HARAMI"
    data.loc[dragonfly_doji,          'bullish_pattern_name'] = "DRAGONFLY_DOJI"
    data.loc[tweezer_bottom,          'bullish_pattern_name'] = "TWEEZER_BOTTOM"
    mask_only_green = simple_green & (data['bullish_pattern_name'] == "")
    data.loc[mask_only_green,         'bullish_pattern_name'] = "BULLISH_CANDLE"

    # ═══════════════════════════════════════════════════════════════
    #  STEP 5 CONDITIONS
    # ═══════════════════════════════════════════════════════════════

    step5_oversold = was_both_below_20
    step5_k_cross_d = recent_crossover
    k_minus_d = K - D
    step5_kd_diff_gt_5 = k_minus_d > 5
    step5_bullish_candle = ANY_BULLISH_PATTERN

    recent_bullish_pattern = pd.Series(False, index=data.index)
    for i in range(0, 3):
        recent_bullish_pattern |= ANY_BULLISH_PATTERN.shift(i).fillna(False)

    STEP_5_BULLISH_PATTERN = (
        TIME_OK &
        step5_oversold &
        step5_k_cross_d &
        step5_kd_diff_gt_5 &
        (step5_bullish_candle | recent_bullish_pattern) &
        k_above_d &
        (K < 80)
    )

    data['step_5_bullish_pattern'] = STEP_5_BULLISH_PATTERN.astype(int)
    data['k_minus_d'] = k_minus_d.round(2)

    # ═══════════════════════════════════════════════════════════════
    #  STEP 5 STATUS LOGGING
    # ═══════════════════════════════════════════════════════════════

    last_idx = len(data) - 1
    print("\n  ─── STEP 5: BULLISH CANDLE PATTERN CHECK ───")
    print(f"    Oversold (was K&D<20):   {'✅' if step5_oversold.iloc[last_idx] else '⬜'}")
    print(f"    K crossed D:             {'✅' if step5_k_cross_d.iloc[last_idx] else '⬜'}")
    print(f"    K - D = {k_minus_d.iloc[last_idx]:.2f}  (>5 needed): {'✅' if step5_kd_diff_gt_5.iloc[last_idx] else '⬜'}")
    print(f"    Bullish Pattern:         {'✅ ' + data['bullish_pattern_name'].iloc[last_idx] if ANY_BULLISH_PATTERN.iloc[last_idx] else '⬜ NONE'}")
    print(f"    STEP 5 VERDICT:          {'🟢 ACTIVE' if STEP_5_BULLISH_PATTERN.iloc[last_idx] else '⚪ INACTIVE'}")

    # ═══════════════════════════════════════════════════════════════
    # ╔═══════════════════════════════════════════════════════════╗
    # ║         TRADE DECISION ENGINE — FINAL SIGNAL             ║
    # ╚═══════════════════════════════════════════════════════════╝
    # ═══════════════════════════════════════════════════════════════

    # --- PATH A: Oversold Recovery (Steps 1 + 2 + 3) ---
    PATH_A_OVERSOLD = (
        TIME_OK &
        STEP_1_ARMED &
        STEP_2_REVERSAL &
        STEP_3_MOMENTUM
    )

    # --- PATH B: Mid-Zone Crossover (Step 4) ---
    PATH_B_MIDZONE = STEP_4_MID_ZONE

    # --- PATH C: Bullish Candle Pattern Entry (Step 5) ---
    PATH_C_BULLISH_PATTERN = STEP_5_BULLISH_PATTERN

    # --- REJECTION OVERRIDES (kill switch) ---
    REJECT_OVERBOUGHT  = K > 85
    REJECT_K_CRASHING  = k_vel < -5
    REJECT_DEAD_VOLUME = volume < (vol_avg_10 * 0.3)
    REJECT_HUGE_K_JUMP = k_speed > 25

    REJECTION = REJECT_OVERBOUGHT | REJECT_K_CRASHING | REJECT_DEAD_VOLUME | REJECT_HUGE_K_JUMP

    # ═══════════════════════════════════════════════════════════════
    # ╔═══════════════════════════════════════════════════════════╗
    # ║  ★★★ APPLY MASTER FILTER TO ALL PATHS ★★★               ║
    # ║                                                           ║
    # ║  EVERY trade signal MUST have:                            ║
    # ║    ✓ Previous 2 candles GREEN                             ║
    # ║    ✓ Current Open > Previous Open                         ║
    # ║    ✓ Current High > Previous High                         ║
    # ╚═══════════════════════════════════════════════════════════╝
    # ═══════════════════════════════════════════════════════════════

    # --- COMBINED RAW SIGNAL (ALL paths filtered through MASTER_FILTER) ---
    long_raw = (
        MASTER_FILTER &                          # ★ MANDATORY for ALL trades ★
        (PATH_A_OVERSOLD | PATH_B_MIDZONE | PATH_C_BULLISH_PATTERN) &
        ~REJECTION
    ).astype(int)

    # ═══════════════════════════════════════════════════════════════
    #  SIGNAL GRADING
    # ═══════════════════════════════════════════════════════════════

    data['signal_grade'] = ""
    data['signal_path']  = ""

    # --- A+ Grade ---
    grade_aplus = (
        (long_raw == 1) &
        PATH_A_OVERSOLD &
        ema5_rising &
        above_ema5 &
        vol_ok &
        v_reversal &
        decent_body
    )
    data.loc[grade_aplus, 'signal_grade'] = "A+"
    data.loc[grade_aplus, 'signal_path']  = "OVERSOLD_RECOVERY"

    # --- A Grade ---
    grade_a = (
        (long_raw == 1) &
        PATH_A_OVERSOLD &
        (data['signal_grade'] == "")
    )
    data.loc[grade_a, 'signal_grade'] = "A"
    data.loc[grade_a, 'signal_path']  = "OVERSOLD_RECOVERY"

    # --- A Grade (Path C) ---
    grade_a_pattern = (
        (long_raw == 1) &
        PATH_C_BULLISH_PATTERN &
        (
            bullish_engulfing |
            morning_star |
            three_white_soldiers |
            bullish_marubozu
        ) &
        vol_ok &
        (data['signal_grade'] == "")
    )
    data.loc[grade_a_pattern, 'signal_grade'] = "A"
    data.loc[grade_a_pattern, 'signal_path']  = "BULLISH_PATTERN_STRONG"

    # --- B+ Grade ---
    grade_bplus = (
        (long_raw == 1) &
        PATH_B_MIDZONE &
        vol_ok &
        above_ema5 &
        (data['signal_grade'] == "")
    )
    data.loc[grade_bplus, 'signal_grade'] = "B+"
    data.loc[grade_bplus, 'signal_path']  = "MIDZONE_CROSSOVER"

    # --- B+ Grade (Path C) ---
    grade_bplus_pattern = (
        (long_raw == 1) &
        PATH_C_BULLISH_PATTERN &
        vol_ok &
        (data['signal_grade'] == "")
    )
    data.loc[grade_bplus_pattern, 'signal_grade'] = "B+"
    data.loc[grade_bplus_pattern, 'signal_path']  = "BULLISH_PATTERN"

    # --- B Grade ---
    grade_b = (
        (long_raw == 1) &
        (data['signal_grade'] == "")
    )
    data.loc[grade_b, 'signal_grade'] = "B"
    mask_b_pathc = grade_b & PATH_C_BULLISH_PATTERN
    mask_b_pathb = grade_b & PATH_B_MIDZONE & ~PATH_C_BULLISH_PATTERN
    mask_b_other = grade_b & ~PATH_C_BULLISH_PATTERN & ~PATH_B_MIDZONE
    data.loc[mask_b_pathc, 'signal_path'] = "BULLISH_PATTERN"
    data.loc[mask_b_pathb, 'signal_path'] = "MIDZONE_CROSSOVER"
    data.loc[mask_b_other, 'signal_path'] = "MIXED"

    # ═══════════════════════════════════════════════════════════════
    #  SIGNAL REASONS — DETAILED LOGGING
    # ═══════════════════════════════════════════════════════════════

    data["signal_reason"] = ""

    # Path A reasons
    mask_path_a = PATH_A_OVERSOLD & (long_raw == 1)
    data.loc[mask_path_a, "signal_reason"] = (
        "BUY: OVERSOLD RECOVERY | "
        "Step1:ARMED ✓ Step2:REVERSAL ✓ Step3:MOMENTUM ✓"
        " | MasterFilter:2GREEN+O/H ✓"
    )

    # Path B reasons
    mask_path_b = PATH_B_MIDZONE & (long_raw == 1) & (data["signal_reason"] == "")
    data.loc[mask_path_b, "signal_reason"] = (
        "BUY: MIDZONE CROSSOVER | "
        "Step4:MIDZONE ✓ Momentum ✓ Velocity ✓"
        " | MasterFilter:2GREEN+O/H ✓"
    )

    # Path C reasons
    mask_path_c = PATH_C_BULLISH_PATTERN & (long_raw == 1) & (data["signal_reason"] == "")
    data.loc[mask_path_c, "signal_reason"] = (
        "BUY: BULLISH CANDLE PATTERN | "
        "Step5:OVERSOLD+K×D+(K-D>5)+PATTERN ✓"
        " | MasterFilter:2GREEN+O/H ✓"
    )

    # Append pattern name for Path C
    mask_path_c_all = PATH_C_BULLISH_PATTERN & (long_raw == 1)
    data.loc[mask_path_c_all, 'signal_reason'] += (
        " Pattern=" + data['bullish_pattern_name'] +
        " K-D=" + k_minus_d.round(1).astype(str)
    )

    # Overlap signals
    mask_both_ac = PATH_A_OVERSOLD & PATH_C_BULLISH_PATTERN & (long_raw == 1)
    data.loc[mask_both_ac, 'signal_reason'] += (
        " + BULLISH_PATTERN_BONUS(" + data['bullish_pattern_name'] + ")"
    )

    # Grade stars
    data.loc[data['signal_grade'] == "A+", 'signal_reason'] += " [★★★★★]"
    data.loc[data['signal_grade'] == "A",  'signal_reason'] += " [★★★★]"
    data.loc[data['signal_grade'] == "B+", 'signal_reason'] += " [★★★]"
    data.loc[data['signal_grade'] == "B",  'signal_reason'] += " [★★]"

    # Append K/D details
    mask_any_signal = (long_raw == 1)
    data.loc[mask_any_signal, 'signal_reason'] += (
        "  K=" + K.round(1).astype(str) +
        " D=" + D.round(1).astype(str) +
        " Kvel=" + k_vel.round(1).astype(str) +
        " Dvel=" + d_vel.round(1).astype(str) +
        " PriceChg=" + price_change_pct.round(3).astype(str) + "%"
    )

    # ═══════════════════════════════════════════════════════════════
    #  COOLDOWN — Prevent signal spam
    # ═══════════════════════════════════════════════════════════════
    cooldown = 5
    long_recent = long_raw.shift(1).rolling(cooldown).sum().fillna(0)

    strong_signal = data['signal_grade'].isin(['A', 'A+'])
    priority_pattern = data['bullish_pattern_name'].isin([
        'MORNING_STAR',
        'BULLISH_ENGULFING',
        'THREE_WHITE_SOLDIERS',
        'BULLISH_MARUBOZU'
    ])

    override = strong_signal | priority_pattern

    long_after_cooldown = (
            (long_raw == 1) &
            (
                    (long_recent == 0) |
                    override
            )
    ).astype(int)

    # ═══════════════════════════════════════════════════════════════
    #  MACHINE STATUS DASHBOARD
    # ═══════════════════════════════════════════════════════════════

    last = len(data) - 1

    print("\n" + "═" * 70)
    print("  📟  5-STEP VERIFICATION MACHINE — STATUS DASHBOARD")
    print("═" * 70)
    print(f"  Symbol:       {symbol}")
    print(f"  Close:        {close.iloc[last]:.2f}")
    print(f"  K:            {K.iloc[last]:.2f}")
    print(f"  D:            {D.iloc[last]:.2f}")
    print(f"  K - D:        {k_minus_d.iloc[last]:.2f}")
    print(f"  K velocity:   {k_vel.iloc[last]:.2f}")
    print(f"  D velocity:   {d_vel.iloc[last]:.2f}")
    print(f"  Price Chg%:   {price_change_pct.iloc[last]:.3f}%")
    print("─" * 70)

    step1_status  = "✅ ARMED" if STEP_1_ARMED.iloc[last] else "⬜ NOT ARMED"
    step2_status  = "✅ REVERSAL" if STEP_2_REVERSAL.iloc[last] else "⬜ NO REVERSAL"
    step3_status  = "✅ MOMENTUM OK" if STEP_3_MOMENTUM.iloc[last] else "⬜ NO MOMENTUM"
    step4_status  = "✅ MID-ZONE" if STEP_4_MID_ZONE.iloc[last] else "⬜ NO MID-ZONE"
    step5_status  = "✅ BULLISH PATTERN" if STEP_5_BULLISH_PATTERN.iloc[last] else "⬜ NO PATTERN"
    master_status = "🟢 PASS" if MASTER_FILTER.iloc[last] else "🔴 FAIL"
    patha_status  = "🟢 ACTIVE" if PATH_A_OVERSOLD.iloc[last] else "⚪ INACTIVE"
    pathb_status  = "🟢 ACTIVE" if PATH_B_MIDZONE.iloc[last] else "⚪ INACTIVE"
    pathc_status  = "🟢 ACTIVE" if PATH_C_BULLISH_PATTERN.iloc[last] else "⚪ INACTIVE"
    reject_status = "🔴 REJECTED" if REJECTION.iloc[last] else "🟢 CLEAR"

    print(f"  ★ MASTER FILTER (2 Green+O/H): {master_status}")
    if not MASTER_FILTER.iloc[last]:
        reasons = []
        if not curr_green.iloc[last]:
            reasons.append("Current candle RED")
        if not prev_candle_green.iloc[last]:
            reasons.append("Previous candle RED")
        if not curr_open_gt_prev_open.iloc[last]:
            reasons.append(f"Open({opn.iloc[last]:.2f}) <= PrevOpen({opn.shift(1).iloc[last]:.2f})")
        if not curr_high_gt_prev_high.iloc[last]:
            reasons.append(f"High({high.iloc[last]:.2f}) <= PrevHigh({high.shift(1).iloc[last]:.2f})")
        print(f"    Fail reasons: {', '.join(reasons)}")
    print("─" * 70)
    print(f"  STEP 1 (Oversold K&D<20):     {step1_status}")
    print(f"  STEP 2 (K×D + V-shape):       {step2_status}")
    print(f"  STEP 3 (Momentum/Velocity):   {step3_status}")
    print(f"  STEP 4 (Mid-Zone Crossover):  {step4_status}")
    print(f"  STEP 5 (Bullish Pattern):     {step5_status}")
    if STEP_5_BULLISH_PATTERN.iloc[last]:
        print(f"         Pattern Found:         🕯️  {data['bullish_pattern_name'].iloc[last]}")
    print("─" * 70)
    print(f"  PATH A (Oversold Recovery):   {patha_status}")
    print(f"  PATH B (Mid-Zone Entry):      {pathb_status}")
    print(f"  PATH C (Bullish Pattern):     {pathc_status}")
    print(f"  REJECTION CHECK:              {reject_status}")
    print("─" * 70)

    if long_raw.iloc[last] == 1:
        print(f"  🔔 RAW SIGNAL:   BUY  (Master Filter: ✅)")
        print(f"     Grade:        {data['signal_grade'].iloc[last]}")
        print(f"     Path:         {data['signal_path'].iloc[last]}")
    else:
        if not MASTER_FILTER.iloc[last] and (PATH_A_OVERSOLD.iloc[last] or PATH_B_MIDZONE.iloc[last] or PATH_C_BULLISH_PATTERN.iloc[last]):
            print(f"  ⚠️  RAW SIGNAL:  BLOCKED BY MASTER FILTER (2 Green Candle + O/H condition)")
        else:
            print(f"  💤 RAW SIGNAL:   NO TRADE")

    if long_after_cooldown.iloc[last] == 1:
        print(f"  ⏱️  COOLDOWN:     CLEAR — Signal passes")
    elif long_raw.iloc[last] == 1:
        print(f"  ⏱️  COOLDOWN:     BLOCKED — Too soon after last signal")

    print("═" * 70 + "\n")

    # ═══════════════════════════════════════════════════════════════
    #  INIT OUTPUT COLUMNS
    # ═══════════════════════════════════════════════════════════════

    data['st_sig_raw']    = long_after_cooldown
    data['confidence']    = 0.0
    data['llm_signal']    = None
    data['llm_reason']    = ""
    data['llm_entry']     = 0.0
    data['llm_stop_loss'] = 0.0
    data['llm_target_1']  = 0.0
    data['llm_target_2']  = 0.0
    data['llm_bias']      = ""

    # ═══════════════════════════════════════════════════════════════
    #  MODE: WITHOUT LLM
    # ═══════════════════════════════════════════════════════════════

    if not use_llm:
        print("📊 LLM Mode: OFF — Pure technical signals")
        data['st_sig'] = long_after_cooldown
        data.loc[data['st_sig'] == 1, 'signal_reason'] += " | 🔧 NO LLM"
        return data

    # ═══════════════════════════════════════════════════════════════
    #  MODE: WITH LLM
    # ═══════════════════════════════════════════════════════════════

    print("🤖 LLM Mode: ON — Final AI verification")

    llm_confidence = 0.0
    llm_signal     = None
    llm_reason     = ""
    llm_entry      = 0.0
    llm_stop_loss  = 0.0
    llm_target_1   = 0.0
    llm_target_2   = 0.0
    llm_bias       = ""

    if long_after_cooldown.iloc[-1] == 1:
        print("=" * 60)
        print(f"🔔 5-STEP VERIFIED BUY SIGNAL → Sending to LLM: {symbol}")
        print(f"   Grade:  {data['signal_grade'].iloc[-1]}")
        print(f"   Path:   {data['signal_path'].iloc[-1]}")
        print(f"   Reason: {data['signal_reason'].iloc[-1]}")
        print("=" * 60)

        try:
            ohlcv = fetch_ohlcv(symbol)

            if not ohlcv:
                print("⚠️  No OHLCV data for LLM — proceeding with technical only")
            else:
                llm_result = llm_trade_signal(symbol, ohlcv, "3m")

                print("\n" + "─" * 50)
                print("  🧠  LLM ANALYSIS RESULT")
                print("─" * 50)

                llm_signal     = str(llm_result.get("signal", "no trade")).strip()
                llm_confidence = float(llm_result.get("confidence", 0.0))
                llm_reason     = str(llm_result.get("reason", ""))
                llm_bias       = str(llm_result.get("bias", ""))
                llm_entry      = float(llm_result.get("entry", 0.0))
                llm_stop_loss  = float(llm_result.get("stop_loss", 0.0))
                llm_target_1   = float(llm_result.get("target_1", 0.0))
                llm_target_2   = float(llm_result.get("target_2", 0.0))

                print(f"  Bias:       {llm_bias}")
                print(f"  Signal:     {llm_signal}")
                print(f"  Confidence: {llm_confidence}")
                print(f"  Entry:      {llm_entry}")
                print(f"  Stop Loss:  {llm_stop_loss}")
                print(f"  Target 1:   {llm_target_1}")
                print(f"  Target 2:   {llm_target_2}")
                print(f"  Reason:     {llm_reason}")

                if llm_signal == "buy" and llm_confidence >= 0.7:
                    print(f"\n  ✅ LLM CONFIRMS BUY (confidence={llm_confidence:.2f})")
                elif llm_signal == "buy":
                    print(f"\n  ⚠️  LLM says BUY but LOW confidence ({llm_confidence:.2f} < 0.7) — BLOCKED")
                elif llm_signal == "sell":
                    print(f"\n  ❌ LLM says SELL — BLOCKED")
                else:
                    print(f"\n  ⏸️  LLM says '{llm_signal}' — BLOCKED")

                print("─" * 50)

        except Exception as e:
            print(f"❌ LLM error: {e}")
            print("   Proceeding without LLM confirmation")

    # Store LLM values
    data.loc[data.index[-1], 'confidence']    = llm_confidence
    data.loc[data.index[-1], 'llm_signal']    = llm_signal
    data.loc[data.index[-1], 'llm_reason']    = llm_reason
    data.loc[data.index[-1], 'llm_bias']      = llm_bias
    data.loc[data.index[-1], 'llm_entry']     = llm_entry
    data.loc[data.index[-1], 'llm_stop_loss'] = llm_stop_loss
    data.loc[data.index[-1], 'llm_target_1']  = llm_target_1
    data.loc[data.index[-1], 'llm_target_2']  = llm_target_2

    # ═══════════════════════════════════════════════════════════════
    #  FINAL TRADE SIGNAL
    # ═══════════════════════════════════════════════════════════════

    data['st_sig'] = (
        (long_after_cooldown == 1) &
        (data['confidence'] >= 0.7) &
        (data['llm_signal'] == 'buy')
    ).astype(int)

    # Tag final reasons
    final_buy = data['st_sig'] == 1
    data.loc[final_buy, 'signal_reason'] += (
        f" | ✅ LLM=BUY conf={llm_confidence:.2f}"
        f" entry={llm_entry} sl={llm_stop_loss}"
        f" t1={llm_target_1} t2={llm_target_2}"
    )

    blocked = (long_after_cooldown == 1) & (data['st_sig'] == 0)
    data.loc[blocked, 'signal_reason'] += (
        f" | ❌ LLM BLOCKED ({llm_signal}, conf={llm_confidence:.2f})"
    )

    # ═══════════════════════════════════════════════════════════════
    #  FINAL MACHINE OUTPUT
    # ═══════════════════════════════════════════════════════════════

    if long_after_cooldown.iloc[-1] == 1:
        if data['st_sig'].iloc[-1] == 1:
            print("\n" + "🟢" * 20)
            print("  ╔═══════════════════════════════════════════╗")
            print("  ║     ✅  TRADE EXECUTED — BUY CONFIRMED   ║")
            print("  ╚═══════════════════════════════════════════╝")
            print(f"  Symbol:     {symbol}")
            print(f"  Grade:      {data['signal_grade'].iloc[-1]}")
            print(f"  Path:       {data['signal_path'].iloc[-1]}")
            print(f"  Confidence: {llm_confidence:.2f}")
            print(f"  Entry:      {llm_entry}")
            print(f"  Stop Loss:  {llm_stop_loss}")
            print(f"  Target 1:   {llm_target_1}")
            print(f"  Target 2:   {llm_target_2}")
            print(f"  🟩 Master:   2 Green Candles + O/H confirmed")
            if PATH_C_BULLISH_PATTERN.iloc[-1]:
                print(f"  🕯️ Pattern:  {data['bullish_pattern_name'].iloc[-1]}")
            print("🟢" * 20 + "\n")
        else:
            print("\n" + "🔴" * 20)
            print("  ╔═══════════════════════════════════════════╗")
            print("  ║     ❌  TRADE BLOCKED BY LLM             ║")
            print("  ╚═══════════════════════════════════════════╝")
            print(f"  Symbol:     {symbol}")
            print(f"  LLM Signal: {llm_signal}")
            print(f"  Confidence: {llm_confidence:.2f}")
            print(f"  Reason:     {llm_reason}")
            print("🔴" * 20 + "\n")

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
