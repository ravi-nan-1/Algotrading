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


def calculate_order_blocks(
        df: pd.DataFrame,
        open_column: str = "Open",
        high_column: str = "High",
        low_column: str = "Low",
        close_column: str = "Close",
        volume_column: str = "Volume",
        min_volume_percentile: float = 60,
        check_mitigation: bool = True,
        mitigation_threshold: float = 0.6,
        lookback_periods: int = 10,
        use_wicks: bool = True
) -> pd.DataFrame:
    """
    TRUE NON-REPAINTING ORDER BLOCK DETECTOR - LIVE CANDLE-BY-CANDLE FIXED
    Same detection logic & accuracy. Fixed mitigation + frozen percentile.

    Fixes vs previous version (detection logic unchanged):
      FIX-1: Volume percentile now computed as expanding-window rank so each
              bar's percentile is frozen to the history visible AT that bar,
              not the full dataset — eliminates live/backtest divergence.
      FIX-2: already_tagged stores datetime labels (not positional ints) so
              it stays correct when df is re-indexed or appended incrementally.
      FIX-3: Swing detection skips the last `swing_strength` bars entirely to
              avoid partial right-window confirmation on live tail candles.
    """

    df = df.copy()

    # ====================== DATETIME HANDLING ======================
    if df.index.name and str(df.index.name).lower() in ['datetime', 'date', 'time']:
        df.index = pd.to_datetime(df.index)
    elif 'Datetime' in df.columns:
        df = df.set_index('Datetime')
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    col_map = {open_column: 'Open', high_column: 'High', low_column: 'Low',
               close_column: 'Close', volume_column: 'Volume'}
    df = df.rename(columns=col_map)

    df['is_bullish_candle'] = df['Close'] > df['Open']
    df['is_bearish_candle'] = df['Close'] < df['Open']

    # ====================== STATE PRESERVATION (Live Safe) ======================
    state_cols = ['volume_percentile_static', 'swing_high', 'swing_low',
                  'ob_bullish_detected', 'ob_bearish_detected',
                  'ob_bullish_mitigated', 'ob_bearish_mitigated',
                  'ob_top', 'ob_bottom', 'ob_volume', 'ob_strength', 'ob_detected_time']

    for col in state_cols:
        if col not in df.columns:
            if 'detected' in col or 'mitigated' in col:
                df[col] = False
            elif col == 'ob_detected_time':
                df[col] = pd.NaT
            else:
                df[col] = np.nan

    if 'ob_detected_time' in df.columns:
     df['ob_detected_time'] = pd.to_datetime(
        df['ob_detected_time'],
        errors='coerce'
     )
    else:
       df['ob_detected_time'] = pd.NaT

    # ====================== FIX-1: Volume percentile frozen via expanding window ======================
    # Each bar's percentile is its rank within [bar_0 .. bar_i] — the only history
    # visible at that moment.  Previously rank() used the full df, so the same bar
    # would get a different percentile as new candles arrived → live/backtest gap.
    unprocessed_mask = pd.isna(df['volume_percentile_static'])
    if unprocessed_mask.any() and df['Volume'].sum() > 0:
        vol = df['Volume']
        for i in df.index[unprocessed_mask]:
            loc = df.index.get_loc(i)
            hist = vol.iloc[:loc + 1]
            # percentile = fraction of historical bars with volume <= current bar
            df.loc[i, 'volume_percentile_static'] = (hist <= hist.iloc[-1]).mean() * 100

    # ====================== SWING STRUCTURE ======================
    swing_strength = max(3, lookback_periods // 3)

    # FIX-3: Never process the last swing_strength bars — they lack a full
    # right window and would get confirmed prematurely in live mode.
    safe_end = len(df) - swing_strength  # exclusive upper bound

    for k in range(swing_strength, safe_end):
        if pd.isna(df['swing_high'].iloc[k]) and pd.isna(df['swing_low'].iloc[k]):
            left_highs = df['High'].iloc[k - swing_strength:k]
            right_highs = df['High'].iloc[k + 1:k + swing_strength + 1]
            if (df['High'].iloc[k] > left_highs.max() and
                    df['High'].iloc[k] >= right_highs.max()):
                df.loc[df.index[k], 'swing_high'] = df['High'].iloc[k]

            left_lows = df['Low'].iloc[k - swing_strength:k]
            right_lows = df['Low'].iloc[k + 1:k + swing_strength + 1]
            if (df['Low'].iloc[k] < left_lows.min() and
                    df['Low'].iloc[k] <= right_lows.min()):
                df.loc[df.index[k], 'swing_low'] = df['Low'].iloc[k]

    # ====================== FIX-2: already_tagged uses datetime labels ======================
    # Previously stored positional ints — these silently shift when df is
    # appended or re-indexed between live calls.
    already_tagged = set(
        df.index[df['ob_bullish_detected'] | df['ob_bearish_detected']]
    )

    # ====================== DETECT ORDER BLOCKS (logic unchanged) ======================
    for i in range(lookback_periods, len(df)):

        # --- Bullish OB ---
        broke_swing_high = any(
            not np.isnan(df['swing_high'].iloc[s]) and
            df['Close'].iloc[i] > df['swing_high'].iloc[s]
            for s in range(i - 1, max(i - lookback_periods * 4, swing_strength) - 1, -1)
        )
        if broke_swing_high:
            for j in range(1, lookback_periods + 1):
                ob_idx = i - j
                if ob_idx < 0:
                    continue
                ob_label = df.index[ob_idx]           # ← datetime label (FIX-2)
                if (ob_label in already_tagged or
                        df['ob_bullish_detected'].iloc[ob_idx]):
                    continue
                if (df['is_bearish_candle'].iloc[ob_idx] and
                        df['volume_percentile_static'].iloc[ob_idx] >= min_volume_percentile):

                    df.loc[ob_label, 'ob_bullish_detected'] = True
                    df.loc[ob_label, 'ob_detected_time'] = df.index[i]
                    if use_wicks:
                        df.loc[ob_label, 'ob_top'] = df['High'].iloc[ob_idx]
                        df.loc[ob_label, 'ob_bottom'] = df['Low'].iloc[ob_idx]
                    else:
                        df.loc[ob_label, 'ob_top'] = max(df['Open'].iloc[ob_idx],
                                                         df['Close'].iloc[ob_idx])
                        df.loc[ob_label, 'ob_bottom'] = min(df['Open'].iloc[ob_idx],
                                                            df['Close'].iloc[ob_idx])
                    df.loc[ob_label, 'ob_volume'] = df['Volume'].iloc[ob_idx]
                    price_move = (abs(df['Close'].iloc[ob_idx] - df['Open'].iloc[ob_idx]) /
                                  (df['Open'].iloc[ob_idx] + 1e-10))
                    vol_score = df['volume_percentile_static'].iloc[ob_idx] / 100
                    df.loc[ob_label, 'ob_strength'] = (price_move * 100 + vol_score) / 2
                    already_tagged.add(ob_label)      # ← store label (FIX-2)
                    break

        # --- Bearish OB ---
        broke_swing_low = any(
            not np.isnan(df['swing_low'].iloc[s]) and
            df['Close'].iloc[i] < df['swing_low'].iloc[s]
            for s in range(i - 1, max(i - lookback_periods * 4, swing_strength) - 1, -1)
        )
        if broke_swing_low:
            for j in range(1, lookback_periods + 1):
                ob_idx = i - j
                if ob_idx < 0:
                    continue
                ob_label = df.index[ob_idx]           # ← datetime label (FIX-2)
                if (ob_label in already_tagged or
                        df['ob_bearish_detected'].iloc[ob_idx]):
                    continue
                if (df['is_bullish_candle'].iloc[ob_idx] and
                        df['volume_percentile_static'].iloc[ob_idx] >= min_volume_percentile):

                    df.loc[ob_label, 'ob_bearish_detected'] = True
                    df.loc[ob_label, 'ob_detected_time'] = df.index[i]
                    if use_wicks:
                        df.loc[ob_label, 'ob_top'] = df['High'].iloc[ob_idx]
                        df.loc[ob_label, 'ob_bottom'] = df['Low'].iloc[ob_idx]
                    else:
                        df.loc[ob_label, 'ob_top'] = max(df['Open'].iloc[ob_idx],
                                                         df['Close'].iloc[ob_idx])
                        df.loc[ob_label, 'ob_bottom'] = min(df['Open'].iloc[ob_idx],
                                                            df['Close'].iloc[ob_idx])
                    df.loc[ob_label, 'ob_volume'] = df['Volume'].iloc[ob_idx]
                    price_move = (abs(df['Close'].iloc[ob_idx] - df['Open'].iloc[ob_idx]) /
                                  (df['Open'].iloc[ob_idx] + 1e-10))
                    vol_score = df['volume_percentile_static'].iloc[ob_idx] / 100
                    df.loc[ob_label, 'ob_strength'] = (price_move * 100 + vol_score) / 2
                    already_tagged.add(ob_label)      # ← store label (FIX-2)
                    break

    # ====================== MITIGATION (Forward Pass - unchanged) ======================
    if check_mitigation:
        active_bullish = []
        active_bearish = []

        for i in range(len(df)):
            row = df.iloc[i]
            idx = df.index[i]

            # Add newly detected OBs
            if row['ob_bullish_detected'] and not row['ob_bullish_mitigated']:
                active_bullish.append((idx, row['ob_top'], row['ob_bottom'], i))
            if row['ob_bearish_detected'] and not row['ob_bearish_mitigated']:
                active_bearish.append((idx, row['ob_top'], row['ob_bottom'], i))

            # Check if current bar mitigates any active bullish OB
            if active_bullish:
                for ob in active_bullish[:]:
                    ob_idx, ob_top, ob_bottom, ob_i = ob
                    ob_size = ob_top - ob_bottom
                    level = ob_top - ob_size * mitigation_threshold
                    if row['Close'] <= level:
                        closes_below = (df.iloc[ob_i + 1:i + 1]['Close'] <= level).sum()
                        if closes_below >= 2:
                            df.loc[ob_idx, 'ob_bullish_mitigated'] = True
                            active_bullish.remove(ob)

            # Check if current bar mitigates any active bearish OB
            if active_bearish:
                for ob in active_bearish[:]:
                    ob_idx, ob_top, ob_bottom, ob_i = ob
                    ob_size = ob_top - ob_bottom
                    level = ob_bottom + ob_size * mitigation_threshold
                    if row['Close'] >= level:
                        closes_above = (df.iloc[ob_i + 1:i + 1]['Close'] >= level).sum()
                        if closes_above >= 2:
                            df.loc[ob_idx, 'ob_bearish_mitigated'] = True
                            active_bearish.remove(ob)

    df['ob_bullish'] = df['ob_bullish_detected'] & ~df['ob_bullish_mitigated']
    df['ob_bearish'] = df['ob_bearish_detected'] & ~df['ob_bearish_mitigated']

    df = df.drop(columns=['is_bullish_candle', 'is_bearish_candle',
                           'swing_high', 'swing_low'], errors='ignore')
    return df




def calculate_cci(high, low, close, length=20, clamp=True):

    import numpy as np
    import pandas as pd

    # =====================================================
    # TYPICAL PRICE
    # =====================================================

    tp = (high + low + close) / 3.0

    # =====================================================
    # SMA OF TYPICAL PRICE
    # =====================================================

    sma_tp = tp.rolling(window=length).mean()

    # =====================================================
    # MEAN DEVIATION
    # =====================================================

    mean_dev = pd.Series(index=tp.index, dtype=float)

    for i in range(len(tp)):

        if i < length - 1:
            mean_dev.iloc[i] = np.nan
            continue

        window_tp = tp.iloc[i - length + 1 : i + 1]

        sma_value = sma_tp.iloc[i]

        deviation = np.abs(window_tp - sma_value)

        mean_dev.iloc[i] = deviation.mean()

    # =====================================================
    # PREVENT DIVISION BY VERY SMALL VALUES
    # =====================================================

    mean_dev = mean_dev.replace(0, np.nan)

    # =====================================================
    # CCI FORMULA
    # =====================================================

    cci = (
        (tp - sma_tp) /
        (0.015 * mean_dev)
    )

    # =====================================================
    # CLEAN EXTREME / BROKEN VALUES
    # =====================================================

    cci = cci.replace([np.inf, -np.inf], np.nan)

    # =====================================================
    # OPTIONAL CLAMP
    # =====================================================

    if clamp:
        cci = cci.clip(-300, 300)

    return cci




def super_trend(symbol, data, use_llm=False, use_ob_arm=True):
    """ Same logic as your original - now safe for live incremental calls """
    data = data.copy()
    close = data['Close']
    high = data['High']
    low = data['Low']
    n = len(data)

    stoch = ta.stoch(high, low, close, 14, 3, 3)
    data['Stoch_K'] = stoch['STOCHk_14_3_3']
    data['Stoch_D'] = stoch['STOCHd_14_3_3']

    ob_df = calculate_order_blocks(data)

    ob_cols = ['ob_bullish', 'ob_bearish', 'ob_top', 'ob_bottom', 'ob_volume',
               'ob_strength', 'ob_bullish_mitigated', 'ob_bearish_mitigated']

    for col in ob_cols:
        if col in ob_df.columns:
            data[col] = ob_df[col].reindex(data.index, fill_value=False)

    K = data['Stoch_K']
    D = data['Stoch_D']
    OB_Bullish = data.get('ob_bullish', pd.Series(False, index=data.index))

    STOCH_BUY_LEVEL = 15
    STOCH_UPTREND_THRESHOLD = 50
    COOLDOWN_BARS = 3
    OB_FLAG_TTL = 5

    signal = np.zeros(n, dtype=int)
    signal_reason = [''] * n
    signal_grade = [''] * n
    armed_source = [''] * n

    armed = False
    armed_bar = -1
    armed_date = None
    last_fired_bar = -999

    for i in range(5, n):
        k_cur = K.iloc[i]
        d_cur = D.iloc[i]
        ob_bullish_cur = OB_Bullish.iloc[i]
        dt_cur = data.index[i]

        if pd.isna(k_cur) or pd.isna(d_cur):
            continue

        curr_date = dt_cur.date() if hasattr(dt_cur, 'date') else pd.Timestamp(dt_cur).date()

        if armed and armed_date and curr_date != armed_date:
            armed = False
        if i-last_fired_bar < COOLDOWN_BARS:
            continue

        if use_ob_arm and not armed and ob_bullish_cur:
            k_in_uptrend = (k_cur > d_cur) and (k_cur > STOCH_UPTREND_THRESHOLD)
            if not k_in_uptrend and k_cur > 5:
                armed = True
                armed_bar = i
                armed_date = curr_date

        if armed and (i-armed_bar) > OB_FLAG_TTL:
            if not (k_cur > d_cur and k_cur > STOCH_BUY_LEVEL):
                armed = False

        if armed and i > armed_bar and k_cur > d_cur and k_cur > STOCH_BUY_LEVEL:
            signal[i] = 1
            signal_reason[i] = f"[OB] + K>D | K={k_cur:.1f} D={d_cur:.1f}"
            signal_grade[i] = "★★★★★"
            armed_source[i] = "OB"
            last_fired_bar = i
            armed = False

   
    data['st_sig'] = signal
    data['st_sig_raw'] = signal
    data['signal_reason'] = signal_reason
    data['signal_grade'] = signal_grade
    data['armed_source'] = armed_source
    print(f"{symbol}: st_sig exists =", 'st_sig' in data.columns)
    print(data.columns.tolist())
    if 'st_sig' not in data.columns:
        data['st_sig'] = 0

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
# ── track processed candle minutes to avoid double-fire ──────
last_processed_minute = {}  # {ticker: (hour, minute)}



while dt.datetime.now(pytz.timezone('Asia/Kolkata')) < endTime:

    try:

        for i in Tickers:

            now_ist = dt.datetime.now(pytz.timezone('Asia/Kolkata'))
            current_minute = (now_ist.hour, now_ist.minute)

            print("###################################################################")
            print('Spot prices of', i, ' ', spot_prices1[i])
            print(now_ist.strftime("%d-%b-%Y %I:%M:%S%p"))
            print("###################################################################")

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
                except Exception as e:
                    print(f"⚠️ Skipping {i}: {e}")
                    continue

                data_fut.drop(data_fut.tail(1).index, inplace=True)
                super_trend(i, data_fut)
                data_list[i] = data_fut

                super_Trend_Long = pd.read_excel(Long_Trade_File)
                Long_Open_Position = super_Trend_Long[super_Trend_Long['Trade Status'] == 'OPEN']
                super_Trend_Short = pd.read_excel(Short_Trade_File)
                Short_Open_Position = super_Trend_Short[super_Trend_Short['Trade Status'] == 'OPEN']

                # ── LONG ENTRY ────────────────────────────────
                if data_list[i]['st_sig'].iloc[-1] == 1:

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
                    Trade_quantity = 75
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

            # ── B1: TRAILING SL + TRAILING TARGET ────────────
            # Logic:
            #   Every 10 points of profit → move SL up by 10 (lock-in)
            #   Every 10 points → move Target up by 10 (trail target)
            Trail_Step = 10

            if profit_from_entry >= Trail_Step:
                # How many full steps have we moved
                steps = int(profit_from_entry // Trail_Step)

                # New SL = entry + (steps-1)*10  → always one step behind price
                # This locks in profit but gives room to breathe
                new_sl     = BuyPrice + ((steps - 1) * Trail_Step)
                new_sl     = max(new_sl, S_Price)   # never move SL down

                # New Target = entry + (steps+1)*10  → one step ahead of price
                new_target = BuyPrice + ((steps + 1) * Trail_Step)
                new_target = max(new_target, Target_Price)  # never move Target down

                if new_sl > S_Price:
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
