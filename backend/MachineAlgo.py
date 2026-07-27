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




def super_trendbkk(symbol: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    AGGRESSIVE MULTI-BRANCH OPTION BUYING STRATEGY
    FIXED: Morning trades + Wick Reversal + Enhanced Range-Bound Filter (PSAR + Candle Logic)
    NEW: Consolidation-Box Breakout branch + Fall-Box (post sell-off) Breakout branch
    NEW: Fall -> Hammer reversal short-circuit inside Fall-Box branch
    NEW: Non-repainting Bullish Order Block retest branch
    NEW: Non-repainting RSI bullish divergence branch
    NEW: Non-repainting MACD bullish divergence branch
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

    # rolling mean of ATR itself, used by the choppy-market filter below
    # to detect when volatility is drying up relative to its own recent average.
    data['ATR_MA'] = data['ATR'].rolling(20).mean()

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

    # ====================== NEW: RSI (Wilder) ======================
    # Standard Wilder RSI(14). Uses only Close up to and including the
    # current row, so it is non-repainting by construction (no future bars
    # are used to compute RSI at row idx).
    def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)  # neutral fill where undefined (early bars / zero loss)
        return rsi

    data['RSI'] = _wilder_rsi(data['Close'], 14)

    # ====================== NEW: MACD (12,26,9) ======================
    # Standard EMA-based MACD. Also uses only current/past Close values,
    # so it is non-repainting the same way EMA5/EMA10/etc. above already are.
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_hist'] = data['MACD'] - data['MACD_signal']

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

    def is_range_bound(idx, lookback=12, threshold_pct=0.85):
        """Enhanced Range Detection using PSAR + Candle Logic"""
        if idx < max(lookback, 20):
            return False

        window = data.iloc[idx-lookback:idx+1]
        c = data.iloc[idx]
        prev = data.iloc[idx-1] if idx > 0 else c

        price_range_pct = (window['High'].max()-window['Low'].min()) / window['Close'].iloc[-1] * 100

        psar_window = data['PSAR_trend'].iloc[idx-lookback:idx+1]
        psar_flips = (psar_window.diff() != 0).sum()

        no_breakout = c['High'] <= prev['High'] * 1.002

        recent_bodies = data['body_pct'].iloc[max(0, idx-7):idx]
        small_body_condition = (recent_bodies < 0.48).mean() >= 0.65

        is_range = (
                price_range_pct < threshold_pct or
                (psar_flips >= 4 and no_breakout and small_body_condition)
        )
        return is_range

    # CHOPPY MARKET FILTER
    # Runs a 6-check "vote" on the current candle. If 3 or more checks fire,
    # the market is flagged choppy/low-quality and the calling detector skips
    # the trade. This is purely a REJECTION layer -- it never creates a signal,
    # only suppresses false ones on top of whatever is_range_bound already does.
    # Note: thresholds are scaled to ATR (not fixed points) so this behaves
    # consistently whether ATR is small (quiet day) or large (volatile day).
    def is_choppy_market(idx, lookback=10):
        if idx < 25:  # not enough bars yet for EMA slope / ATR_MA to be meaningful
            return False

        c = data.iloc[idx]
        atr_now = data['ATR'].iloc[idx]
        if pd.isna(atr_now) or atr_now == 0:
            return False

        window = data.iloc[idx-lookback+1:idx+1]
        checks = []

        # 1. EMA5 slope flat -> no directional momentum
        ema_slope = abs(data['EMA5'].iloc[idx] - data['EMA5'].iloc[idx-5])
        checks.append(ema_slope < 0.08 * atr_now)

        # 2. Range compression -> last N candles stuck in a tight band
        range_points = window['High'].max() - window['Low'].min()
        checks.append(range_points < 1.8 * atr_now)

        # 3. Candle overlap -> most candles sit fully inside the prior candle's range
        overlap_count = 0
        for j in range(idx-lookback+1, idx+1):
            cj = data.iloc[j]
            pj = data.iloc[j-1]
            if cj['High'] <= pj['High'] and cj['Low'] >= pj['Low']:
                overlap_count += 1
        checks.append((overlap_count / lookback) >= 0.70)

        # 4. ATR below its own moving average -> volatility drying up
        atr_ma_now = data['ATR_MA'].iloc[idx]
        checks.append((not pd.isna(atr_ma_now)) and atr_now < atr_ma_now)

        # 5. PSAR sitting too close to price -> weak trend strength
        psar_distance = abs(c['Close'] - c['PSAR'])
        checks.append(psar_distance < 0.3 * atr_now)

        # 6. No real breakout -> close hasn't cleared the recent range
        recent_high = data['High'].iloc[idx-lookback:idx].max()
        checks.append(c['Close'] < recent_high * 1.002)

        return sum(checks) >= 3

    data['consolidation'] = False
    for i in range(20, n):
        win = data.iloc[i-20:i]
        rng = (win['High'].max()-win['Low'].min()) / win['Close'].iloc[-1]
        data.loc[data.index[i], 'consolidation'] = rng < 0.019

    # tracks which calendar dates have already produced a BASE_MORNING trade,
    # so detect_base() can enforce "only one BASE_MORNING trade per session".
    base_morning_dates_fired = set()

    # ====================== NEW: SHARED NON-REPAINTING SWING PIVOT HELPERS ======================
    # A swing low/high at position p is only "confirmed" once `right` bars have
    # closed AFTER p. So when we are standing at idx and ask "is there a
    # confirmed swing low as of idx", we only look at pivot_idx = idx - right
    # (or earlier) -- never at anything closer to idx than that. This mirrors
    # the confirmed-fractal-pivot approach used in your other OB code, so no
    # look-ahead is introduced.
    def _confirmed_swing_low(pivot_idx, left=3, right=3):
        if pivot_idx-left < 0:
            return False
        window = data['Low'].iloc[pivot_idx-left:pivot_idx+right+1]
        return data['Low'].iloc[pivot_idx] == window.min()

    def _confirmed_swing_high(pivot_idx, left=3, right=3):
        if pivot_idx-left < 0:
            return False
        window = data['High'].iloc[pivot_idx-left:pivot_idx+right+1]
        return data['High'].iloc[pivot_idx] == window.max()

    def _last_confirmed_swing_lows(idx, count=2, left=3, right=3, max_scan=80):
        """Scan backward from the most recent point that COULD be confirmed
        as of idx (i.e. pivot_idx = idx-right) and collect the most recent
        `count` confirmed swing-low pivot indices, newest first."""
        found = []
        latest_possible = idx-right
        p = latest_possible
        scanned = 0
        while p >= left and len(found) < count and scanned < max_scan:
            if _confirmed_swing_low(p, left, right):
                found.append(p)
                p -= (left+1)  # skip past this pivot's local window to avoid duplicates
            else:
                p -= 1
            scanned += 1
        return found

    # ====================== DETECTORS ======================
    def detect_base(idx):
        if idx < 20 or is_range_bound(idx, lookback=12, threshold_pct=0.85) or is_choppy_market(idx):
            return False, {}

        tm = data.index[idx].time()
        if not (dt.time(9, 15) <= tm <= dt.time(9, 55)):
            return False, {}

        current_date = data.index[idx].date()
        if current_date in base_morning_dates_fired:
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
        if idx < 14 or is_range_bound(idx, lookback=14, threshold_pct=0.95) or is_choppy_market(idx):
            return False, {}

        lookback = 9
        window = data.iloc[idx-lookback:idx+1]
        low_idx_rel = window['Low'].values.argmin()
        pivot_idx = idx-lookback+low_idx_rel
        bars_since_pivot = idx-pivot_idx

        if bars_since_pivot < 0 or bars_since_pivot > 1:
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

        # ====================== BOX (OX) BREAKOUT AFTER DECLINE ======================
        if pattern_type is None:
            box_min_candles = 3
            box_start = pivot_idx + 1
            box_end = idx - 1
            if box_end - box_start + 1 >= box_min_candles:
                box_window = data.iloc[box_start:box_end + 1]
                box_high = box_window['High'].max()
                box_low = box_window['Low'].min()
                box_range = box_high - box_low
                decline_size = swing_high - pivot_low

                box_is_tight = decline_size > 0 and box_range <= 0.45 * decline_size
                box_breakout_confirmed = c['Close'] > box_high and c['is_bullish']

                if box_is_tight and box_breakout_confirmed:
                    pattern_type = 'BOX_BREAKOUT'

        if pattern_type is None:
            return False, {}

        recovery_pct = (c['Close']-pivot_low) / (swing_high-pivot_low+1e-9)
        min_recovery = 0.14 if 'WICK' in pattern_type else 0.30
        if recovery_pct < min_recovery:
            return False, {}

        if pivot_idx == idx:
            recent_high = swing_high
        else:
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
        if idx < 10 or is_range_bound(idx, lookback=10, threshold_pct=0.9) or is_choppy_market(idx):
            return False, {}

        tm = data.index[idx].time()
        if tm < dt.time(10, 0):
            return False, {}

        c = data.iloc[idx]
        ema10 = data['EMA10'].iloc[idx]
        ema20 = data['EMA20'].iloc[idx]

        trend_ok = c['Close'] > ema10 > ema20
        body_ok = c['is_bullish'] and c['body_pct'] >= 0.48 and c['close_pos'] >= 0.55
        vol_ok = c['vol_ratio'] > 0.75
        pullback = data['Close'].iloc[idx-4:idx].min() < ema10 * 0.999
        breakout = c['Close'] > data['High'].iloc[idx-3:idx].max() * 0.999

        if not (trend_ok and body_ok and vol_ok and pullback and breakout):
            return False, {}

        return True, {
            'pattern_type': 'CONTINUATION',
            'strength': 'MEDIUM',
            'vol': round(c['vol_ratio'] * 100, 1)
        }

    # ====================== NEW BRANCH 1: SIDEWAYS CONSOLIDATION BOX BREAKOUT ======================
    def detect_box_consolidation_breakout(idx, box_lookback=12, min_candles_in_box=8):
        if idx < box_lookback+5 or is_choppy_market(idx):
            return False, {}

        atr_now = data['ATR'].iloc[idx]
        if pd.isna(atr_now) or atr_now == 0:
            return False, {}

        box_window = data.iloc[idx-box_lookback:idx]
        box_high = box_window['High'].max()
        box_low = box_window['Low'].min()
        box_range = box_high-box_low

        pre_box_start = idx-box_lookback
        pre_box_lookback = box_lookback * 2
        pre_start_idx = max(0, pre_box_start-pre_box_lookback)
        pre_window = data.iloc[pre_start_idx:pre_box_start]

        if len(pre_window) < 3:
            return False, {}

        pre_high = pre_window['High'].max()
        pre_low = pre_window['Low'].min()
        prior_fall = pre_high-pre_low

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
        for j in range(idx-box_lookback+1, idx):
            cj = data.iloc[j]
            pj = data.iloc[j-1]
            if cj['High'] <= pj['High'] * 1.0015 and cj['Low'] >= pj['Low'] * 0.9985:
                overlap_count += 1
        if (overlap_count / (box_lookback-1)) < 0.25:
            return False, {}

        c = data.iloc[idx]

        breakout_confirmed = (
                c['Close'] > box_high and
                c['is_bullish'] and
                c['body_pct'] >= 0.20 and
                c['close_pos'] >= 0.25
        )
        if not breakout_confirmed:
            return False, {}

        if c['vol_ratio'] < 0.65:
            return False, {}

        return True, {
            'pattern_type': 'CONSOLIDATION_BOX_BREAKOUT',
            'box_high': round(box_high, 2),
            'box_low': round(box_low, 2),
            'box_range': round(box_range, 2),
            'prior_fall': round(prior_fall, 2),
            'vol': round(c['vol_ratio'] * 100, 1)
        }

    # ====================== NEW BRANCH 2: FALL -> SMALL BOX -> BREAKOUT (+ HAMMER SHORT-CIRCUIT) ======================
    def detect_fall_box_breakout(idx, min_red_streak=4, max_wait=8):
        if idx < min_red_streak+6 or is_choppy_market(idx):
            return False, {}

        c = data.iloc[idx]
        if not c['is_bullish']:
            return False, {}

        for gap in range(1, max_wait+1):
            end_idx = idx-gap
            if end_idx < min_red_streak+1:
                continue
            if data.iloc[end_idx]['is_bullish']:
                continue

            red_streak = 0
            j = end_idx
            while j >= 0 and not data.iloc[j]['is_bullish']:
                red_streak += 1
                j -= 1
            streak_start_idx = j+1

            if red_streak <= min_red_streak:
                continue
            if end_idx-1 < 0:
                continue

            last_red_idx = end_idx
            second_last_red_idx = end_idx-1
            red1 = data.iloc[last_red_idx]
            red2 = data.iloc[second_last_red_idx]

            box_high = max(red1['Open'], red1['Close'], red2['Open'], red2['Close'])
            box_low = min(red1['Open'], red1['Close'], red2['Open'], red2['Close'])

            streak_high = data['High'].iloc[streak_start_idx:last_red_idx+1].max()
            streak_low = data['Low'].iloc[streak_start_idx:last_red_idx+1].min()
            total_fall = streak_high-streak_low

            if total_fall <= 40:
                continue

            atr_now = data['ATR'].iloc[idx]
            if pd.isna(atr_now) or atr_now == 0:
                continue

            if (streak_high-streak_low) < 1.5 * atr_now:
                continue

            # HAMMER REVERSAL SHORT-CIRCUIT: if a hammer forms after the red
            # streak, the very next candle is the signal -- no box breakout required.
            hammer_idx = None
            for h in range(last_red_idx+1, idx):
                hc = data.iloc[h]
                hc_range = hc['High']-hc['Low']
                if hc_range <= 0:
                    continue
                hc_body_pct = abs(hc['Close']-hc['Open']) / hc_range
                hc_upper_pct = (hc['High']-max(hc['Open'], hc['Close'])) / hc_range
                hc_lower_pct = (min(hc['Open'], hc['Close'])-hc['Low']) / hc_range
                if hc_lower_pct >= 0.50 and hc_body_pct <= 0.30 and hc_upper_pct <= 0.10:
                    hammer_idx = h
                    break

            if hammer_idx is not None and hammer_idx == idx-1:
                if c['vol_ratio'] < 0.55:
                    continue
                return True, {
                    'pattern_type': 'FALL_HAMMER_REVERSAL',
                    'red_streak': red_streak,
                    'total_fall': round(total_fall, 2),
                    'hammer_idx': int(hammer_idx),
                    'vol': round(c['vol_ratio'] * 100, 1)
                }

            already_broken = False
            for k in range(last_red_idx+1, idx):
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
                'pattern_type': 'FALL_BOX_BREAKOUT',
                'red_streak': red_streak,
                'total_fall': round(total_fall, 2),
                'box_high': round(box_high, 2),
                'box_low': round(box_low, 2),
                'vol': round(c['vol_ratio'] * 100, 1)
            }

        return False, {}

    # ====================== NEW BRANCH 3: NON-REPAINTING BULLISH ORDER BLOCK RETEST ======================
    # Logic: find a confirmed swing low -> confirm BOS (a close breaking the
    # prior structure high with a strong-bodied displacement candle) -> the
    # order block is the last RED candle before that impulsive breakout leg
    # -> wait for price to come back and retest the OB zone with a bullish
    # rejection candle. Each OB anchor can only fire once (ob_fired set),
    # matching the non-repainting / no-refire discipline you use elsewhere.
    ob_fired = set()

    def detect_order_block_reversal(idx, left=3, right=3, structure_lookback=40, max_ob_age=40):
        if idx < structure_lookback+right+5 or is_choppy_market(idx):
            return False, {}

        swing_lows = _last_confirmed_swing_lows(idx, count=1, left=left, right=right, max_scan=structure_lookback)
        if not swing_lows:
            return False, {}
        pivot_idx = swing_lows[0]

        # structure high = highest high BEFORE the swing low (what price must
        # reclaim to confirm a bullish break of structure)
        struct_start = max(0, pivot_idx-structure_lookback)
        if pivot_idx <= struct_start:
            return False, {}
        structure_high = data['High'].iloc[struct_start:pivot_idx].max()
        if pd.isna(structure_high):
            return False, {}

        # find the BOS candle: first candle AFTER the swing low that closes
        # above structure_high with real displacement (strong body)
        bos_idx = None
        for k in range(pivot_idx+1, idx+1):
            ck = data.iloc[k]
            if ck['Close'] > structure_high and ck['is_bullish'] and ck['body_pct'] >= 0.45:
                bos_idx = k
                break
        if bos_idx is None or bos_idx >= idx:
            return False, {}  # no BOS yet, or BOS is literally the current bar (need time to retest)

        if (idx - bos_idx) > max_ob_age:
            return False, {}  # BOS too old, OB likely already invalidated / stale

        # OB anchor = last RED candle before the BOS candle (walk back from bos_idx-1)
        ob_idx = None
        for k in range(bos_idx-1, pivot_idx-1, -1):
            if not data.iloc[k]['is_bullish']:
                ob_idx = k
                break
        if ob_idx is None:
            return False, {}

        if ob_idx in ob_fired:
            return False, {}

        ob = data.iloc[ob_idx]
        ob_high = max(ob['Open'], ob['Close'])
        ob_low = min(ob['Open'], ob['Close'])

        # Zone must not have been closed clean through (invalidated) between
        # the OB and now -- if any candle closed below ob_low, structure failed.
        between = data.iloc[ob_idx+1:idx]
        if not between.empty and (between['Close'] < ob_low).any():
            return False, {}

        c = data.iloc[idx]

        # retest: current low taps into the zone, and candle shows a bullish
        # rejection out of it (close back above the zone, decent close position)
        tapped_zone = c['Low'] <= ob_high and c['Low'] >= ob_low * 0.997
        rejection = c['is_bullish'] and c['Close'] >= ob_low and c['close_pos'] >= 0.55
        if not (tapped_zone and rejection):
            return False, {}

        if c['vol_ratio'] < 0.60:
            return False, {}

        return True, {
            'pattern_type': 'ORDER_BLOCK_RETEST',
            'ob_idx': int(ob_idx),
            'ob_high': round(ob_high, 2),
            'ob_low': round(ob_low, 2),
            'bos_idx': int(bos_idx),
            'vol': round(c['vol_ratio'] * 100, 1)
        }, ob_idx  # extra element consumed below to mark ob_fired

    # ====================== NEW BRANCH 4: RSI BULLISH DIVERGENCE ======================
    # Compares the two most recent CONFIRMED swing-low pivots: price prints a
    # lower low while RSI prints a higher low = bullish divergence. Requires
    # the current candle to itself be a bullish reversal candle (not just any
    # candle after the divergence exists) before firing. Each pivot-pair only
    # fires once via rsi_div_fired.
    rsi_div_fired = set()

    def detect_rsi_bullish_divergence(idx, left=3, right=3, recency_bars=6):
        if idx < 30 or is_choppy_market(idx):
            return False, {}

        pivots = _last_confirmed_swing_lows(idx, count=2, left=left, right=right, max_scan=80)
        if len(pivots) < 2:
            return False, {}

        p1, p2 = pivots[0], pivots[1]  # p1 = newer pivot, p2 = older pivot

        # divergence must still be "fresh" -- current candle within recency_bars
        # of the newer pivot, otherwise it's stale and we skip it
        if (idx - p1) > recency_bars:
            return False, {}

        pair_key = (p2, p1)
        if pair_key in rsi_div_fired:
            return False, {}

        price_lower_low = data['Low'].iloc[p1] < data['Low'].iloc[p2]
        rsi_higher_low = data['RSI'].iloc[p1] > data['RSI'].iloc[p2]
        if not (price_lower_low and rsi_higher_low):
            return False, {}

        c = data.iloc[idx]
        trigger_candle_ok = c['is_bullish'] and c['body_pct'] >= 0.40 and c['close_pos'] >= 0.55
        if not trigger_candle_ok:
            return False, {}

        if c['vol_ratio'] < 0.60:
            return False, {}

        return True, {
            'pattern_type': 'RSI_BULLISH_DIVERGENCE',
            'pivot_old': int(p2),
            'pivot_new': int(p1),
            'rsi_old': round(data['RSI'].iloc[p2], 1),
            'rsi_new': round(data['RSI'].iloc[p1], 1),
            'vol': round(c['vol_ratio'] * 100, 1)
        }, pair_key

    # ====================== NEW BRANCH 5: MACD BULLISH DIVERGENCE ======================
    # Same pivot-pair logic as RSI divergence above, but compares MACD
    # histogram values at the two confirmed swing lows instead of RSI.
    macd_div_fired = set()

    def detect_macd_bullish_divergence(idx, left=3, right=3, recency_bars=6):
        if idx < 30 or is_choppy_market(idx):
            return False, {}

        pivots = _last_confirmed_swing_lows(idx, count=2, left=left, right=right, max_scan=80)
        if len(pivots) < 2:
            return False, {}

        p1, p2 = pivots[0], pivots[1]

        if (idx - p1) > recency_bars:
            return False, {}

        pair_key = (p2, p1)
        if pair_key in macd_div_fired:
            return False, {}

        price_lower_low = data['Low'].iloc[p1] < data['Low'].iloc[p2]
        macd_higher_low = data['MACD_hist'].iloc[p1] > data['MACD_hist'].iloc[p2]
        if not (price_lower_low and macd_higher_low):
            return False, {}

        c = data.iloc[idx]
        trigger_candle_ok = c['is_bullish'] and c['body_pct'] >= 0.40 and c['close_pos'] >= 0.55
        if not trigger_candle_ok:
            return False, {}

        if c['vol_ratio'] < 0.60:
            return False, {}

        return True, {
            'pattern_type': 'MACD_BULLISH_DIVERGENCE',
            'pivot_old': int(p2),
            'pivot_new': int(p1),
            'macd_hist_old': round(data['MACD_hist'].iloc[p2], 2),
            'macd_hist_new': round(data['MACD_hist'].iloc[p1], 2),
            'vol': round(c['vol_ratio'] * 100, 1)
        }, pair_key

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

    COOLDOWN_CANDLES = 10
    last_signal_idx = -10**9

    for idx in range(18, n):
        if idx - last_signal_idx < COOLDOWN_CANDLES:
            continue

        candidates = []

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

        box_det, box_data = detect_box_consolidation_breakout(idx)
        if box_det:
            entry, stop, target, rr = calculate_levels(idx, 'CONSOLIDATION_BOX')
            quality = get_quality(rr)
            if quality != "REJECT":
                candidates.append({
                    'condition': 'CONSOLIDATION_BOX_BREAKOUT', 'quality': quality, 'entry': entry,
                    'stop': stop, 'target': target, 'rr': rr, 'confidence': 74,
                    'data': box_data, 'priority': 4
                })

        fb_det, fb_data = detect_fall_box_breakout(idx)
        if fb_det:
            entry, stop, target, rr = calculate_levels(idx, 'FALL_BOX')
            quality = get_quality(rr)
            if quality != "REJECT":
                candidates.append({
                    'condition': fb_data['pattern_type'], 'quality': quality, 'entry': entry,
                    'stop': stop, 'target': target, 'rr': rr, 'confidence': 80,
                    'data': fb_data, 'priority': 2
                })

        # NEW: Branch 3 - non-repainting order block retest
        ob_result = detect_order_block_reversal(idx)
        if ob_result[0]:
            ob_det, ob_data, ob_anchor_idx = ob_result
            entry, stop, target, rr = calculate_levels(idx, 'ORDER_BLOCK')
            quality = get_quality(rr)
            if quality != "REJECT":
                candidates.append({
                    'condition': 'ORDER_BLOCK_RETEST', 'quality': quality, 'entry': entry,
                    'stop': stop, 'target': target, 'rr': rr, 'confidence': 78,
                    'data': ob_data, 'priority': 2, '_ob_anchor': ob_anchor_idx
                })

        # NEW: Branch 4 - RSI bullish divergence
        rsi_result = detect_rsi_bullish_divergence(idx)
        if rsi_result[0]:
            rsi_det, rsi_data, rsi_pair_key = rsi_result
            entry, stop, target, rr = calculate_levels(idx, 'RSI_DIV')
            quality = get_quality(rr)
            if quality != "REJECT":
                candidates.append({
                    'condition': 'RSI_BULLISH_DIVERGENCE', 'quality': quality, 'entry': entry,
                    'stop': stop, 'target': target, 'rr': rr, 'confidence': 76,
                    'data': rsi_data, 'priority': 2, '_rsi_pair': rsi_pair_key
                })

        # NEW: Branch 5 - MACD bullish divergence
        macd_result = detect_macd_bullish_divergence(idx)
        if macd_result[0]:
            macd_det, macd_data, macd_pair_key = macd_result
            entry, stop, target, rr = calculate_levels(idx, 'MACD_DIV')
            quality = get_quality(rr)
            if quality != "REJECT":
                candidates.append({
                    'condition': 'MACD_BULLISH_DIVERGENCE', 'quality': quality, 'entry': entry,
                    'stop': stop, 'target': target, 'rr': rr, 'confidence': 76,
                    'data': macd_data, 'priority': 2, '_macd_pair': macd_pair_key
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

            if condition == 'BASE_MORNING':
                base_morning_dates_fired.add(data.index[idx].date())

            # NEW: mark the winning pattern's anti-refire memory so it never
            # fires again on the same OB / divergence pair
            if '_ob_anchor' in best:
                ob_fired.add(best['_ob_anchor'])
            if '_rsi_pair' in best:
                rsi_div_fired.add(best['_rsi_pair'])
            if '_macd_pair' in best:
                macd_div_fired.add(best['_macd_pair'])

            last_signal_idx = idx

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





import pandas as pd
import numpy as np
import datetime as dt
from collections import deque, Counter

try:
    import pandas_ta as ta  # only needed by the ported multi-div indicator calc
except ImportError as e:
    raise ImportError(
        "pandas_ta is required for the multi-indicator divergence engine "
        "(pip install pandas_ta)"
    ) from e


# =====================================================================================
# =========================  FALSE-SIGNAL FIX CHANGELOG  ============================
# =====================================================================================
# 1. calculate_levels / calculate_levels_bear: fixed 11/30-point SL/TP replaced with
#    ATR-relative sizing so RR/quality actually reflects current volatility instead of
#    being a constant ~2.7x for every trade regardless of conditions.
# 2. Volume thresholds on breakout/reversal branches raised from below-average
#    (0.55-0.65x) to at/above-average, since below-average-volume breakouts are the
#    classic false-breakout signature.
# 3. New optional `trend_persistence` param on super_trend() (default=1, i.e. old
#    behavior unchanged) lets you require N consecutive ema_trend-confirmed candles
#    instead of a single noisy candle before a pending signal is allowed to fire.
# 4. detect_v_shape: added an EMA50-slope guard so the branch doesn't buy V-shape
#    reversals against a strongly falling macro trend (falling-knife protection).
# 5. detect_box_consolidation_breakout: added invalidation if the same resistance
#    level was already closed-above (and failed) earlier -- a "breakout" of a level
#    that already failed once is a weaker signal.
# 6. detect_order_block_reversal: added a tap-count decay -- an order block that's
#    already been retested more than twice is treated as weakened/invalid.
# 7. Multi-indicator divergence: confidence now scales with how many indicators
#    agree (agree_count) instead of a flat 78 regardless of 1 vs 5 indicators.
# 8. detect_base: added an optional index-level trend confirmation (only applied
#    when index_data is supplied -- no behavior change if you don't pass it).
# =====================================================================================


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
        i, src, close, prsc,
        pl_positions, pl_vals,
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
        i, src, close, prsc,
        ph_positions, ph_vals,
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
        index_df: pd.DataFrame,
        prd: int = 5,
        source: str = "Close",
        searchdiv: str = "Regular",
        showlimit: int = 3,
        maxpp: int = 10,
        maxbars: int = 100,
        dontconfirm: bool = False,
        showlast: bool = False,
        calcmacd: bool = True,
        calcmacda: bool = True,
        calcrsi: bool = True,
        calcstoc: bool = True,
        calccci: bool = True,
        calcmom: bool = True,
        calcobv: bool = True,
        calcvwmacd: bool = True,
        calccmf: bool = True,
        calcmfi: bool = True,
        verbose: bool = True,
) -> pd.DataFrame:
    """
    Identical logic to your second file's calculate_divergences(). The ONLY
    addition is the `verbose` block at the end, which prints every fired
    label so you can see exactly how many bullish / bearish multi-indicator
    divergence events the index produced -- independent of what the option
    strategy later does with them.

    Must be called on INDEX candles (Nifty/BankNifty/etc), not option
    premium candles.
    """
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

    n = len(df)

    remove_last_pos_divs = False
    remove_last_neg_divs = False

    pos_label_history = []
    neg_label_history = []

    for i in range(n):
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


# =====================================================================================
# ================================  MAIN STRATEGY  ===================================
# =====================================================================================

def super_trend(symbol: str, data: pd.DataFrame, confirm_max_wait: int = 4,
                 index_data: pd.DataFrame = None,
                 multi_div_pivot_period: int = 5,
                 multi_div_min_count: int = 1,
                 multi_div_maxpp: int = 10,
                 multi_div_maxbars: int = 100,
                 multi_div_search: str = "Regular",
                 multi_div_showlast: bool = False,
                 multi_div_align_tolerance: "pd.Timedelta|None" = None,
                 trend_persistence: int = 1) -> pd.DataFrame:
    """
    `data` -- OPTION PREMIUM candles (either the CE chain or the PE chain --
              call this function once per side, same as before).
    `index_data` -- underlying INDEX candles (Nifty/BankNifty/etc), REQUIRED
              for the multi-indicator-divergence branch and for the index
              volume-profile false-signal filter. If omitted, both are
              skipped (a warning is printed) and every other branch runs
              unchanged on the option data.
    `multi_div_align_tolerance` -- how close (in time) an option candle must
              be to an index divergence label to "claim" it. Defaults to
              2x the index feed's own median bar interval. If your feeds
              have irregular gaps (illiquid option strikes, missing
              minutes), pass a wider explicit value, e.g. pd.Timedelta(minutes=10).
    `trend_persistence` -- FALSE-SIGNAL FIX: number of consecutive candles
              that must agree with the required ema_trend direction before a
              pending signal is allowed to confirm and fire. Default (1)
              preserves the original single-candle-confirmation behavior.
              Raising to 2-3 filters out whipsaw confirmations where
              ema_trend flips for one bar and immediately flips back.

    Bullish (bottom) divergence label on the index -> feeds the CE pending
    queue (buy call). Bearish (top) divergence label on the index -> feeds
    the PE pending queue (buy put). Both are still gated by your EMA5/HLC3
    ema_trend confirmation exactly like every other pending-queue branch,
    and both still pass through the index volume-profile rejection filter
    before they're allowed to fire.
    """
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
    data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
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

    # NOTE: RSI/MACD above are kept only as general-purpose diagnostic columns
    # on the option data (e.g. for your own charting/inspection). They are no
    # longer inputs to any divergence detector -- the multi-indicator
    # divergence engine now runs entirely on index_data, computing its own
    # RSI/MACD/Stoch/CCI/MOM/OBV/VWMACD/CMF/MFI internally via
    # _md_calculate_indicators(). The old option-data Stoch/CCI/MOM/OBV/
    # VWMACD/CMF/MFI block has been removed since nothing on the option side
    # consumed it any more.

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

    data['PSAR'], data['PSAR_trend'] = psar(data['High'].values, data['Low'].values, data['Close'].values)

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

    def _confirmed_swing_low(pivot_idx, left=3, right=3):
        if pivot_idx - left < 0:
            return False
        window = data['Low'].iloc[pivot_idx - left:pivot_idx + right + 1]
        return data['Low'].iloc[pivot_idx] == window.min()

    def _confirmed_swing_high(pivot_idx, left=3, right=3):
        if pivot_idx - left < 0:
            return False
        window = data['High'].iloc[pivot_idx - left:pivot_idx + right + 1]
        return data['High'].iloc[pivot_idx] == window.max()

    def _last_confirmed_swing_lows(idx, count=2, left=3, right=3, max_scan=80):
        found = []
        latest_possible = idx - right
        p = latest_possible
        scanned = 0
        while p >= left and len(found) < count and scanned < max_scan:
            if _confirmed_swing_low(p, left, right):
                found.append(p)
                p -= (left + 1)
            else:
                p -= 1
            scanned += 1
        return found

    # ====================== INDEX-DRIVEN MULTI-INDICATOR DIVERGENCE ======================
    # Run ONCE, up front, on index_data -- never touches option premium candles.
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
            verbose=True,
        )
    else:
        print("⚠️  index_data not supplied -- MULTI_INDICATOR_DIVERGENCE branch and the "
              "index volume-profile filter are both DISABLED for this run. Every other "
              "branch still runs normally on the option data.")

    # ---- Align index candles + divergence labels onto the option-data timeline ----
    # Same non-repainting principle as your original index-volume-profile alignment:
    # a backward as-of merge, so at any option candle idx we only ever see index
    # information timestamped <= that candle's own timestamp.
    use_index_filter = index_data is not None
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
        idx_df = idx_df.sort_index()[['Open', 'High', 'Low', 'Close', 'Volume']].add_prefix('idx_')

        # Median bar interval of the index feed -- used to size the alignment
        # tolerance so labels don't linger across many option candles, and
        # don't get missed if the two feeds aren't perfectly synced.
        if len(idx_df) > 1:
            median_interval = pd.Series(idx_df.index).diff().dropna().median()
        else:
            median_interval = pd.Timedelta(minutes=1)
        if pd.isna(median_interval) or median_interval <= pd.Timedelta(0):
            median_interval = pd.Timedelta(minutes=1)
        align_tol = multi_div_align_tolerance if multi_div_align_tolerance is not None else 2 * median_interval

        opt_start, opt_end = data.index.min(), data.index.max()
        idx_start, idx_end = idx_df.index.min(), idx_df.index.max()
        overlap_start, overlap_end = max(opt_start, idx_start), min(opt_end, idx_end)
        print(f"🔗 Aligning index_data onto option timeline | option range: {opt_start} → {opt_end} | "
              f"index range: {idx_start} → {idx_end} | overlap: "
              f"{overlap_start} → {overlap_end if overlap_start <= overlap_end else 'NONE (no date overlap!)'} "
              f"| tolerance: {align_tol}")

        data = data.sort_index()
        data_reset = data.reset_index()  # 'Datetime' column
        idx_reset = idx_df.reset_index()  # 'Datetime' column
        data_reset = pd.merge_asof(
            data_reset, idx_reset, on='Datetime', direction='backward'
        )
        matched_ohlc = data_reset['idx_Close'].notna().sum()
        data = data_reset.set_index('Datetime')
        print(f"   -> {matched_ohlc}/{len(data)} option candles matched to an index OHLC row.")
    else:
        print("⚠️  index_data not supplied -- MULTI_INDICATOR_DIVERGENCE branch and the "
              "index volume-profile filter are both DISABLED for this run. Every other "
              "branch still runs normally on the option data.")

    if div_labels is not None and not div_labels.empty and use_index_filter:
        dl = div_labels.sort_index().reset_index()  # columns: Datetime, Close, pos_reg_div, ...
        dl = dl.rename(columns={'Datetime': '_div_label_time'})
        dl = dl[['_div_label_time', 'label_type', 'divergence_indicators',
                  'pos_reg_div', 'neg_reg_div', 'pos_hid_div', 'neg_hid_div']]
        dl_sorted = dl.sort_values('_div_label_time')
        data_reset = data.reset_index()  # 'Datetime' column
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

    # A label can legitimately be the nearest-preceding match for more than one
    # option candle (if option candles are more frequent than index candles, or
    # timestamps aren't perfectly synced). We de-duplicate that inside the main
    # loop below by comparing each row's `_div_label_time` (the SOURCE label's
    # own timestamp, carried through the merge) against the last one already
    # consumed -- not the option candle's own timestamp, which would differ on
    # every row and defeat the de-dup.

    VP_LOOKBACK = 50
    VP_BINS = 24
    VP_VALUE_AREA_PCT = 0.70

    def _compute_index_volume_profile(idx, lookback=VP_LOOKBACK, num_bins=VP_BINS,
                                       value_area_pct=VP_VALUE_AREA_PCT):
        if not use_index_filter:
            return None
        window = data.iloc[max(0, idx - lookback + 1):idx + 1]
        window = window.dropna(subset=['idx_High', 'idx_Low', 'idx_Close', 'idx_Volume'])
        if len(window) < max(10, lookback // 3):
            return None
        vp_high = window['idx_High'].max()
        vp_low = window['idx_Low'].min()
        if vp_high <= vp_low:
            return None
        bin_edges = np.linspace(vp_low, vp_high, num_bins + 1)
        bin_vols = np.zeros(num_bins)
        for _, row in window.iterrows():
            h, l, v = row['idx_High'], row['idx_Low'], row['idx_Volume']
            if v <= 0 or h <= l or pd.isna(v):
                continue
            lo_bin = max(0, min(num_bins - 1, int(np.searchsorted(bin_edges, l, side='right')) - 1))
            hi_bin = max(0, min(num_bins - 1, int(np.searchsorted(bin_edges, h, side='right')) - 1))
            span = hi_bin - lo_bin + 1
            bin_vols[lo_bin:hi_bin + 1] += v / span
        total_vol = bin_vols.sum()
        if total_vol <= 0:
            return None
        poc_bin = int(np.argmax(bin_vols))
        poc_price = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2
        target_vol = total_vol * value_area_pct
        acc_vol = bin_vols[poc_bin]
        lo, hi = poc_bin, poc_bin
        while acc_vol < target_vol and (lo > 0 or hi < num_bins - 1):
            left_vol = bin_vols[lo - 1] if lo > 0 else -1
            right_vol = bin_vols[hi + 1] if hi < num_bins - 1 else -1
            if right_vol >= left_vol:
                hi += 1
                acc_vol += bin_vols[hi]
            else:
                lo -= 1
                acc_vol += bin_vols[lo]
        vah = bin_edges[hi + 1]
        val = bin_edges[lo]
        cur_close = data['idx_Close'].iloc[idx]
        if pd.isna(cur_close):
            return None
        cur_bin = max(0, min(num_bins - 1, int(np.searchsorted(bin_edges, cur_close, side='right')) - 1))
        avg_vol = bin_vols.mean()
        is_lvn = bin_vols[cur_bin] < 0.6 * avg_vol
        return {'poc': poc_price, 'vah': vah, 'val': val, 'cur_close': cur_close, 'is_lvn': is_lvn}

    def index_volume_profile_confirms_bullish(idx):
        if not use_index_filter:
            return True
        vp = _compute_index_volume_profile(idx)
        if vp is None:
            return True
        if vp['cur_close'] < vp['poc']:
            return False
        if vp['cur_close'] < vp['vah'] and not vp['is_lvn']:
            return False
        return True

    def index_volume_profile_confirms_bearish(idx):
        if not use_index_filter:
            return True
        vp = _compute_index_volume_profile(idx)
        if vp is None:
            return True
        if vp['cur_close'] > vp['poc']:
            return False
        if vp['cur_close'] > vp['val'] and not vp['is_lvn']:
            return False
        return True

    # ====================== FALSE-SIGNAL FIX: trend persistence helper ======================
    def _trend_confirmed(idx, direction, bars=trend_persistence):
        """Require `bars` consecutive candles agreeing with `direction`
        (1 for bullish, -1 for bearish) instead of a single noisy candle.
        bars=1 reproduces the original single-candle check exactly."""
        bars = max(1, int(bars))
        start = idx - bars + 1
        if start < 0:
            return False
        window = data['ema_trend'].iloc[start:idx + 1]
        if len(window) < bars:
            return False
        return bool((window == direction).all())

    # ====================== DETECTORS (unchanged) ======================
    def detect_base(idx):
        if idx < 20 or is_range_bound(idx, lookback=12, threshold_pct=0.85) or is_choppy_market(idx):
            return False, {}
        tm = data.index[idx].time()
        if not (dt.time(9, 15) <= tm <= dt.time(9, 55)):
            return False, {}
        current_date = data.index[idx].date()
        if current_date in base_morning_dates_fired:
            return False, {}
        c = data.iloc[idx]
        uptrend = c['Close'] > c['EMA5'] > c['EMA20']
        strong_body = c['is_bullish'] and c['body_pct'] >= 0.55 and c['close_pos'] >= 0.60
        momentum = (c['Close'] - c['Open']) / (c['High'] - c['Low'] + 1e-9) > 0.62
        vol_ok = c['vol_ratio'] > 1.20
        if not (uptrend and strong_body and momentum and vol_ok):
            return False, {}
        if c['Close'] < data['Low'].iloc[idx - 8:idx].min() * 0.985:
            return False, {}
        # FALSE-SIGNAL FIX: don't buy a morning breakout against a falling index.
        # Only applied when index_data was actually supplied; no behavior change otherwise.
        if use_index_filter:
            idx_close_now = data['idx_Close'].iloc[idx]
            idx_close_prev = data['idx_Close'].iloc[max(0, idx - 5)]
            if not pd.isna(idx_close_now) and not pd.isna(idx_close_prev):
                if idx_close_now < idx_close_prev:
                    return False, {}
        return True, {'pattern_type': 'BASE_MORNING', 'strength': 'HIGH', 'vol': round(c['vol_ratio'] * 100, 1)}

    def detect_v_shape(idx):
        if idx < 14 or is_range_bound(idx, lookback=14, threshold_pct=0.95) or is_choppy_market(idx):
            return False, {}
        lookback = 9
        window = data.iloc[idx - lookback:idx + 1]
        low_idx_rel = window['Low'].values.argmin()
        pivot_idx = idx - lookback + low_idx_rel
        bars_since_pivot = idx - pivot_idx
        if bars_since_pivot < 0 or bars_since_pivot > 1:
            return False, {}
        if data['Low'].iloc[pivot_idx] > data['Low'].iloc[max(0, pivot_idx - 6):pivot_idx + 7].min():
            return False, {}
        left_start = max(0, pivot_idx - 8)
        left_window = data.iloc[left_start:pivot_idx + 1]
        swing_high = left_window['High'].max()
        pivot_low = data['Low'].iloc[pivot_idx]
        decline_pct = (swing_high - pivot_low) / swing_high * 100 if swing_high > 0 else 0
        points_drop = swing_high - pivot_low
        red_streak = 0
        for j in range(pivot_idx, left_start - 1, -1):
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
            for j in range(left_start, pivot_idx + 1)
        )
        if not ema5_below_found and 'EMA5' in data.columns:
            return False, {}
        # FALSE-SIGNAL FIX: don't buy a V-shape reversal against a strongly
        # falling macro trend (falling-knife guard). Compares EMA50 now vs
        # ~10 bars back, scaled by ATR so it adapts to volatility regime.
        if idx >= 10:
            atr_now = data['ATR'].iloc[idx]
            if not pd.isna(atr_now) and atr_now > 0:
                ema50_slope = data['EMA50'].iloc[idx] - data['EMA50'].iloc[idx - 10]
                if ema50_slope < -0.6 * atr_now:
                    return False, {}
        c = data.iloc[idx]
        candle_range = c['High'] - c['Low']
        lower_wick = min(c['Open'], c['Close']) - c['Low']
        lower_wick_pct = lower_wick / candle_range if candle_range > 0 else 0
        upper_wick_pct = (c['High'] - max(c['Open'], c['Close'])) / candle_range if candle_range > 0 else 0
        is_prev_long_wick_red = False
        if idx > 0:
            pc = data.iloc[idx - 1]
            p_range = pc['High'] - pc['Low']
            if p_range > 0 and not pc['is_bullish']:
                p_lower = min(pc['Open'], pc['Close']) - pc['Low']
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
            box_min_candles = 3
            box_start = pivot_idx + 1
            box_end = idx - 1
            if box_end - box_start + 1 >= box_min_candles:
                box_window = data.iloc[box_start:box_end + 1]
                box_high = box_window['High'].max()
                box_low = box_window['Low'].min()
                box_range = box_high - box_low
                decline_size = swing_high - pivot_low
                box_is_tight = decline_size > 0 and box_range <= 0.45 * decline_size
                box_breakout_confirmed = c['Close'] > box_high and c['is_bullish']
                if box_is_tight and box_breakout_confirmed:
                    pattern_type = 'BOX_BREAKOUT'
        if pattern_type is None:
            return False, {}
        recovery_pct = (c['Close'] - pivot_low) / (swing_high - pivot_low + 1e-9)
        min_recovery = 0.14 if 'WICK' in pattern_type else 0.30
        if recovery_pct < min_recovery:
            return False, {}
        if pivot_idx == idx:
            recent_high = swing_high
        else:
            recent_high = data['High'].iloc[pivot_idx:idx].max()
        breakout_price = c['High'] if 'WICK' in pattern_type else c['Close']
        if breakout_price < recent_high * 0.996:
            return False, {}
        min_vol = 0.85 if 'WICK' in pattern_type else 1.10
        if c['vol_ratio'] < min_vol:
            return False, {}
        return True, {
            'pattern_type': pattern_type, 'decline_pct': round(decline_pct, 2),
            'recovery_pct': round(recovery_pct * 100, 2),
            'lower_wick_pct': round(lower_wick_pct * 100, 2), 'vol': round(c['vol_ratio'] * 100, 1)
        }

    def detect_continuation(idx):
        if idx < 10 or is_range_bound(idx, lookback=10, threshold_pct=0.9) or is_choppy_market(idx):
            return False, {}
        tm = data.index[idx].time()
        if tm < dt.time(10, 0):
            return False, {}
        c = data.iloc[idx]
        ema10 = data['EMA10'].iloc[idx]
        ema20 = data['EMA20'].iloc[idx]
        trend_ok = c['Close'] > ema10 > ema20
        body_ok = c['is_bullish'] and c['body_pct'] >= 0.48 and c['close_pos'] >= 0.55
        # FALSE-SIGNAL FIX: 0.75 (below average volume) raised to 0.90 -- a
        # continuation breakout on below-average volume is a weak signal.
        vol_ok = c['vol_ratio'] > 0.90
        pullback = data['Close'].iloc[idx - 4:idx].min() < ema10 * 0.999
        breakout = c['Close'] > data['High'].iloc[idx - 3:idx].max() * 0.999
        # FALSE-SIGNAL FIX: don't chase a continuation move that's already
        # deeply overbought -- exhaustion risk is high and reversion is likely.
        rsi_ok = data['RSI'].iloc[idx] <= 75
        if not (trend_ok and body_ok and vol_ok and pullback and breakout and rsi_ok):
            return False, {}
        return True, {'pattern_type': 'CONTINUATION', 'strength': 'MEDIUM', 'vol': round(c['vol_ratio'] * 100, 1)}

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
        # FALSE-SIGNAL FIX: if this exact resistance level was already closed
        # above earlier (and evidently failed, since we're still forming a
        # box under it now), today's "breakout" is a retest of a level that
        # has already proven weak once -- skip it.
        prior_break = (pre_window['Close'] > box_high).any()
        if prior_break:
            return False, {}
        c = data.iloc[idx]
        breakout_confirmed = (c['Close'] > box_high and c['is_bullish']
                               and c['body_pct'] >= 0.20 and c['close_pos'] >= 0.25)
        if not breakout_confirmed:
            return False, {}
        # FALSE-SIGNAL FIX: 0.65 (below average volume) raised to 0.90 --
        # box breakouts on sub-average volume are the classic false-breakout.
        if c['vol_ratio'] < 0.90:
            return False, {}
        return True, {
            'pattern_type': 'CONSOLIDATION_BOX_BREAKOUT', 'box_high': round(box_high, 2),
            'box_low': round(box_low, 2), 'box_range': round(box_range, 2),
            'prior_fall': round(prior_fall, 2), 'vol': round(c['vol_ratio'] * 100, 1)
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
                # FALSE-SIGNAL FIX: 0.55 (below average volume) raised to 0.75.
                if c['vol_ratio'] < 0.75:
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
            # FALSE-SIGNAL FIX: 0.55 (below average volume) raised to 0.75.
            if c['vol_ratio'] < 0.75:
                continue
            return True, {
                'pattern_type': 'FALL_BOX_BREAKOUT', 'red_streak': red_streak,
                'total_fall': round(total_fall, 2), 'box_high': round(box_high, 2),
                'box_low': round(box_low, 2), 'vol': round(c['vol_ratio'] * 100, 1)
            }
        return False, {}

    ob_fired = set()
    ob_tap_count = {}  # FALSE-SIGNAL FIX: decay an order block's reliability after repeated taps

    def detect_order_block_reversal(idx, left=3, right=3, structure_lookback=40, max_ob_age=40):
        if idx < structure_lookback + right + 5 or is_choppy_market(idx):
            return False, {}
        swing_lows = _last_confirmed_swing_lows(idx, count=1, left=left, right=right, max_scan=structure_lookback)
        if not swing_lows:
            return False, {}
        pivot_idx = swing_lows[0]
        struct_start = max(0, pivot_idx - structure_lookback)
        if pivot_idx <= struct_start:
            return False, {}
        structure_high = data['High'].iloc[struct_start:pivot_idx].max()
        if pd.isna(structure_high):
            return False, {}
        bos_idx = None
        for k in range(pivot_idx + 1, idx + 1):
            ck = data.iloc[k]
            if ck['Close'] > structure_high and ck['is_bullish'] and ck['body_pct'] >= 0.45:
                bos_idx = k
                break
        if bos_idx is None or bos_idx >= idx:
            return False, {}
        if (idx - bos_idx) > max_ob_age:
            return False, {}
        ob_idx = None
        for k in range(bos_idx - 1, pivot_idx - 1, -1):
            if not data.iloc[k]['is_bullish']:
                ob_idx = k
                break
        if ob_idx is None:
            return False, {}
        if ob_idx in ob_fired:
            return False, {}
        ob = data.iloc[ob_idx]
        ob_high = max(ob['Open'], ob['Close'])
        ob_low = min(ob['Open'], ob['Close'])
        between = data.iloc[ob_idx + 1:idx]
        if not between.empty and (between['Close'] < ob_low).any():
            return False, {}
        c = data.iloc[idx]
        tapped_zone = c['Low'] <= ob_high and c['Low'] >= ob_low * 0.997
        rejection = c['is_bullish'] and c['Close'] >= ob_low and c['close_pos'] >= 0.55
        if not (tapped_zone and rejection):
            return False, {}
        # FALSE-SIGNAL FIX: a zone that's already been tapped and re-tapped
        # without a clean, decisive break is weakening -- reject after the
        # 2nd tap instead of treating every retest as equally reliable.
        ob_tap_count[ob_idx] = ob_tap_count.get(ob_idx, 0) + 1
        if ob_tap_count[ob_idx] > 2:
            return False, {}
        # FALSE-SIGNAL FIX: 0.60 (below average volume) raised to 0.75 --
        # a genuine rejection candle off a real order block should show
        # participation, not sub-average volume.
        if c['vol_ratio'] < 0.75:
            return False, {}
        return True, {
            'pattern_type': 'ORDER_BLOCK_RETEST', 'ob_idx': int(ob_idx), 'ob_high': round(ob_high, 2),
            'ob_low': round(ob_low, 2), 'bos_idx': int(bos_idx), 'vol': round(c['vol_ratio'] * 100, 1)
        }, ob_idx

    # ====================== LEVELS, QUALITY, RATING ======================
    # FALSE-SIGNAL FIX: the original fixed 11-point SL / 30-point TP made the
    # RR-based quality gate meaningless -- every trade scored the same ~2.7x RR
    # regardless of actual volatility, so "REJECT" almost never fired based on
    # real risk. SL is now sized as a volatility-clamped % of the option
    # premium (15-35%, driven by ATR), and TP scales with the current
    # volatility regime (ATR vs its 20-bar average) instead of being constant.
    _TP_MULT_BY_REGIME = {'LOW': 2.2, 'NORMAL': 2.6, 'HIGH': 3.2}

    def _volatility_regime(idx):
        atr_now = data['ATR'].iloc[idx]
        atr_ma = data['ATR_MA'].iloc[idx]
        if pd.isna(atr_now) or pd.isna(atr_ma) or atr_ma == 0:
            return 'NORMAL'
        ratio = atr_now / atr_ma
        if ratio >= 1.3:
            return 'HIGH'
        if ratio <= 0.75:
            return 'LOW'
        return 'NORMAL'

    def calculate_levels(idx, condition):
        entry = data['Close'].iloc[idx]
        atr_now = data['ATR'].iloc[idx]
        if pd.isna(atr_now) or atr_now <= 0 or entry <= 0:
            atr_now = entry * 0.05 if entry > 0 else 1.0
        sl_pct = float(np.clip(atr_now / entry, 0.15, 0.35)) if entry > 0 else 0.20
        stop_points = entry * sl_pct
        tp_mult = _TP_MULT_BY_REGIME[_volatility_regime(idx)]
        target_points = stop_points * tp_mult
        stop = entry - stop_points
        target = entry + target_points
        rr = abs(target - entry) / (abs(entry - stop) + 1e-9)
        return entry, stop, target, rr

    def calculate_levels_bear(idx, condition):
        entry = data['Close'].iloc[idx]
        atr_now = data['ATR'].iloc[idx]
        if pd.isna(atr_now) or atr_now <= 0 or entry <= 0:
            atr_now = entry * 0.05 if entry > 0 else 1.0
        sl_pct = float(np.clip(atr_now / entry, 0.15, 0.35)) if entry > 0 else 0.20
        stop_points = entry * sl_pct
        tp_mult = _TP_MULT_BY_REGIME[_volatility_regime(idx)]
        target_points = stop_points * tp_mult
        stop = entry + stop_points
        target = entry - target_points
        rr = abs(entry - target) / (abs(stop - entry) + 1e-9)
        return entry, stop, target, rr

    def get_quality(rr):
        if rr < 1.25:
            return "REJECT"
        return "PREMIUM" if rr >= 2.2 else "HIGH" if rr >= 1.8 else "MEDIUM"

    _QUALITY_WEIGHT = {"PREMIUM": 3, "HIGH": 2, "MEDIUM": 1, "REJECT": -999}

    def _rating_score(confidence, quality, priority):
        """Unified cross-branch rating. Higher = better. Combines the
        detector's own confidence, the RR-derived quality bucket (this
        dominates -- a PREMIUM RR candidate always outranks a MEDIUM one
        regardless of confidence), and priority as a tiebreaker only."""
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
    last_signal_idx = -10 ** 9

    pending_signals = []       # CE (bullish) branches awaiting ema_trend == 1 confirmation
    pending_signals_bear = []  # PE (bearish) branches awaiting ema_trend == -1 confirmation

    multi_div_bull_seen = 0
    multi_div_bear_seen = 0
    multi_div_bull_queued = 0
    multi_div_bear_queued = 0
    last_multi_div_bull_time = None
    last_multi_div_bear_time = None

    for idx in range(18, n):

        candidates_this_idx = []  # each fully-priced, rated, ready to fire

        # ---- STEP 1a: evaluate pending CE (bullish) confirmation queue ----
        still_pending = []
        for p in pending_signals:
            age = idx - p['detect_idx']
            if age < 0:
                still_pending.append(p)
                continue
            if age > confirm_max_wait:
                continue  # expired
            # FALSE-SIGNAL FIX: require `trend_persistence` consecutive bullish
            # candles instead of a single noisy ema_trend flip (trend_persistence=1
            # reproduces the original behavior exactly).
            if _trend_confirmed(idx, 1):
                if index_volume_profile_confirms_bullish(idx):
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
                continue  # one-shot: consumed whether or not it wins the rating
            else:
                still_pending.append(p)
        pending_signals = still_pending

        # ---- STEP 1b: evaluate pending PE (bearish) confirmation queue ----
        still_pending_bear = []
        for p in pending_signals_bear:
            age = idx - p['detect_idx']
            if age < 0:
                still_pending_bear.append(p)
                continue
            if age > confirm_max_wait:
                continue
            if _trend_confirmed(idx, -1):
                if index_volume_profile_confirms_bearish(idx):
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

        # ---- STEP 2: Order-Block branch (unchanged: immediate, no ema-trend gate) ----
        ob_anchor_for_candidate = None
        ob_result = detect_order_block_reversal(idx)
        if ob_result[0]:
            ob_det, ob_data, ob_anchor_idx = ob_result
            entry, stop, target, rr = calculate_levels(idx, 'ORDER_BLOCK')
            quality = get_quality(rr)
            if quality != "REJECT" and index_volume_profile_confirms_bullish(idx):
                score = _rating_score(78, quality, 2)
                candidates_this_idx.append({
                    'side': 'CE', 'condition': 'ORDER_BLOCK_RETEST', 'quality': quality,
                    'entry': entry, 'stop': stop, 'target': target, 'rr': rr,
                    'confidence': 78, 'priority': 2, 'data': ob_data, 'score': score,
                    'wait_note': "", '_ob_anchor': ob_anchor_idx
                })

        # ---- STEP 3: fire the single best-rated candidate for this candle ----
        if candidates_this_idx and (idx - last_signal_idx) >= COOLDOWN_CANDLES:
            best = max(candidates_this_idx, key=lambda x: x['score'])
            result = _write_signal(
                idx, best['side'], best['condition'], best['data'], best['confidence'],
                best['entry'], best['stop'], best['target'], best['rr'], best['quality'],
                best['score'], best['wait_note']
            )
            signals_list.append(result)
            last_signal_idx = idx
            if '_ob_anchor' in best:
                ob_fired.add(best['_ob_anchor'])
            # already handled this candle -- skip fresh detection below
            continue

        if idx - last_signal_idx < COOLDOWN_CANDLES:
            continue

        # ---- STEP 4: fresh detection (feeds pending queues for future candles) ----
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
            pending_signals.append({'detect_idx': idx, 'condition': 'BULLISH_CONTINUATION',
                                     'data': c_data, 'confidence': 66, 'priority': 3})

        box_det, box_data = detect_box_consolidation_breakout(idx)
        if box_det:
            pending_signals.append({'detect_idx': idx, 'condition': 'CONSOLIDATION_BOX_BREAKOUT',
                                     'data': box_data, 'confidence': 74, 'priority': 4})

        fb_det, fb_data = detect_fall_box_breakout(idx)
        if fb_det:
            pending_signals.append({'detect_idx': idx, 'condition': fb_data['pattern_type'],
                                     'data': fb_data, 'confidence': 80, 'priority': 2})

        # ---- Multi-indicator divergence (INDEX-driven) -> pending queues ----
        if use_multi_div:
            row_label = data['label_type'].iloc[idx] if 'label_type' in data.columns else None
            if isinstance(row_label, str):
                indicators_str = data['divergence_indicators'].iloc[idx]
                pos_reg = data['pos_reg_div'].iloc[idx] if 'pos_reg_div' in data.columns else 0
                pos_hid = data['pos_hid_div'].iloc[idx] if 'pos_hid_div' in data.columns else 0
                neg_reg = data['neg_reg_div'].iloc[idx] if 'neg_reg_div' in data.columns else 0
                neg_hid = data['neg_hid_div'].iloc[idx] if 'neg_hid_div' in data.columns else 0

                src_time = data['_div_label_time'].iloc[idx] if '_div_label_time' in data.columns else data.index[idx]

                if row_label == 'bottom' and src_time != last_multi_div_bull_time:
                    last_multi_div_bull_time = src_time
                    multi_div_bull_seen += 1
                    agree_count = int(pos_reg + pos_hid)
                    # FALSE-SIGNAL FIX: confidence now scales with how many
                    # indicators actually agree, instead of a flat 78 for a
                    # single indicator vs. five indicators agreeing.
                    md_confidence = float(min(90, 55 + agree_count * 9))
                    print(f"[MULTI-DIV] BULLISH #{multi_div_bull_seen} @ {data.index[idx]} "
                          f"| agree={agree_count} | indicators=({indicators_str}) -> queued for CE "
                          f"(confidence={md_confidence:.0f})")
                    pending_signals.append({
                        'detect_idx': idx, 'condition': 'MULTI_INDICATOR_BULLISH_DIVERGENCE',
                        'data': {'agree_count': agree_count, 'pos_reg_div': pos_reg, 'pos_hid_div': pos_hid},
                        'confidence': md_confidence, 'priority': 2
                    })
                    multi_div_bull_queued += 1

                elif row_label == 'top' and src_time != last_multi_div_bear_time:
                    last_multi_div_bear_time = src_time
                    multi_div_bear_seen += 1
                    agree_count = int(neg_reg + neg_hid)
                    md_confidence = float(min(90, 55 + agree_count * 9))
                    print(f"[MULTI-DIV] BEARISH #{multi_div_bear_seen} @ {data.index[idx]} "
                          f"| agree={agree_count} | indicators=({indicators_str}) -> queued for PE "
                          f"(confidence={md_confidence:.0f})")
                    pending_signals_bear.append({
                        'detect_idx': idx, 'condition': 'MULTI_INDICATOR_BEARISH_DIVERGENCE',
                        'data': {'agree_count': agree_count, 'neg_reg_div': neg_reg, 'neg_hid_div': neg_hid},
                        'confidence': md_confidence, 'priority': 2
                    })
                    multi_div_bear_queued += 1

    # ====================== SUMMARY ======================
    print(f"\n📊 SIGNALS GENERATED: {len(signals_list)}\n")
    if use_multi_div:
        print("🔎 MULTI-INDICATOR DIVERGENCE FUNNEL (index-driven branch):")
        print(f"   Bullish labels detected on index : {multi_div_bull_seen}  "
              f"(queued for CE confirmation: {multi_div_bull_queued})")
        print(f"   Bearish labels detected on index : {multi_div_bear_seen}  "
              f"(queued for PE confirmation: {multi_div_bear_queued})")
        md_fired = [s for s in signals_list if 'MULTI_INDICATOR' in s['condition']]
        md_fired_bull = sum(1 for s in md_fired if s['signal_type'] == 'CE')
        md_fired_bear = sum(1 for s in md_fired if s['signal_type'] == 'PE')
        print(f"   Actually FIRED as trades         : {len(md_fired)}  "
              f"(CE: {md_fired_bull}, PE: {md_fired_bear})")
        print("   [Everything queued-but-not-fired was dropped by: ema_trend never confirming "
              "within confirm_max_wait, the index volume-profile filter, the RR quality gate, "
              "cooldown, or losing the cross-branch rating to a higher-scored candidate.]\n")

    if signals_list:
        print(f"{'DateTime':<26} {'Close':<10} {'Side':<5} {'Condition':<32} "
              f"{'Quality':<10} {'RR':<7} {'Score':<7} {'Conf':<6}")
        print("-" * 115)
        for sig in signals_list[-25:]:
            print(f"{str(sig['datetime']):<26} {sig['close']:<10.2f} {sig.get('signal_type','CE'):<5} "
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
