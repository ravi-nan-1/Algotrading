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










def fetch_ohlcv(symbol, timeframe="3m", candles=30):
    print('call11 fetch_ohlcv')
    scripcode = scripcode_lookup(instrument_df, symbol)
    print(scripcode)
    if not scripcode:
        return None

    df = pd.DataFrame(
        client.historical_data(
            Exch='N', ExchangeSegment='D',
            ScripCode=scripcode,
            time=timeframe,
            From=dt.date.today() - dt.timedelta(5),
            To=dt.date.today()
        )
    )
    print(df)
    # 🔥 SAFE DATETIME HANDLING
    datetime_col = None
    for col in ["Datetime", "DateTime", "Date", "Time"]:
        if col in df.columns:
            datetime_col = col
            break

    if datetime_col:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(datetime_col)

    # Ensure required columns exist
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        return None

    df = df.tail(candles)

    return [
        {
            "o": round(float(r["Open"]), 2),
            "h": round(float(r["High"]), 2),
            "l": round(float(r["Low"]), 2),
            "c": round(float(r["Close"]), 2),
            "v": int(r["Volume"])
        }
        for _, r in df.iterrows()
    ]









def llm_trade_signal(symbol, ohlcv, timeframe="3m"):
    """
    Call Groq LLM to validate a BUY signal.

    Returns:
        {
            "signal": "buy" | "sell" | "no trade",
            "confidence": 0.0 to 1.0,
            "reason": "explanation",
            "bias": "Bullish | Bearish | Neutral",
            "entry": float,
            "stop_loss": float,
            "target_1": float,
            "target_2": float
        }
    """

    # ===========================================================
    #  SAFE DEFAULTS — returned on ANY failure
    # ===========================================================
    SAFE_DEFAULT = {
        "signal": "no trade",
        "confidence": 0.0,
        "reason": "LLM call failed",
        "bias": "Neutral",
        "entry": 0.0,
        "stop_loss": 0.0,
        "target_1": 0.0,
        "target_2": 0.0
    }

    try:
        print(f"🤖 Calling LLM for {symbol}...")

        # ===========================================================
        #  FORMAT OHLCV DATA
        # ===========================================================
        recent_data = ohlcv[-50:] if len(ohlcv) > 50 else ohlcv

        candle_text = ""
        for candle in recent_data:
            if isinstance(candle, dict):
                candle_text += (
                    f"  Time={candle.get('time', '?')} "
                    f"O={candle.get('open', '?')} "
                    f"H={candle.get('high', '?')} "
                    f"L={candle.get('low', '?')} "
                    f"C={candle.get('close', '?')} "
                    f"V={candle.get('volume', '?')}\n"
                )
            elif isinstance(candle, (list, tuple)):
                candle_text += (
                    f"  {candle[0]} O={candle[1]} H={candle[2]} "
                    f"L={candle[3]} C={candle[4]} V={candle[5]}\n"
                )

        if not candle_text.strip():
            print("⚠️  No candle data to send to LLM")
            SAFE_DEFAULT["reason"] = "No candle data available"
            return SAFE_DEFAULT

        # ===========================================================
        #  PROMPT
        # ===========================================================
        prompt = f""" You are a very selective 3-minute timeframe intraday options trader in Indian markets (Nifty/BankNifty heavy). 
        Your edge comes from waiting for high-conviction oversold recoveries with momentum confirmation. 
        You are an elite intraday stock trader for Indian markets, aiming for 90%+ win rate by being ultra-conservative and multi-factor validated.You are a professional discretionary trader.

Analyze OHLCV data for {symbol} on {timeframe}.
Trade only if quality is high.

SYMBOL: {symbol}
TIMEFRAME: {timeframe}

RECENT CANDLE DATA (latest at bottom):
{candle_text}




YOUR TASK:

1. Analyze price action, momentum, volume,volume profile  and candle structure 
2. Look for WARNING signs:
   - Bearish divergence
   - Strong resistance nearby
   - Exhaustion / climax candles
   - Fake breakout patterns
   - Low volume on bounce
3. If trade is valid, suggest entry, stop loss and targets

4. Calibrate confidence high only for pristine setups (target 90% accuracy by rejecting marginal trades).

RESPOND ONLY with valid JSON. No markdown. No explanation outside JSON.

{{
  "bias": "Bullish" or "Bearish" or "Neutral",
  "signal": "BUY" or "SELL" or "NO TRADE",
  "entry": number,
  "stop_loss": number,
  "target_1": number,
  "target_2": number,
  "confidence": number between 0.0 and 1.0,
  "reason": "1-2 sentence explanation"
}}

CONFIDENCE CALIBRATION:
- 0.85-1.0: Textbook setup, strong momentum, clean structure
- 0.70-0.84: Good setup, minor concerns but tradeable
- 0.50-0.69: Mixed signals, risky
- 0.30-0.49: Weak, high chance of failure
- 0.00-0.29: Dangerous, clear reversal signals
"""

        # ===========================================================
        #  API CALL
        # ===========================================================
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "temperature": 0.1,
                "max_tokens": 300,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a trading analyst. Respond ONLY in valid JSON. No markdown."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            timeout=30
        )

        # ===========================================================
        #  HANDLE API FAILURE
        # ===========================================================
        if response.status_code != 200:
            print(f"❌ Groq API error: {response.status_code}")
            print(f"   Details: {response.text[:500]}")
            SAFE_DEFAULT["reason"] = f"API error: {response.status_code}"
            return SAFE_DEFAULT

        data = response.json()

        if "choices" not in data:
            print(f"❌ Invalid Groq response: {data}")
            SAFE_DEFAULT["reason"] = "Invalid API response"
            return SAFE_DEFAULT

        content = data["choices"][0]["message"]["content"]
        print(f"📝 Raw LLM response: {content}")

        # ===========================================================
        #  PARSE JSON
        # ===========================================================
        # Remove markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        # Extract JSON object
        start = content.index("{")
        end = content.rindex("}") + 1
        json_str = content[start:end]

        result = json.loads(json_str)

        # ===========================================================
        #  VALIDATE & NORMALIZE
        # ===========================================================
        # Signal
        raw_signal = str(result.get("signal", "NO TRADE")).strip().upper()
        valid_signals = {
            "BUY": "buy",
            "SELL": "sell",
            "NO TRADE": "no trade",
            "HOLD": "no trade",
            "NEUTRAL": "no trade"
        }
        signal = valid_signals.get(raw_signal, "no trade")

        # Confidence (clamp 0-1)
        try:
            confidence = float(result.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.0

        # Other fields
        reason = str(result.get("reason", "No reason provided"))

        bias = str(result.get("bias", "Neutral")).strip()
        if bias not in ["Bullish", "Bearish", "Neutral"]:
            bias = "Neutral"

        def safe_float(val, default=0.0):
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        entry = safe_float(result.get("entry"))
        stop_loss = safe_float(result.get("stop_loss"))
        target_1 = safe_float(result.get("target_1"))
        target_2 = safe_float(result.get("target_2"))

        return {
            "signal": signal,
            "confidence": confidence,
            "reason": reason,
            "bias": bias,
            "entry": entry,
            "stop_loss": stop_loss,
            "target_1": target_1,
            "target_2": target_2
        }

    # ===========================================================
    #  ERROR HANDLING
    # ===========================================================
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        SAFE_DEFAULT["reason"] = f"JSON parse error: {e}"
        return SAFE_DEFAULT

    except requests.exceptions.Timeout:
        print("❌ LLM request timed out (30s)")
        SAFE_DEFAULT["reason"] = "Request timed out"
        return SAFE_DEFAULT

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Groq API")
        SAFE_DEFAULT["reason"] = "Connection error"
        return SAFE_DEFAULT

    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        SAFE_DEFAULT["reason"] = f"Error: {e}"
        return SAFE_DEFAULT



import json
import requests
import numpy as np
import pandas as pd
import traceback


def compute_all_indicators(ohlcv):
    """
    Compute all technical indicators from OHLCV data.
    Handles multiple input formats robustly.
    """

    # ===========================================================
    #  STEP 1: DEBUG — Print what we actually received
    # ===========================================================
    print(f"  📦 OHLCV type: {type(ohlcv)}")

    if ohlcv is None:
        raise ValueError("OHLCV data is None!")

    if isinstance(ohlcv, pd.DataFrame):
        print(f"  📦 DataFrame columns: {list(ohlcv.columns)}")
        print(f"  📦 DataFrame shape: {ohlcv.shape}")
        print(f"  📦 First row: {ohlcv.iloc[0].to_dict() if len(ohlcv) > 0 else 'EMPTY'}")
        df = ohlcv.copy()
    elif isinstance(ohlcv, (list, tuple)):
        print(f"  📦 List length: {len(ohlcv)}")
        if len(ohlcv) == 0:
            raise ValueError("OHLCV list is empty!")

        first = ohlcv[0]
        print(f"  📦 First element type: {type(first)}")
        print(f"  📦 First element: {first}")

        if isinstance(first, dict):
            print(f"  📦 Dict keys: {list(first.keys())}")
            df = pd.DataFrame(ohlcv)
        elif isinstance(first, (list, tuple)):
            print(f"  📦 List element length: {len(first)}")
            # Could be [time, o, h, l, c, v] or [o, h, l, c, v] or other formats
            if len(first) >= 6:
                df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            elif len(first) == 5:
                df = pd.DataFrame(ohlcv, columns=['open', 'high', 'low', 'close', 'volume'])
                df['time'] = range(len(df))
            else:
                raise ValueError(f"Unexpected list element length: {len(first)}")
        elif isinstance(first, (int, float)):
            raise ValueError(f"OHLCV appears to be a flat list of numbers, not candle data")
        else:
            raise ValueError(f"Unsupported OHLCV element type: {type(first)}")
    else:
        raise ValueError(f"Unsupported OHLCV type: {type(ohlcv)}")

    # ===========================================================
    #  STEP 2: NORMALIZE COLUMN NAMES
    # ===========================================================
    print(f"  📦 DataFrame columns before rename: {list(df.columns)}")

    # Map common alternative column names to standard names
    column_mappings = {
        # open variants
        'o': 'open', 'Open': 'open', 'OPEN': 'open',
        'op': 'open', 'opening': 'open',
        # high variants
        'h': 'high', 'High': 'high', 'HIGH': 'high',
        'hi': 'high',
        # low variants
        'l': 'low', 'Low': 'low', 'LOW': 'low',
        'lo': 'low',
        # close variants
        'c': 'close', 'Close': 'close', 'CLOSE': 'close',
        'cl': 'close', 'closing': 'close', 'ltp': 'close', 'last': 'close',
        # volume variants
        'v': 'volume', 'Volume': 'volume', 'VOLUME': 'volume',
        'vol': 'volume', 'Vol': 'volume',
        # time variants
        't': 'time', 'Time': 'time', 'TIME': 'time',
        'date': 'time', 'Date': 'time', 'datetime': 'time',
        'Datetime': 'time', 'timestamp': 'time', 'Timestamp': 'time',
    }

    # If columns are numeric (0, 1, 2, 3, 4, 5), map them
    if all(isinstance(c, int) for c in df.columns):
        if len(df.columns) >= 6:
            df.columns = ['time', 'open', 'high', 'low', 'close', 'volume'] + \
                         [f'extra_{i}' for i in range(len(df.columns) - 6)]
        elif len(df.columns) == 5:
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df['time'] = range(len(df))

    # Apply rename mapping
    rename_map = {}
    for col in df.columns:
        col_str = str(col)
        if col_str in column_mappings:
            rename_map[col] = column_mappings[col_str]
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        print(f"  📦 Renamed columns: {rename_map}")

    print(f"  📦 Final columns: {list(df.columns)}")

    # ===========================================================
    #  STEP 3: VALIDATE REQUIRED COLUMNS EXIST
    # ===========================================================
    required = ['open', 'high', 'low', 'close']
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Last resort: try to assign by position
        print(f"  ⚠️ Missing columns: {missing}. Trying positional assignment...")
        cols = list(df.columns)
        if len(cols) >= 5:
            # Assume format: time, open, high, low, close, volume (or without time)
            # Find which ones look like price data
            numeric_cols = [c for c in cols if pd.to_numeric(df[c], errors='coerce').notna().sum() > len(df) * 0.5]
            print(f"  📦 Numeric columns: {numeric_cols}")

        raise ValueError(
            f"Cannot find required columns {missing}. "
            f"Available columns: {list(df.columns)}. "
            f"First row: {df.iloc[0].to_dict() if len(df) > 0 else 'EMPTY'}"
        )

    # Add volume column if missing
    if 'volume' not in df.columns:
        df['volume'] = 0
        print("  ⚠️ No volume column found, defaulting to 0")

    # Add time column if missing
    if 'time' not in df.columns:
        df['time'] = range(len(df))

    # ===========================================================
    #  STEP 4: CONVERT TO NUMERIC
    # ===========================================================
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['close'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < 5:
        raise ValueError(f"Only {len(df)} valid rows after cleaning. Need at least 5.")

    print(f"  ✅ Clean data: {len(df)} rows")
    print(f"  📦 Sample row: O={df['open'].iloc[-1]} H={df['high'].iloc[-1]} "
          f"L={df['low'].iloc[-1]} C={df['close'].iloc[-1]} V={df['volume'].iloc[-1]}")

    # ===========================================================
    #  STEP 5: COMPUTE ALL INDICATORS
    # ===========================================================
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_price = df['open']

    # ========================
    #  1. MOVING AVERAGES
    # ========================
    df['EMA_5'] = close.ewm(span=5, adjust=False).mean()
    df['EMA_9'] = close.ewm(span=9, adjust=False).mean()
    df['EMA_15'] = close.ewm(span=15, adjust=False).mean()
    df['EMA_21'] = close.ewm(span=21, adjust=False).mean()
    df['SMA_20'] = close.rolling(window=min(20, len(df))).mean()

    # VWAP
    typical_price = (high + low + close) / 3
    cum_vol = volume.cumsum().replace(0, np.nan)
    df['VWAP'] = (typical_price * volume).cumsum() / cum_vol

    # ========================
    #  2. RSI (14-period)
    # ========================
    period = min(14, len(df) - 1)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss_series = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss_series.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # ========================
    #  3. STOCHASTIC (14, 3, 3)
    # ========================
    stoch_period = min(14, len(df))
    lowest_14 = low.rolling(window=stoch_period).min()
    highest_14 = high.rolling(window=stoch_period).max()
    denom = (highest_14 - lowest_14).replace(0, np.nan)
    df['STOCH_K'] = 100 * (close - lowest_14) / denom
    df['STOCH_D'] = df['STOCH_K'].rolling(window=3).mean()

    # ========================
    #  4. MACD (12, 26, 9)
    # ========================
    ema_12 = close.ewm(span=min(12, len(df)), adjust=False).mean()
    ema_26 = close.ewm(span=min(26, len(df)), adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=min(9, len(df)), adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # ========================
    #  5. BOLLINGER BANDS (20, 2)
    # ========================
    bb_period = min(20, len(df))
    bb_sma = close.rolling(window=bb_period).mean()
    bb_std = close.rolling(window=bb_period).std()
    df['BB_Upper'] = bb_sma + (2 * bb_std)
    df['BB_Middle'] = bb_sma
    df['BB_Lower'] = bb_sma - (2 * bb_std)
    bb_range = (df['BB_Upper'] - df['BB_Lower']).replace(0, np.nan)
    df['BB_Width'] = bb_range / df['BB_Middle'].replace(0, np.nan)
    df['BB_PctB'] = (close - df['BB_Lower']) / bb_range

    # ========================
    #  6. ATR (14-period)
    # ========================
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_period = min(14, len(df))
    df['ATR_14'] = true_range.ewm(span=atr_period, adjust=False).mean()

    # ========================
    #  7. ADX (14-period)
    # ========================
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr_safe = df['ATR_14'].replace(0, np.nan)
    plus_di = 100 * (plus_dm.ewm(span=atr_period, adjust=False).mean() / atr_safe)
    minus_di = 100 * (minus_dm.ewm(span=atr_period, adjust=False).mean() / atr_safe)
    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    df['ADX'] = dx.ewm(span=atr_period, adjust=False).mean()

    # ========================
    #  8. VOLUME INDICATORS
    # ========================
    vol_ma_period = min(20, len(df))
    df['Vol_MA_20'] = volume.rolling(window=vol_ma_period).mean()
    df['Vol_Ratio'] = volume / df['Vol_MA_20'].replace(0, np.nan)

    # OBV
    obv_list = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv_list.append(obv_list[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv_list.append(obv_list[-1] - volume.iloc[i])
        else:
            obv_list.append(obv_list[-1])
    df['OBV'] = obv_list

    # ========================
    #  9. SUPERTREND (10, 3)
    # ========================
    st_atr = true_range.ewm(span=min(10, len(df)), adjust=False).mean()
    hl2 = (high + low) / 2
    upper_band = hl2 + (3 * st_atr)
    lower_band = hl2 - (3 * st_atr)

    supertrend = pd.Series(index=df.index, dtype='float64')
    direction = pd.Series(index=df.index, dtype='int64')
    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = -1

    for i in range(1, len(df)):
        if close.iloc[i] > upper_band.iloc[i - 1]:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
        else:
            if direction.iloc[i - 1] == 1:
                supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i - 1])
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i - 1])
                direction.iloc[i] = -1

    df['Supertrend'] = supertrend
    df['Supertrend_Dir'] = direction

    # ========================
    # 10. CANDLE STRUCTURE
    # ========================
    body = (close - open_price).abs()
    candle_range = (high - low).replace(0, np.nan)
    df['Body_Pct'] = (body / candle_range) * 100
    df['Is_Green'] = (close >= open_price).astype(int)

    green = df['Is_Green']
    consec = [0] * len(green)
    for i in range(1, len(green)):
        if green.iloc[i] == green.iloc[i - 1]:
            consec[i] = consec[i - 1] + (1 if green.iloc[i] == 1 else -1)
        else:
            consec[i] = 1 if green.iloc[i] == 1 else -1
    df['Consec_Candles'] = consec

    # ========================
    # 11. PIVOT POINTS
    # ========================
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    pivot = (prev_high + prev_low + prev_close) / 3
    df['Pivot'] = pivot
    df['Pivot_R1'] = (2 * pivot) - prev_low
    df['Pivot_S1'] = (2 * pivot) - prev_high
    df['Pivot_R2'] = pivot + (prev_high - prev_low)
    df['Pivot_S2'] = pivot - (prev_high - prev_low)

    # Round floats
    float_cols = df.select_dtypes(include=[np.floating]).columns
    df[float_cols] = df[float_cols].round(2)

    return df


def format_indicator_summary(df):
    """Create concise text summary of latest indicator values."""
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    lines = []
    lines.append("=" * 50)
    lines.append("INDICATOR SUMMARY (Latest Candle)")
    lines.append("=" * 50)

    # Price
    lines.append(f"PRICE: O={latest['open']} H={latest['high']} L={latest['low']} C={latest['close']} V={latest['volume']}")

    # EMAs
    ema5 = latest.get('EMA_5', 'N/A')
    ema9 = latest.get('EMA_9', 'N/A')
    ema15 = latest.get('EMA_15', 'N/A')
    ema21 = latest.get('EMA_21', 'N/A')
    lines.append(f"EMA: 5={ema5} 9={ema9} 15={ema15} 21={ema21}")

    try:
        aligned = "YES" if float(ema5) > float(ema9) > float(ema15) else "NO"
    except (ValueError, TypeError):
        aligned = "?"
    lines.append(f"EMA Aligned (5>9>15): {aligned}")

    # VWAP
    vwap = latest.get('VWAP', 'N/A')
    lines.append(f"VWAP: {vwap} | Price vs VWAP: {'ABOVE' if latest['close'] > (vwap if isinstance(vwap, (int, float)) else 0) else 'BELOW'}")

    # RSI
    rsi = latest.get('RSI_14', 'N/A')
    rsi_zone = "OVERSOLD" if isinstance(rsi, (int, float)) and rsi < 30 else \
               "OVERBOUGHT" if isinstance(rsi, (int, float)) and rsi > 70 else "NEUTRAL"
    lines.append(f"RSI(14): {rsi} [{rsi_zone}]")

    # Stochastic
    sk = latest.get('STOCH_K', 'N/A')
    sd = latest.get('STOCH_D', 'N/A')
    stoch_cross = "?"
    try:
        pk = prev.get('STOCH_K', 0) or 0
        pd_val = prev.get('STOCH_D', 0) or 0
        if pk < pd_val and float(sk) > float(sd):
            stoch_cross = "BULLISH CROSS"
        elif pk > pd_val and float(sk) < float(sd):
            stoch_cross = "BEARISH CROSS"
        else:
            stoch_cross = "No cross"
    except (ValueError, TypeError):
        pass
    lines.append(f"STOCH: K={sk} D={sd} [{stoch_cross}]")

    # MACD
    macd_h = latest.get('MACD_Hist', 'N/A')
    prev_h = prev.get('MACD_Hist', 0) or 0
    macd_trend = "?"
    try:
        mh = float(macd_h)
        if mh > 0 and mh > prev_h:
            macd_trend = "POSITIVE & EXPANDING"
        elif mh > 0:
            macd_trend = "POSITIVE but shrinking"
        elif mh < 0 and mh < prev_h:
            macd_trend = "NEGATIVE & EXPANDING"
        else:
            macd_trend = "NEGATIVE but recovering"
    except (ValueError, TypeError):
        pass
    lines.append(f"MACD: Line={latest.get('MACD', 'N/A')} Signal={latest.get('MACD_Signal', 'N/A')} Hist={macd_h} [{macd_trend}]")

    # Bollinger
    lines.append(f"BB: Upper={latest.get('BB_Upper', 'N/A')} Mid={latest.get('BB_Middle', 'N/A')} Lower={latest.get('BB_Lower', 'N/A')} %B={latest.get('BB_PctB', 'N/A')}")

    # ATR
    lines.append(f"ATR(14): {latest.get('ATR_14', 'N/A')}")

    # ADX
    adx = latest.get('ADX', 'N/A')
    adx_str = "?"
    try:
        a = float(adx)
        adx_str = "STRONG TREND" if a > 40 else "TRENDING" if a > 25 else "WEAK" if a > 20 else "NO TREND"
    except (ValueError, TypeError):
        pass
    lines.append(f"ADX: {adx} +DI={latest.get('Plus_DI', 'N/A')} -DI={latest.get('Minus_DI', 'N/A')} [{adx_str}]")

    # Volume
    vol_ratio = latest.get('Vol_Ratio', 'N/A')
    vol_str = "?"
    try:
        vr = float(vol_ratio)
        vol_str = "VERY HIGH" if vr > 2.0 else "ABOVE AVG" if vr > 1.3 else "AVERAGE" if vr > 0.7 else "LOW"
    except (ValueError, TypeError):
        pass
    lines.append(f"VOLUME: Current={latest['volume']} MA20={latest.get('Vol_MA_20', 'N/A')} Ratio={vol_ratio} [{vol_str}]")

    # Supertrend
    st_dir = latest.get('Supertrend_Dir', 'N/A')
    st_label = "BULLISH" if st_dir == 1 else "BEARISH" if st_dir == -1 else "?"
    lines.append(f"SUPERTREND: {latest.get('Supertrend', 'N/A')} [{st_label}]")

    # OBV
    obv_curr = latest.get('OBV', 0) or 0
    obv_prev = prev.get('OBV', 0) or 0
    lines.append(f"OBV: {obv_curr} [{'RISING' if obv_curr > obv_prev else 'FALLING'}]")

    # Candle
    lines.append(f"CANDLE: {'GREEN' if latest.get('Is_Green', 0) == 1 else 'RED'} Body={latest.get('Body_Pct', 'N/A')}% Streak={latest.get('Consec_Candles', 'N/A')}")

    # Pivots
    lines.append(f"PIVOTS: R2={latest.get('Pivot_R2', 'N/A')} R1={latest.get('Pivot_R1', 'N/A')} P={latest.get('Pivot', 'N/A')} S1={latest.get('Pivot_S1', 'N/A')} S2={latest.get('Pivot_S2', 'N/A')}")

    # ========================
    # MULTI-FACTOR SCORE
    # ========================
    score = 0
    total = 0

    def safe_gt(a, b):
        try:
            return float(a or 0) > float(b or 0)
        except (ValueError, TypeError):
            return False

    def safe_between(val, lo, hi):
        try:
            v = float(val or 0)
            return lo <= v <= hi
        except (ValueError, TypeError):
            return False

    checks = {
        "EMA 5>9>15": aligned == "YES",
        "Price > VWAP": safe_gt(latest['close'], latest.get('VWAP', 0)),
        "RSI 30-65": safe_between(rsi, 30, 65),
        "Stoch K>D": safe_gt(sk, sd),
        "Stoch <80": not safe_gt(sk, 80),
        "MACD Hist>0": safe_gt(macd_h, 0),
        "ADX >20": safe_gt(adx, 20),
        "+DI > -DI": safe_gt(latest.get('Plus_DI', 0), latest.get('Minus_DI', 0)),
        "Vol > MA20": safe_gt(vol_ratio, 1.0),
        "Supertrend Bull": st_dir == 1,
        "Green Candle": latest.get('Is_Green', 0) == 1,
        "OBV Rising": obv_curr > obv_prev,
    }

    lines.append("")
    lines.append("VALIDATION CHECKLIST:")
    for name, passed in checks.items():
        total += 1
        if passed:
            score += 1
        lines.append(f"  {'PASS' if passed else 'FAIL'} - {name}")

    lines.append(f"SCORE: {score}/{total} ({score / total * 100:.0f}%)")

    return "\n".join(lines)


def llm_trade_signaldd(symbol, ohlcv, timeframe="3m"):
    """
    Compute ALL technical indicators and send to Groq LLM for trade validation.
    Handles None data, different column formats, and all edge cases.
    """

    SAFE_DEFAULT = {
        "signal": "no trade",
        "confidence": 0.0,
        "reason": "LLM call failed",
        "bias": "Neutral",
        "entry": 0.0,
        "stop_loss": 0.0,
        "target_1": 0.0,
        "target_2": 0.0
    }

    try:
        print(f"\n🤖 Calling LLM for {symbol}...")

        # ===========================================================
        #  VALIDATE INPUT DATA
        # ===========================================================
        if ohlcv is None:
            print("❌ OHLCV data is None! Cannot proceed.")
            SAFE_DEFAULT["reason"] = "OHLCV data is None"
            return SAFE_DEFAULT

        if isinstance(ohlcv, (list, tuple)) and len(ohlcv) == 0:
            print("❌ OHLCV data is empty list!")
            SAFE_DEFAULT["reason"] = "OHLCV data is empty"
            return SAFE_DEFAULT

        if isinstance(ohlcv, pd.DataFrame) and len(ohlcv) == 0:
            print("❌ OHLCV DataFrame is empty!")
            SAFE_DEFAULT["reason"] = "OHLCV DataFrame is empty"
            return SAFE_DEFAULT

        # ===========================================================
        #  COMPUTE INDICATORS
        # ===========================================================
        df = compute_all_indicators(ohlcv)

        if df.empty or len(df) < 5:
            print(f"⚠️ Not enough data: {len(df)} rows")
            SAFE_DEFAULT["reason"] = f"Insufficient data: {len(df)} rows"
            return SAFE_DEFAULT

        # ===========================================================
        #  FORMAT CANDLE DATA (last 30 candles)
        # ===========================================================
        recent_df = df.tail(30)

        candle_text = ""
        for _, row in recent_df.iterrows():
            candle_text += (
                f"T={row.get('time', '?')} "
                f"O={row['open']} H={row['high']} L={row['low']} C={row['close']} V={row['volume']} | "
                f"EMA5={row.get('EMA_5', '?')} EMA9={row.get('EMA_9', '?')} EMA15={row.get('EMA_15', '?')} | "
                f"RSI={row.get('RSI_14', '?')} StK={row.get('STOCH_K', '?')} StD={row.get('STOCH_D', '?')} | "
                f"MACD_H={row.get('MACD_Hist', '?')} VolR={row.get('Vol_Ratio', '?')}\n"
            )

        if not candle_text.strip():
            SAFE_DEFAULT["reason"] = "No candle data formatted"
            return SAFE_DEFAULT

        # ===========================================================
        #  INDICATOR SUMMARY
        # ===========================================================
        indicator_summary = format_indicator_summary(df)
        print(f"📊 Indicators computed successfully")

        # ===========================================================
        #  PROMPT
        # ===========================================================
        prompt = f"""You are a very selective 3-minute timeframe intraday options trader in Indian markets (Nifty/BankNifty heavy). Your edge comes from waiting for high-conviction oversold recoveries with momentum confirmation. You aim for 90%+ win rate by being ultra-conservative.

SYMBOL: {symbol}
TIMEFRAME: {timeframe}

CANDLE DATA WITH INDICATORS (latest at bottom, last 30 candles):
{candle_text}

{indicator_summary}

MY TECHNICAL ANALYSIS has generated a BUY signal based on:
- Stochastic K crossed above D from oversold zone (<25)
- Green candle confirmation
- EMA alignment (5 > 9 > 15) is favorable
- K has room to run (not overbought)

YOUR TASK — Use ALL provided indicators:
1. Verify: EMAs aligned, RSI 30-65, Stoch crossover, MACD positive, Vol > MA20
2. Check trend: ADX > 20, +DI > -DI, Supertrend bullish
3. Warnings: Bearish divergence, resistance at pivots, overbought signals, low volume
4. Risk-reward: Use ATR for stop, pivot levels for targets, minimum 1:2 RR
5. Only BUY if majority of checks pass. Otherwise NO TRADE.

RESPOND ONLY with valid JSON:
{{
  "bias": "Bullish" or "Bearish" or "Neutral",
  "signal": "BUY" or "SELL" or "NO TRADE",
  "entry": number,
  "stop_loss": number,
  "target_1": number,
  "target_2": number,
  "confidence": number between 0.0 and 1.0,
  "reason": "1-2 sentence explanation"
}}

CONFIDENCE: 0.85-1.0 = perfect setup, 0.70-0.84 = good, 0.50-0.69 = mixed, below 0.50 = avoid."""

        # ===========================================================
        #  API CALL
        # ===========================================================
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "temperature": 0.1,
                "max_tokens": 400,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a trading analyst. Respond ONLY in valid JSON. No markdown."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            timeout=30
        )

        # ===========================================================
        #  HANDLE API RESPONSE
        # ===========================================================
        if response.status_code != 200:
            print(f"❌ Groq API error: {response.status_code}")
            print(f"   Details: {response.text[:500]}")
            SAFE_DEFAULT["reason"] = f"API error: {response.status_code}"
            return SAFE_DEFAULT

        data = response.json()

        if "choices" not in data:
            print(f"❌ Invalid response: {data}")
            SAFE_DEFAULT["reason"] = "Invalid API response"
            return SAFE_DEFAULT

        content = data["choices"][0]["message"]["content"]
        print(f"📝 Raw LLM: {content}")

        # ===========================================================
        #  PARSE JSON
        # ===========================================================
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        start = content.index("{")
        end = content.rindex("}") + 1
        json_str = content[start:end]
        result = json.loads(json_str)

        # ===========================================================
        #  NORMALIZE
        # ===========================================================
        raw_signal = str(result.get("signal", "NO TRADE")).strip().upper()
        valid_signals = {
            "BUY": "buy", "SELL": "sell", "NO TRADE": "no trade",
            "HOLD": "no trade", "NEUTRAL": "no trade"
        }
        signal = valid_signals.get(raw_signal, "no trade")

        try:
            confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
        except (ValueError, TypeError):
            confidence = 0.0

        reason = str(result.get("reason", "No reason"))
        bias = str(result.get("bias", "Neutral")).strip()
        if bias not in ["Bullish", "Bearish", "Neutral"]:
            bias = "Neutral"

        def safe_float(val, default=0.0):
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        final = {
            "signal": signal,
            "confidence": confidence,
            "reason": reason,
            "bias": bias,
            "entry": safe_float(result.get("entry")),
            "stop_loss": safe_float(result.get("stop_loss")),
            "target_1": safe_float(result.get("target_1")),
            "target_2": safe_float(result.get("target_2"))
        }

        print(f"✅ LLM: signal={signal} conf={confidence} bias={bias}")
        print(f"   {reason}")
        return final

    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        SAFE_DEFAULT["reason"] = f"JSON parse error: {e}"
        return SAFE_DEFAULT

    except requests.exceptions.Timeout:
        print("❌ Timeout")
        SAFE_DEFAULT["reason"] = "Request timed out"
        return SAFE_DEFAULT

    except requests.exceptions.ConnectionError:
        print("❌ Connection error")
        SAFE_DEFAULT["reason"] = "Connection error"
        return SAFE_DEFAULT

    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        SAFE_DEFAULT["reason"] = f"Error: {e}"
        return SAFE_DEFAULT


def super_trend(symbol, data, use_llm=True):
    """
    LONG ONLY: OVERSOLD + K CROSSOVER D STRATEGY
    ==============================================

    PARAMS:
        symbol   : Stock symbol
        data     : DataFrame with OHLCV
        use_llm  : True  → Filter through LLM
                   False → Raw technical signals only
    """

    # ===========================================================
    # INDICATORS
    # ===========================================================
    stoch = ta.stoch(data['High'], data['Low'], data['Close'], 14, 3, 3)
    data['Stoch_K'] = stoch['STOCHk_14_3_3']
    data['Stoch_D'] = stoch['STOCHd_14_3_3']

    data['EMA5'] = ta.ema(data['Close'], length=5)
    data['EMA9'] = ta.ema(data['Close'], length=9)
    data['EMA15'] = ta.ema(data['Close'], length=15)

    K = data['Stoch_K']
    D = data['Stoch_D']
    close = data['Close']
    low = data['Low']
    ema5 = data['EMA5']
    ema9 = data['EMA9']
    ema15 = data['EMA15']

    k_vel = K - K.shift(1)

    # ===========================================================
    # TIME FILTER
    # ===========================================================
    if isinstance(data.index, pd.DatetimeIndex):
        time_series = data.index.time
    else:
        time_series = pd.to_datetime(data.index).time

    data['time_mins'] = [t.hour * 60 + t.minute for t in time_series]
    TIME_OK = (data['time_mins'] >= 575) & (data['time_mins'] <= 910)

    # ===========================================================
    # CANDLE BASICS
    # ===========================================================
    green = close > data['Open']

    # ===========================================================
    # K CROSSOVER D
    # ===========================================================
    prev_k_below_d = K.shift(1) < D.shift(1)
    curr_k_above_d = K > D
    prev_k_equal_d = K.shift(1) == D.shift(1)
    k_crossover_d_v2 = (prev_k_below_d | prev_k_equal_d) & curr_k_above_d

    # ===========================================================
    # OVERSOLD
    # ===========================================================
    OVERSOLD_LEVEL = 25

    was_oversold = pd.Series(False, index=data.index)
    for i in range(0, 15):
        was_oversold |= (K.shift(i) < OVERSOLD_LEVEL)

    bars_in_oversold = pd.Series(0, index=data.index, dtype=float)
    for i in range(0, 15):
        bars_in_oversold += (K.shift(i) < 30).astype(int)
    sufficient_oversold = bars_in_oversold >= 2

    # ===========================================================
    # CONFIRMATIONS
    # ===========================================================
    k_has_room = K < 80
    ema5_rising = ema5 > ema5.shift(1)

    # ===========================================================
    # PRIMARY LONG SIGNAL
    # ===========================================================
    LONG_SIGNAL = (
        TIME_OK &
        was_oversold &
        sufficient_oversold &
        k_crossover_d_v2 &
        k_has_room &
        green
    )

    # ===========================================================
    # EMA BOUNCES
    # ===========================================================
    k_above_d = K > D
    k_rising = k_vel > 0
    recovery_context = (K > 30) & (K < 70) & was_oversold & k_above_d

    near_ema9 = (abs(low - ema9) / ema9 * 100) < 0.15
    touch_ema9 = low <= ema9 * 1.001
    bounce_off_ema9 = (touch_ema9 | near_ema9) & (close > ema9) & green
    ema9_rising = ema9 > ema9.shift(1)

    LONG_EMA9_BOUNCE = (
        TIME_OK &
        bounce_off_ema9 &
        recovery_context &
        ema9_rising &
        k_rising &
        (close > ema5)
    )

    near_ema15 = (abs(low - ema15) / ema15 * 100) < 0.2
    touch_ema15 = low <= ema15 * 1.002
    bounce_off_ema15 = (touch_ema15 | near_ema15) & (close > ema15) & green
    ema15_rising = ema15 > ema15.shift(1)
    strong_bounce = bounce_off_ema15 & (close > close.shift(1)) & (close > ema9)

    LONG_EMA15_BOUNCE = (
        TIME_OK &
        strong_bounce &
        recovery_context &
        ema15_rising &
        k_rising &
        k_above_d
    )

    # ===========================================================
    # REJECTIONS
    # ===========================================================
    LONG_REJECT = (K > 80) | (k_vel < -3)

    # ===========================================================
    # COMBINE
    # ===========================================================
    long_raw = (
        (LONG_SIGNAL | LONG_EMA9_BOUNCE | LONG_EMA15_BOUNCE) &
        ~LONG_REJECT
    ).astype(int)

    # ===========================================================
    # GRADING
    # ===========================================================
    data['signal_grade'] = ""
    data.loc[(long_raw == 1) & LONG_SIGNAL & ema5_rising & (close > ema5), 'signal_grade'] = "A+"
    data.loc[(long_raw == 1) & (data['signal_grade'] == "") & LONG_SIGNAL, 'signal_grade'] = "A"
    data.loc[(long_raw == 1) & (data['signal_grade'] == ""), 'signal_grade'] = "B"

    # ===========================================================
    # REASONS
    # ===========================================================
    data["signal_reason"] = ""
    data.loc[LONG_SIGNAL & (long_raw == 1), "signal_reason"] = "BUY: Oversold + K Crossed Over D"
    data.loc[LONG_EMA9_BOUNCE & (long_raw == 1) & (data["signal_reason"] == ""), "signal_reason"] = "BUY: 9 EMA Bounce"
    data.loc[LONG_EMA15_BOUNCE & (long_raw == 1) & (data["signal_reason"] == ""), "signal_reason"] = "BUY: 15 EMA Bounce"

    data.loc[data['signal_grade'] == "A+", 'signal_reason'] += " [★★★★]"
    data.loc[data['signal_grade'] == "A", 'signal_reason'] += " [★★★]"
    data.loc[data['signal_grade'] == "B", 'signal_reason'] += " [★★]"

    mask_long = (long_raw == 1)
    data.loc[mask_long, 'signal_reason'] += (
        "  K=" + K.round(1).astype(str) +
        " D=" + D.round(1).astype(str) +
        " vel=" + k_vel.round(1).astype(str)
    )

    # ===========================================================
    # COOLDOWN
    # ===========================================================
    cooldown = 5
    long_recent = long_raw.shift(1).rolling(cooldown).sum().fillna(0)
    long_after_cooldown = ((long_raw == 1) & (long_recent == 0)).astype(int)

    # ===========================================================
    # INIT COLUMNS
    # ===========================================================
    data['st_sig_raw'] = long_after_cooldown
    data['confidence'] = 0.0
    data['llm_signal'] = None
    data['llm_reason'] = ""
    data['llm_entry'] = 0.0
    data['llm_stop_loss'] = 0.0
    data['llm_target_1'] = 0.0
    data['llm_target_2'] = 0.0
    data['llm_bias'] = ""

    # ===========================================================
    # MODE: WITHOUT LLM
    # ===========================================================
    if not use_llm:
        print("📊 LLM Mode: OFF")
        data['st_sig'] = long_after_cooldown
        data.loc[data['st_sig'] == 1, 'signal_reason'] += " | 🔧 NO LLM"
        return data

    # ===========================================================
    # MODE: WITH LLM
    # ===========================================================
    print("🤖 LLM Mode: ON")

    llm_confidence = 0.0
    llm_signal = None
    llm_reason = ""
    llm_entry = 0.0
    llm_stop_loss = 0.0
    llm_target_1 = 0.0
    llm_target_2 = 0.0
    llm_bias = ""

    if long_after_cooldown.iloc[-1] == 1:
        print("=" * 60)
        print(f"🔔 RAW BUY SIGNAL: {symbol}")
        print(f"   Grade:  {data['signal_grade'].iloc[-1]}")
        print(f"   Reason: {data['signal_reason'].iloc[-1]}")
        print("=" * 60)

        try:
            ohlcv = fetch_ohlcv(symbol)

            if not ohlcv:
                print("⚠️  No OHLCV data for LLM")
            else:
                llm_result = llm_trade_signal(symbol, ohlcv, "3m")

                print("========== LLM RESULT ==========")
                print(f"  Result: {llm_result}")

                # Extract values (LLM now returns lowercase signal)
                llm_signal = str(llm_result.get("signal", "no trade")).strip()
                llm_confidence = float(llm_result.get("confidence", 0.0))
                llm_reason = str(llm_result.get("reason", ""))
                llm_bias = str(llm_result.get("bias", ""))
                llm_entry = float(llm_result.get("entry", 0.0))
                llm_stop_loss = float(llm_result.get("stop_loss", 0.0))
                llm_target_1 = float(llm_result.get("target_1", 0.0))
                llm_target_2 = float(llm_result.get("target_2", 0.0))

                print(f"  Bias:       {llm_bias}")
                print(f"  Signal:     {llm_signal}")
                print(f"  Confidence: {llm_confidence}")
                print(f"  Entry:      {llm_entry}")
                print(f"  Stop Loss:  {llm_stop_loss}")
                print(f"  Target 1:   {llm_target_1}")
                print(f"  Target 2:   {llm_target_2}")
                print(f"  Reason:     {llm_reason}")

                # Decision
                if llm_signal == "buy" and llm_confidence >= 0.7:
                    print(f"  ✅ LLM CONFIRMS BUY (conf={llm_confidence})")
                elif llm_signal == "buy":
                    print(f"  ⚠️  LOW conf ({llm_confidence} < 0.7) — BLOCKED")
                elif llm_signal == "sell":
                    print(f"  ❌ LLM says SELL — BLOCKED")
                else:
                    print(f"  ⏸️  LLM says '{llm_signal}' — BLOCKED")

                print("=" * 45)

        except Exception as e:
            print(f"❌ LLM error: {e}")

    # Store values
    data.loc[data.index[-1], 'confidence'] = llm_confidence
    data.loc[data.index[-1], 'llm_signal'] = llm_signal
    data.loc[data.index[-1], 'llm_reason'] = llm_reason
    data.loc[data.index[-1], 'llm_bias'] = llm_bias
    data.loc[data.index[-1], 'llm_entry'] = llm_entry
    data.loc[data.index[-1], 'llm_stop_loss'] = llm_stop_loss
    data.loc[data.index[-1], 'llm_target_1'] = llm_target_1
    data.loc[data.index[-1], 'llm_target_2'] = llm_target_2
    #tele_msg("Long Entry Taken llm data  "+str(llm_confidence)+" confidence "+str(llm_signal)+"  reason "+str(llm_reason)+"   llm_entry "+str(llm_entry)+" llm_target_1 "+str(llm_target_1)+"  llm_target_2 "+str(llm_target_2))
    #tele_msg("Short Entry Taken For "+i+" Total Quantity "+str(Trade_quantity)+" And the Target Price is "+str(Target_Price))
    # ===========================================================
    # FINAL SIGNAL
    # ===========================================================
    data['st_sig'] = (
        (long_after_cooldown == 1) &
        (data['confidence'] >= 0.7) &
        (data['llm_signal'] == 'buy')
    ).astype(int)

    # Tag reasons
    final_buy = data['st_sig'] == 1
    data.loc[final_buy, 'signal_reason'] += (
        f" | ✅ LLM=BUY conf={llm_confidence:.2f}"
        f" entry={llm_entry} sl={llm_stop_loss}"
    )

    blocked = (long_after_cooldown == 1) & (data['st_sig'] == 0)
    data.loc[blocked, 'signal_reason'] += (
        f" | ❌ BLOCKED ({llm_signal}, conf={llm_confidence:.2f})"
    )

    # Summary
    if long_after_cooldown.iloc[-1] == 1:
        if data['st_sig'].iloc[-1] == 1:
            print("\n" + "🟢" * 15)
            print(f"✅ TAKING BUY: {symbol}")
            print(f"   Conf: {llm_confidence:.2f}")
            print(f"   Entry: {llm_entry} | SL: {llm_stop_loss}")
            #tele_msg("Long Entry Taken llm data  "+str(llm_confidence)+" confidence "+str(llm_signal)+"  reason "+str(llm_reason)+"   llm_entry "+str(llm_entry)+" llm_target_1 "+str(llm_target_1)+"  llm_target_2 "+str(llm_target_2))
            print("🟢" * 15 + "\n")
        else:
            print("\n" + "🔴" * 15)
            print(f"❌ BLOCKED: {symbol}")
            print(f"   Signal: {llm_signal} | Conf: {llm_confidence:.2f}")
            print("🔴" * 15 + "\n")

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
