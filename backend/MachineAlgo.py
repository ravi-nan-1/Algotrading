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
                                             From=dt.date.today()-dt.timedelta(2), To=dt.date.today()))

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
def fetch_ohlcv(symbol, timeframe="3m", candles=20):
    print('call11 fetch_ohlcv')
    scripcode = scripcode_lookup(symbol)
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



GROQ_API_KEY = os.getenv("GROQ_API_KEY")
def llm_trade_signal(symbol, ohlcv, timeframe):
    print('call11 fetch_ohlcv')

    prompt = f"""
You are a professional discretionary trader.

Analyze OHLCV data for {symbol} on {timeframe}.
Trade only if quality is high.

OHLCV:
{json.dumps(ohlcv)}

Respond ONLY with valid JSON. No explanation. No markdown.

Format:
{{
  "bias": "Bullish | Bearish | Neutral",
  "signal": "BUY | SELL | NO TRADE",
  "entry": number,
  "stop_loss": number,
  "target_1": number,
  "target_2": number,
  "confidence": number,
  "reason": "short explanation"
}}
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=30
    )

    # 🔥 HANDLE API FAILURE
    if response.status_code != 200:
        return {"error": "Groq API error", "details": response.text}

    data = response.json()

    # 🔥 HANDLE MISSING CHOICES
    if "choices" not in data:
        return {"error": "Invalid Groq response", "raw": data}

    content = data["choices"][0]["message"]["content"]

    # 🔥 EXTRACT JSON SAFELY
    try:
        start = content.index("{")
        end = content.rindex("}") + 1
        json_str = content[start:end]
        return json.loads(json_str)
    except Exception as e:
        return {
            "error": "LLM did not return valid JSON",
            "raw_response": content
        }






def super_trend(symbol, data):

    # ==========================
    # CORE INDICATORS (UNCHANGED)
    # ==========================
    data['EMA9'] = ta.ema(data['Close'], 9)
    data['EMA5'] = ta.ema(data['Close'], 5)
    data['EMA20'] = ta.ema(data['Close'], 20)
    data['EMA50'] = ta.ema(data['Close'], 50)

    data['RSI'] = ta.rsi(data['Close'], 14)
    num = (data['Close']-data['Open']).rolling(10).mean()
    den = (data['High']-data['Low']).rolling(10).mean()

    data['RVGI'] = num / den
    data['RVGI_Signal'] = data['RVGI'].rolling(4).mean()
    rvgi_strength_ok = (
            (data['RVGI'] > data['RVGI_Signal']) &
            (data['RVGI'] > data['RVGI'].shift(1))
    )
    stoch = ta.stoch(data['High'], data['Low'], data['Close'], 14, 3, 3)
    data['Stoch_K'] = stoch['STOCHk_14_3_3']
    data['Stoch_D'] = stoch['STOCHd_14_3_3']

    adx_df = ta.adx(data['High'], data['Low'], data['Close'], 14)
    data['ADX'] = adx_df['ADX_14']
    data['Plus_DI'] = adx_df['DMP_14']
    data['Minus_DI'] = adx_df['DMN_14']

    bb = ta.bbands(data['Close'], 20, 2)
    print(bb.columns)
    data['BB_lower'] = bb['BBL_20_2_2.0']
    data['BB_upper'] = bb['BBU_20_2_2.0']
    data['BB_mid'] = bb['BBM_20_2_2.0']

    data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], 14)
    data['momentum']=(data['Close'].shift(1) - data['Close'].shift(2))
    data['momentum1'] = (data['Close']-data['Close'].shift(1))
    data['StockKmom'] = (data['Stoch_K'] - data['Stoch_K'].shift(1))
    data['Volume_MA'] = data['Volume'].rolling(20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
    data['touches'] = 0
    data[['High', 'Low', 'Close', 'Volume']] = data[['High', 'Low', 'Close', 'Volume']].astype(float)

    # Typical Price
    tp = (data['High']+data['Low']+data['Close']) / 3

    # VWAP (session-based cumulative)
    data['VWAP'] = (tp * data['Volume']).cumsum() / data['Volume'].cumsum()
    # ==========================
    # TIME FILTER (UNCHANGED)
    # ==========================
    if isinstance(data.index, pd.DatetimeIndex):
        time_series = data.index.time
    else:
        time_series = pd.to_datetime(data.index).time

    data['time_mins'] = [t.hour * 60 + t.minute for t in time_series]
    TIME_OK = (data['time_mins'] >= 570) & (data['time_mins'] <= 885)
    opt_type = data["Option_Type"].iloc[0]
    # =============================================================
    # SELECT CE OR PE LOGIC BLOCK
    # =============================================================
    stoch_deep_oversold_bounce = (
            (data['Stoch_K'].shift(1) < 25) &
            (data['Stoch_D'].shift(1) < 25) &
            (data['Stoch_K']-data['Stoch_K'].shift(1) > 3) &
            (
                    ((data['Stoch_K'] > data['Stoch_D']) &
                     (data['Stoch_K'].shift(1) <= data['Stoch_D'].shift(1))) |
                    ((data['Stoch_K'] > data['Stoch_K'].shift(1)+3) &
                     (data['Stoch_K'] > 15))
            )
    )

    stoch_oversold_recovery = (
            (data['Stoch_K'].shift(1) < 40) &
            (data['Stoch_D'].shift(1) < 40) &
            (data['Stoch_K'] > 20) &
            (data['Stoch_D'] > 20) &
            (data['Stoch_K'] > data['Stoch_D']) &
            (data['Stoch_K'].shift(1) <= data['Stoch_D'].shift(1)) &
            ((data['Stoch_K']-data['Stoch_K'].shift(1)) > 5)
    )

    stoch_early_bounce = (
            (data['Stoch_K'] < 20) &
            (data['Stoch_K'].shift(1) >= 8) &
            (data['Stoch_D'] < 25) &
            (data['RSI'] > data['RSI'].shift(1)) &
            (data['Close'] > data['Open']) &
            ((data['Stoch_K']-data['Stoch_K'].shift(1)) > 3) &
            (data['Stoch_K'] > data['Stoch_D'])
    )

    stoch_buy_signal = (
            stoch_deep_oversold_bounce |
            stoch_oversold_recovery |
            stoch_early_bounce
    )

    # === VOLUME ANALYSIS ===
    data['Volume_MA'] = data['Volume'].rolling(20).mean()
    volume_surge = data['Volume'] > (1.5 * data['Volume_MA'])
    volume_positive = data['Volume'] > data['Volume_MA']

    # === TREND FILTERS ===
    uptrend = (
            (data['EMA5'] > data['EMA20']) &
            (data['EMA20'] > data['EMA50']) &
            (data['ADX'] > 25)
    )

    short_term_bullish = data['Close'] > data['EMA5']

    momentum_positive = (
            (data['RSI'] > 45) &
            (data['RSI'] < 70)
    )

    if opt_type == "CE":

        # ============================
        # CE SETUPS START HERE
        # ============================

        # ---------- SETUP 1 ----------
        stoch_oversold = data['Stoch_K'] < 20
        stoch_turning = data['Stoch_K'] > data['Stoch_K'].shift(1)
        stoch_cross = data['Stoch_K'] > data['Stoch_D']

        rsi_ok = (data['RSI'] > 30) & (data['RSI'] < 52)
        rsi_turn = data['RSI'] > data['RSI'].shift(1)

        green = data['Close'] > data['Open']
        higher_close = data['Close'] > data['Close'].shift(1)
        momentum_ok = (data['Close'].shift(1) - data['Close'].shift(2)) >1


        volume_ok = data['Volume_Ratio'] > 0.75
        volume_strong = data['Volume_Ratio'] > 1.2

        # ---------- STOCHASTIC WAIT + TRIGGER ----------

        stoch_oversold = data['Stoch_K'] < 20

        WAIT_BARS = 5
        was_oversold = stoch_oversold.rolling(WAIT_BARS).max() == 1

        stoch_k_above_d = data['Stoch_K'] > data['Stoch_D']
        stoch_cross_20 = (data['Stoch_K'] > 20) & (data['Stoch_K'].shift(1) <= 20)

        # ---------- RSI ----------
        rsi_ok = (data['RSI'] > 30) & (data['RSI'] < 55)
        rsi_turn = data['RSI'] > data['RSI'].shift(1)

        # ---------- PRICE ----------
        green = data['Close'] > data['Open']
        higher_close = data['Close'] > data['Close'].shift(1)

        # ---------- VOLUME ----------
        volume_ok = data['Volume_Ratio'] > 0.75
        volume_strong = data['Volume_Ratio'] > 1.2

        # ---------- SETUPS ----------
        SETUP_1A = (
                TIME_OK &
                was_oversold &
                stoch_k_above_d &
                stoch_cross_20 &
                rsi_ok &
                rsi_turn &
                (green | higher_close) &
                volume_strong
        )

        SETUP_1B = (
                TIME_OK &
                was_oversold &
                stoch_k_above_d &
                stoch_cross_20 &
                rsi_ok &
                (green | higher_close) &
                volume_ok &
                ~volume_strong
        )

        # ----------------- CE SETUP 2 -----------------
        # ---------- BOLLINGER REJECTION ----------
        bb_touch = (
                (data['Low'] <= data['BB_lower'] * 1.002) &
                (data['Close'] > data['BB_lower'])
        )

        # ---------- STRONG BUYER CANDLE ----------
        body = data['Close']-data['Open']
        candle_range = data['High']-data['Low']

        good_candle = (
                (body > 0) &
                (body > candle_range * 0.4) &
                (data['Close'] > data['Close'].shift(1))
        )

        # ---------- STOCHASTIC WAIT + CONFIRM ----------
        was_oversold = (data['Stoch_K'] < 20).rolling(5).max() == 1

        stoch_confirm = (
                (data['Stoch_K'] > data['Stoch_D']) &
                (data['Stoch_K'] > 20) &
                (data['Stoch_K'].shift(1) <= 20)
        )

        # ---------- VOLATILITY EXPANSION (OPTION KEY) ----------
        bb_width = data['BB_upper']-data['BB_lower']
        bb_expanding = bb_width > bb_width.shift(1)

        # ---------- VOLUME ----------
        volume_ok = data['Volume_Ratio'] > 0.75
        volume_strong = data['Volume_Ratio'] > 1.2

        # ---------- FINAL SETUPS ----------
        SETUP_2A = (
                TIME_OK &
                bb_touch &
                good_candle &
                was_oversold &
                stoch_confirm &
                bb_expanding &
                volume_strong
        )

        SETUP_2B = (
                TIME_OK &
                bb_touch &
                good_candle &
                was_oversold &
                (data['Stoch_K'] > data['Stoch_D']) &
                bb_expanding &
                volume_ok &
                ~volume_strong
        )



        # ----------------- CE SETUP 3 -----------------
        near_ema = (data['Low'] <= data['EMA20'] * 1.005) & (data['Close'] > data['EMA20'])
        ema_up = data['EMA9'] > data['EMA20']
        di_bull = data['Plus_DI'] > data['Minus_DI']
        stoch_pullback = (data['Stoch_K'] > 25) & (data['Stoch_K'] < 55)

        SETUP_3A = (
            TIME_OK &
            near_ema &
            (ema_up | di_bull) &
            stoch_pullback &
            stoch_cross &
            green &
            volume_strong &
            momentum_ok
        )

        SETUP_3B = (
            TIME_OK &
            near_ema &
            stoch_pullback &
            (stoch_cross | stoch_turning) &
            (green | higher_close) &
            volume_ok &
            ~volume_strong &
            momentum_ok
        )

        # ----------------- CE SETUP 4 -----------------
        # ---------- MARKET STRUCTURE ----------
        higher_low = data['Low'] > data['Low'].shift(1)
        hl_pattern = (
                higher_low &
                (data['Low'].shift(1) > data['Low'].shift(2))
        )

        # ---------- STOCHASTIC (MOMENTUM START, NOT LATE) ----------
        was_oversold = (data['Stoch_K'] < 25).rolling(6).max() == 1

        stoch_confirm = (
                (data['Stoch_K'] > data['Stoch_D']) &
                (data['Stoch_K'] > 25) &
                (data['Stoch_K'].shift(1) <= 25)
        )

        # ---------- TREND + ACCELERATION ----------
        adx_strong = data['ADX'] > 22
        adx_rising = data['ADX'] > data['ADX'].shift(1)

        strong_trend = adx_strong & adx_rising & di_bull

        # ---------- VOLATILITY EXPANSION (OPTION KEY) ----------
        atr_rising = data['ATR'] > data['ATR'].shift(1)

        # ---------- FINAL OPTION SETUP ----------
        SETUP_4 = (
                TIME_OK &
                hl_pattern &
                was_oversold &
                stoch_confirm &
                strong_trend &
                atr_rising &
                green &
                volume_ok
        )



        # ----------------- CE SETUP 5 -----------------
        vwap_stretch = data['Low'] < data['VWAP'] * 0.995  # ~0.5% below VWAP

        # 2️⃣ Reclaim VWAP (acceptance)
        vwap_reclaim = data['Close'] > data['VWAP']

        # 3️⃣ Strong bullish candle
        strong_reversal = (
                (data['Close'] > data['Open']) &
                ((data['Close']-data['Open']) > (data['High']-data['Low']) * 0.5)
        )

        # 4️⃣ Optional pullback confirmation
        vwap_pullback_hold = (
                (data['Low'] <= data['VWAP'] * 1.002) &
                (data['Close'] > data['VWAP'])
        )

        # -------- FINAL SETUP --------

        SETUP_5 = (
                TIME_OK &
                vwap_stretch &
                (vwap_reclaim | vwap_pullback_hold) &
                strong_reversal &
                momentum_ok
        )

    else:
        # =============================================================
        # PE SETUPS START HERE
        # =============================================================

        # ---------- SETUP 1 (PE) ----------
        stoch_oversold = data['Stoch_K'] < 25
        stoch_turning = data['Stoch_K'] > data['Stoch_K'].shift(1)
        stoch_cross = data['Stoch_K'] > data['Stoch_D']

        rsi_ok = (data['RSI'] > 30) & (data['RSI'] < 52)
        rsi_turn = data['RSI'] > data['RSI'].shift(1)

        green = data['Close'] > data['Open']
        higher_close = data['Close'] > data['Close'].shift(1)

        volume_ok = data['Volume_Ratio'] > 0.75
        volume_strong = data['Volume_Ratio'] > 1.2

        # ---------- STOCHASTIC WAIT + TRIGGER ----------

        stoch_oversold = data['Stoch_K'] < 20

        WAIT_BARS = 5
        was_oversold = stoch_oversold.rolling(WAIT_BARS).max() == 1

        stoch_k_above_d = data['Stoch_K'] > data['Stoch_D']
        stoch_cross_20 = (data['Stoch_K'] > 20) & (data['Stoch_K'].shift(1) <= 20)

        # ---------- RSI ----------
        rsi_ok = (data['RSI'] > 30) & (data['RSI'] < 55)
        rsi_turn = data['RSI'] > data['RSI'].shift(1)

        # ---------- PRICE ----------
        green = data['Close'] > data['Open']
        higher_close = data['Close'] > data['Close'].shift(1)

        # ---------- VOLUME ----------
        volume_ok = data['Volume_Ratio'] > 0.75
        volume_strong = data['Volume_Ratio'] > 1.2

        # ---------- SETUPS ----------
        SETUP_1A = (
                TIME_OK &
                was_oversold &
                stoch_k_above_d &
                stoch_cross_20 &
                rsi_ok &
                rsi_turn &
                (green | higher_close) &
                volume_strong
        )

        SETUP_1B = (
                TIME_OK &
                was_oversold &
                stoch_k_above_d &
                stoch_cross_20 &
                rsi_ok &
                (green | higher_close) &
                volume_ok &
                ~volume_strong
        )

        # ---------- SETUP 2 (PE) ----------
        # ---------- BOLLINGER REJECTION ----------
        bb_touch = (
                (data['Low'] <= data['BB_lower'] * 1.002) &
                (data['Close'] > data['BB_lower'])
        )

        # ---------- STRONG BUYER CANDLE ----------
        body = data['Close']-data['Open']
        candle_range = data['High']-data['Low']

        good_candle = (
                (body > 0) &
                (body > candle_range * 0.4) &
                (data['Close'] > data['Close'].shift(1))
        )

        # ---------- STOCHASTIC WAIT + CONFIRM ----------
        was_oversold = (data['Stoch_K'] < 20).rolling(5).max() == 1

        stoch_confirm = (
                (data['Stoch_K'] > data['Stoch_D']) &
                (data['Stoch_K'] > 20) &
                (data['Stoch_K'].shift(1) <= 20)
        )

        # ---------- VOLATILITY EXPANSION (OPTION KEY) ----------
        bb_width = data['BB_upper']-data['BB_lower']
        bb_expanding = bb_width > bb_width.shift(1)

        # ---------- VOLUME ----------
        volume_ok = data['Volume_Ratio'] > 0.75
        volume_strong = data['Volume_Ratio'] > 1.2

        # ---------- FINAL SETUPS ----------
        SETUP_2A = (
                TIME_OK &
                bb_touch &
                good_candle &
                was_oversold &
                stoch_confirm &
                bb_expanding &
                volume_strong
        )

        SETUP_2B = (
                TIME_OK &
                bb_touch &
                good_candle &
                was_oversold &
                (data['Stoch_K'] > data['Stoch_D']) &
                bb_expanding &
                volume_ok &
                ~volume_strong
        )



        # ---------- SETUP 3 (PE) ----------
        near_ema = (data['Low'] <= data['EMA20'] * 1.005) & (data['Close'] > data['EMA20'])
        ema_up = data['EMA9'] > data['EMA20']
        di_bull = data['Plus_DI'] > data['Minus_DI']
        stoch_pullback = (data['Stoch_K'] > 25) & (data['Stoch_K'] < 55)

        SETUP_3A = (
            TIME_OK &
            near_ema &
            (ema_up | di_bull) &
            stoch_pullback &
            stoch_cross &
            green &
            volume_strong
        )

        SETUP_3B = (
            TIME_OK &
            near_ema &
            stoch_pullback &
            (stoch_cross | stoch_turning) &
            (green | higher_close) &
            volume_ok &
            ~volume_strong
        )

        # ---------- SETUP 4 (PE) ----------
        # ---------- MARKET STRUCTURE ----------
        higher_low = data['Low'] > data['Low'].shift(1)
        hl_pattern = (
                higher_low &
                (data['Low'].shift(1) > data['Low'].shift(2))
        )

        # ---------- STOCHASTIC (MOMENTUM START, NOT LATE) ----------
        was_oversold = (data['Stoch_K'] < 25).rolling(6).max() == 1

        stoch_confirm = (
                (data['Stoch_K'] > data['Stoch_D']) &
                (data['Stoch_K'] > 25) &
                (data['Stoch_K'].shift(1) <= 25)
        )

        # ---------- TREND + ACCELERATION ----------
        adx_strong = data['ADX'] > 22
        adx_rising = data['ADX'] > data['ADX'].shift(1)

        strong_trend = adx_strong & adx_rising & di_bull

        # ---------- VOLATILITY EXPANSION (OPTION KEY) ----------
        atr_rising = data['ATR'] > data['ATR'].shift(1)

        # ---------- FINAL OPTION SETUP ----------
        SETUP_4 = (
                TIME_OK &
                hl_pattern &
                was_oversold &
                stoch_confirm &
                strong_trend &
                atr_rising &
                green &
                volume_ok
        )

        # ---------- PREVENT MULTIPLE ENTRIES ----------


        # ---------- SETUP 5 (PE) ----------
        vwap_stretch = data['Low'] < data['VWAP'] * 0.995  # ~0.5% below VWAP

        # 2️⃣ Reclaim VWAP (acceptance)
        vwap_reclaim = data['Close'] > data['VWAP']

        # 3️⃣ Strong bullish candle
        strong_reversal = (
                (data['Close'] > data['Open']) &
                ((data['Close']-data['Open']) > (data['High']-data['Low']) * 0.5)
        )

        # 4️⃣ Optional pullback confirmation
        vwap_pullback_hold = (
                (data['Low'] <= data['VWAP'] * 1.002) &
                (data['Close'] > data['VWAP'])
        )

        # -------- FINAL SETUP --------

        SETUP_5 = (
                TIME_OK &
                vwap_stretch &
                (vwap_reclaim | vwap_pullback_hold) &
                strong_reversal

        )

    # ============================================================
    # REST OF LOGIC (UNCHANGED)
    branch1 = stoch_buy_signal & short_term_bullish
    branch2 = momentum_positive & stoch_buy_signal
    branch3 = stoch_oversold_recovery
    branch4 = stoch_buy_signal & (data['RSI'] > 30)

    recent_extreme_oversold = data['EMA5'] > data['EMA9']
    precondition = recent_extreme_oversold

    # === TRENDLINE TOUCH DETECTION ===
    data['touches'] = 0

    # print("\n" + "="*80)
    # print("TRENDLINE DETECTION (Extended Backwards)")
    # print("="*80)

    # Detect touches with backward extension
    touch_indices = detect_trendline_touches_for_strategy(
        data,
        swing_period=8,  # 8 candles = 24 minutes
        min_hl_points=2,  # Minimum 2 Higher Lows
        lookback_candles=100,  # Analyze last 100 candles (5 hours)
        tolerance=0.007,  # 0.7% tolerance
        extend_back=True  # EXTEND BACKWARDS to find all touches
    )

    # Mark all touches
    for touch_idx in touch_indices:
        if touch_idx < len(data):
            data.iloc[touch_idx, data.columns.get_loc('touches')] = 1

    # === COMBINE SIGNALS ===
    original_signal = (
            (branch1 | branch2 | branch3 | branch4)
            & (data['touches'] == 1)
    )



    # ============================================================



    GRADE_A = (SETUP_1A | SETUP_2A | SETUP_3A | SETUP_5).astype(int)
    GRADE_B = (SETUP_1B | SETUP_2B | SETUP_3B | SETUP_4).astype(int) & ~GRADE_A.astype(bool)

    overbought = (data['Stoch_K'] > 82) | (data['RSI'] > 72)
    big_red = (data['Open'] - data['Close']) > data['ATR'] * 1.0
    recent_dist = big_red | big_red.shift(1)
    extended = (data['Close'] - data['EMA20']) / data['EMA20'] > 0.028
    small_candle = candle_range < data['ATR'] * 0.4


    REJECTION = overbought | recent_dist | extended | small_candle

    GRADE_A = GRADE_A & ~REJECTION
    GRADE_B = GRADE_B & ~REJECTION

    higher_lows_2bar = data['Low'] > data['Low'].shift(1)

    BEST_GRADE_B = (
        GRADE_B.astype(bool) &
        (
            higher_lows_2bar |
            (data['Volume_Ratio'] > 1.0) |
            rsi_turn
        )
    )

    raw_sig = (GRADE_A | BEST_GRADE_B | original_signal).astype(int)

    data['signal_grade'] = ""
    data.loc[GRADE_A.astype(bool), 'signal_grade'] = "A"
    data.loc[BEST_GRADE_B, 'signal_grade'] = "B"

    data["signal_reason"] = ""
    data.loc[(SETUP_1A | SETUP_1B) & raw_sig.astype(bool), "signal_reason"] += "Oversold | "
    data.loc[(SETUP_2A | SETUP_2B) & raw_sig.astype(bool), "signal_reason"] += "BB Bounce | "
    data.loc[(SETUP_3A | SETUP_3B) & raw_sig.astype(bool), "signal_reason"] += "EMA Pullback | "
    data.loc[SETUP_4 & raw_sig.astype(bool), "signal_reason"] += "Momentum | "
    data.loc[SETUP_5 & raw_sig.astype(bool), "signal_reason"] += "Volume Spike | "

    data["signal_reason"] = data["signal_reason"].str.rstrip(" | ")

    data.loc[data['signal_grade'] == "A", 'signal_reason'] += " [★★★]"
    data.loc[data['signal_grade'] == "B", 'signal_reason'] += " [★★]"

    cooldown = 5
    recent = raw_sig.shift(1).rolling(cooldown).sum().fillna(0)




    data['st_sig'] = ((raw_sig == 1) & (recent == 0)).astype(int)

    

    llm_confidence = None  # <-- ADD THIS LINE

    if raw_sig.iloc[-1] == 1:
        print("RAW SIGNAL TRIGGERED (LIVE)")
    
        ohlcv = fetch_ohlcv(symbol)
    
        if not ohlcv:
            print("Market data not available")
        else:
            llm_result = llm_trade_signal(symbol, ohlcv, "3m")
    
            print("========== LLM RESULT (LIVE) ==========")
            print(llm_result)
            print("======================================")
    
            llm_confidence = llm_result.get("confidence", 0)  # <-- CHANGE
            print("LLM confidence:", llm_confidence)
    
    
    # 🔧 ONLY APPLY CONFIDENCE TO LAST ROW
    data['st_sig'] = ((raw_sig == 1) & (recent == 0)).astype(int)
    
    if llm_confidence is not None:
        if llm_confidence <= 70:
            data.iloc[-1, data.columns.get_loc('st_sig')] = 0

    return data




def super_trend1we(symbol, data):

    # ==========================
    # CORE INDICATORS (UNCHANGED)
    # ==========================
    data['EMA9'] = ta.ema(data['Close'], 9)
    data['EMA20'] = ta.ema(data['Close'], 20)
    data['EMA50'] = ta.ema(data['Close'], 50)

    data['RSI'] = ta.rsi(data['Close'], 14)

    stoch = ta.stoch(data['High'], data['Low'], data['Close'], 14, 3, 3)
    data['Stoch_K'] = stoch['STOCHk_14_3_3']
    data['Stoch_D'] = stoch['STOCHd_14_3_3']

    adx_df = ta.adx(data['High'], data['Low'], data['Close'], 14)
    data['ADX'] = adx_df['ADX_14']
    data['Plus_DI'] = adx_df['DMP_14']
    data['Minus_DI'] = adx_df['DMN_14']

    bb = ta.bbands(data['Close'], 20, 2)
    print(bb.columns)
    data['BB_lower'] = bb['BBL_20_2.0_2.0']
    data['BB_upper'] = bb['BBU_20_2.0_2.0']
    data['BB_mid'] = bb['BBM_20_2.0_2.0']

    data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], 14)

    data['Volume_MA'] = data['Volume'].rolling(20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
    data['touches'] = 0

    # ==========================
    # TIME FILTER (UNCHANGED)
    # ==========================
    if isinstance(data.index, pd.DatetimeIndex):
        time_series = data.index.time
    else:
        time_series = pd.to_datetime(data.index).time

    data['time_mins'] = [t.hour * 60 + t.minute for t in time_series]
    TIME_OK = (data['time_mins'] >= 570) & (data['time_mins'] <= 885)

    # =============================================================
    # SELECT CE OR PE LOGIC BLOCK
    # =============================================================
    opt_type = data["Option_Type"].iloc[0]

    if opt_type == "CE":

        # ============================
        # CE SETUPS START HERE
        # ============================

        # ---------- SETUP 1 ----------
        stoch_oversold = data['Stoch_K'] < 25
        stoch_turning = data['Stoch_K'] > data['Stoch_K'].shift(1)
        stoch_cross = data['Stoch_K'] > data['Stoch_D']

        rsi_ok = (data['RSI'] > 30) & (data['RSI'] < 52)
        rsi_turn = data['RSI'] > data['RSI'].shift(1)

        green = data['Close'] > data['Open']
        higher_close = data['Close'] > data['Close'].shift(1)

        volume_ok = data['Volume_Ratio'] > 0.75
        volume_strong = data['Volume_Ratio'] > 1.2

        SETUP_1A = (
            TIME_OK &
            stoch_oversold &
            (stoch_turning & stoch_cross) &
            rsi_ok &
            rsi_turn &
            (green | higher_close) &
            volume_strong
        )

        SETUP_1B = (
            TIME_OK &
            stoch_oversold &
            (stoch_turning | stoch_cross) &
            rsi_ok &
            (green | higher_close) &
            volume_ok &
            ~volume_strong
        )

        # ----------------- CE SETUP 2 -----------------
        bb_touch = (data['Low'] <= data['BB_lower'] * 1.005) & (data['Close'] > data['BB_lower'])
        body = data['Close'] - data['Open']
        candle_range = data['High'] - data['Low']
        good_candle = (body > 0) & (body > candle_range * 0.3)

        SETUP_2A = (
            TIME_OK &
            bb_touch &
            good_candle &
            (data['Stoch_K'] < 25) &
            stoch_turning &
            volume_strong
        )

        SETUP_2B = (
            TIME_OK &
            bb_touch &
            (green | good_candle) &
            (data['Stoch_K'] < 20) &
            (stoch_turning | stoch_cross) &
            volume_ok &
            ~volume_strong
        )

        # ----------------- CE SETUP 3 -----------------
        near_ema = (data['Low'] <= data['EMA20'] * 1.005) & (data['Close'] > data['EMA20'])
        ema_up = data['EMA9'] > data['EMA20']
        di_bull = data['Plus_DI'] > data['Minus_DI']
        stoch_pullback = (data['Stoch_K'] > 25) & (data['Stoch_K'] < 55)

        SETUP_3A = (
            TIME_OK &
            near_ema &
            (ema_up | di_bull) &
            stoch_pullback &
            stoch_cross &
            green &
            volume_strong
        )

        SETUP_3B = (
            TIME_OK &
            near_ema &
            stoch_pullback &
            (stoch_cross | stoch_turning) &
            (green | higher_close) &
            volume_ok &
            ~volume_strong
        )

        # ----------------- CE SETUP 4 -----------------
        higher_low = data['Low'] > data['Low'].shift(1)
        hl_pattern = higher_low & (data['Low'].shift(1) > data['Low'].shift(2))
        stoch_healthy = (data['Stoch_K'] > 30) & (data['Stoch_K'] < 70)
        strong_trend = (data['ADX'] > 22) & di_bull

        SETUP_4 = (
            TIME_OK &
            hl_pattern &
            stoch_healthy &
            stoch_cross &
            strong_trend &
            green &
            volume_ok
        )

        # ----------------- CE SETUP 5 -----------------
        volume_climax = data['Volume_Ratio'] > 1.8
        stoch_extreme = data['Stoch_K'] < 25
        rsi_extreme = data['RSI'] < 38
        strong_reversal = (body > 0) & (body > candle_range * 0.5)

        SETUP_5 = (
            TIME_OK &
            volume_climax &
            (stoch_extreme | rsi_extreme) &
            (strong_reversal | (green & stoch_turning))
        )

    else:
        # =============================================================
        # PE SETUPS START HERE
        # =============================================================

        # ---------- SETUP 1 (PE) ----------
        stoch_oversold = data['Stoch_K'] < 25
        stoch_turning = data['Stoch_K'] > data['Stoch_K'].shift(1)
        stoch_cross = data['Stoch_K'] > data['Stoch_D']

        rsi_ok = (data['RSI'] > 30) & (data['RSI'] < 52)
        rsi_turn = data['RSI'] > data['RSI'].shift(1)

        green = data['Close'] > data['Open']
        higher_close = data['Close'] > data['Close'].shift(1)

        volume_ok = data['Volume_Ratio'] > 0.75
        volume_strong = data['Volume_Ratio'] > 1.2

        SETUP_1A = (
            TIME_OK &
            stoch_oversold &
            (stoch_turning & stoch_cross) &
            rsi_ok &
            rsi_turn &
            (green | higher_close) &
            volume_strong
        )

        SETUP_1B = (
            TIME_OK &
            stoch_oversold &
            (stoch_turning | stoch_cross) &
            rsi_ok &
            (green | higher_close) &
            volume_ok &
            ~volume_strong
        )

        # ---------- SETUP 2 (PE) ----------
        bb_touch = (data['Low'] <= data['BB_lower'] * 1.005) & (data['Close'] > data['BB_lower'])
        body = data['Close'] - data['Open']
        candle_range = data['High'] - data['Low']
        good_candle = (body > 0) & (body > candle_range * 0.3)

        SETUP_2A = (
            TIME_OK &
            bb_touch &
            good_candle &
            (data['Stoch_K'] < 25) &
            stoch_turning &
            volume_strong
        )

        SETUP_2B = (
            TIME_OK &
            bb_touch &
            (green | good_candle) &
            (data['Stoch_K'] < 20) &
            (stoch_turning | stoch_cross) &
            volume_ok &
            ~volume_strong
        )

        # ---------- SETUP 3 (PE) ----------
        near_ema = (data['Low'] <= data['EMA20'] * 1.005) & (data['Close'] > data['EMA20'])
        ema_up = data['EMA9'] > data['EMA20']
        di_bull = data['Plus_DI'] > data['Minus_DI']
        stoch_pullback = (data['Stoch_K'] > 25) & (data['Stoch_K'] < 55)

        SETUP_3A = (
            TIME_OK &
            near_ema &
            (ema_up | di_bull) &
            stoch_pullback &
            stoch_cross &
            green &
            volume_strong
        )

        SETUP_3B = (
            TIME_OK &
            near_ema &
            stoch_pullback &
            (stoch_cross | stoch_turning) &
            (green | higher_close) &
            volume_ok &
            ~volume_strong
        )

        # ---------- SETUP 4 (PE) ----------
        higher_low = data['Low'] > data['Low'].shift(1)
        hl_pattern = higher_low & (data['Low'].shift(1) > data['Low'].shift(2))
        stoch_healthy = (data['Stoch_K'] > 30) & (data['Stoch_K'] < 70)
        strong_trend = (data['ADX'] > 22) & di_bull

        SETUP_4 = (
            TIME_OK &
            hl_pattern &
            stoch_healthy &
            stoch_cross &
            strong_trend &
            green &
            volume_ok
        )

        # ---------- SETUP 5 (PE) ----------
        volume_climax = data['Volume_Ratio'] > 1.8
        stoch_extreme = data['Stoch_K'] < 25
        rsi_extreme = data['RSI'] < 38
        strong_reversal = (body > 0) & (body > candle_range * 0.5)

        SETUP_5 = (
            TIME_OK &
            volume_climax &
            (stoch_extreme | rsi_extreme) &
            (strong_reversal | (green & stoch_turning))
        )

    # ============================================================
    # REST OF LOGIC (UNCHANGED)
    # ============================================================

    GRADE_A = (SETUP_1A | SETUP_2A | SETUP_3A | SETUP_5).astype(int)
    GRADE_B = (SETUP_1B | SETUP_2B | SETUP_3B | SETUP_4).astype(int) & ~GRADE_A.astype(bool)

    overbought = (data['Stoch_K'] > 82) | (data['RSI'] > 72)
    big_red = (data['Open'] - data['Close']) > data['ATR'] * 1.0
    recent_dist = big_red | big_red.shift(1)
    extended = (data['Close'] - data['EMA20']) / data['EMA20'] > 0.028
    small_candle = candle_range < data['ATR'] * 0.4

    REJECTION = overbought | recent_dist | extended | small_candle

    GRADE_A = GRADE_A & ~REJECTION
    GRADE_B = GRADE_B & ~REJECTION

    higher_lows_2bar = data['Low'] > data['Low'].shift(1)

    BEST_GRADE_B = (
        GRADE_B.astype(bool) &
        (
            higher_lows_2bar |
            (data['Volume_Ratio'] > 1.0) |
            rsi_turn
        )
    )

    raw_sig = (GRADE_A | BEST_GRADE_B).astype(int)

    data['signal_grade'] = ""
    data.loc[GRADE_A.astype(bool), 'signal_grade'] = "A"
    data.loc[BEST_GRADE_B, 'signal_grade'] = "B"

    data["signal_reason"] = ""
    data.loc[(SETUP_1A | SETUP_1B) & raw_sig.astype(bool), "signal_reason"] += "Oversold | "
    data.loc[(SETUP_2A | SETUP_2B) & raw_sig.astype(bool), "signal_reason"] += "BB Bounce | "
    data.loc[(SETUP_3A | SETUP_3B) & raw_sig.astype(bool), "signal_reason"] += "EMA Pullback | "
    data.loc[SETUP_4 & raw_sig.astype(bool), "signal_reason"] += "Momentum | "
    data.loc[SETUP_5 & raw_sig.astype(bool), "signal_reason"] += "Volume Spike | "

    data["signal_reason"] = data["signal_reason"].str.rstrip(" | ")

    data.loc[data['signal_grade'] == "A", 'signal_reason'] += " [★★★]"
    data.loc[data['signal_grade'] == "B", 'signal_reason'] += " [★★]"

    cooldown = 5
    recent = raw_sig.shift(1).rolling(cooldown).sum().fillna(0)
    data['st_sig'] = ((raw_sig == 1) & (recent == 0)).astype(int)

    return data


# ======================================================================
# 3. SUPER TREND STRATEGY
# ======================================================================


def super_trendttest(symbol, data):
    import pandas_ta as ta
    import numpy as np
    import pandas as pd

    # === SYMBOL PARSING ===
    parts = symbol.split()
    ticker = parts[0]
    expiry = f"{parts[1]} {parts[2]} {parts[3]}"
    opttype = parts[4]
    strike = float(parts[5])

    # === CORE INDICATORS ===
    data['EMA5'] = ta.ema(data['Close'], length=5)
    data['EMA9'] = ta.ema(data['Close'], length=9)
    data['EMA20'] =pd.to_numeric(ta.ema(data['Close'], length=20)) 
    data['EMA50'] =pd.to_numeric(ta.ema(data['Close'], length=50)) 

    # === MOMENTUM INDICATORS ===
    data['ADX'] = ta.adx(data['High'], data['Low'], data['Close'], length=14)['ADX_14']
    data['RSI'] = ta.rsi(data['Close'], length=14)
    data['VO'] = volume_oscillator(data, fast=10, slow=20)

    # === BOLLINGER BANDS ===
    bb = ta.bbands(data['Close'], length=20, std=2)
    data['BB_upper'] = bb['BBU_20_2.0_2.0']
    data['BB_lower'] = bb['BBL_20_2.0_2.0']
    data['BB_middle'] = bb['BBM_20_2.0_2.0']
    data['BB_width'] = (data['BB_upper']-data['BB_lower']) / data['BB_middle'] * 100

    # === STOCHASTIC ===
    stoch = ta.stoch(data['High'], data['Low'], data['Close'], k=10, d=3, smooth_k=3)
    data['Stoch_K'] = stoch['STOCHk_10_3_3']
    data['Stoch_D'] = stoch['STOCHd_10_3_3']

    # === STOCHASTIC PATTERNS ===

    stoch_deep_oversold_bounce = (
            (data['Stoch_K'].shift(1) < 25) &
            (data['Stoch_D'].shift(1) < 25) &
            (data['Stoch_K']-data['Stoch_K'].shift(1) > 3) &
            (
                    ((data['Stoch_K'] > data['Stoch_D']) &
                     (data['Stoch_K'].shift(1) <= data['Stoch_D'].shift(1))) |
                    ((data['Stoch_K'] > data['Stoch_K'].shift(1)+3) &
                     (data['Stoch_K'] > 15))
            )
    )

    stoch_oversold_recovery = (
            (data['Stoch_K'].shift(1) < 40) &
            (data['Stoch_D'].shift(1) < 40) &
            (data['Stoch_K'] > 20) &
            (data['Stoch_D'] > 20) &
            (data['Stoch_K'] > data['Stoch_D']) &
            (data['Stoch_K'].shift(1) <= data['Stoch_D'].shift(1)) &
            ((data['Stoch_K']-data['Stoch_K'].shift(1)) > 5)
    )

    stoch_early_bounce = (
            (data['Stoch_K'] < 20) &
            (data['Stoch_K'].shift(1) >= 8) &
            (data['Stoch_D'] < 25) &
            (data['RSI'] > data['RSI'].shift(1)) &
            (data['Close'] > data['Open']) &
            ((data['Stoch_K']-data['Stoch_K'].shift(1)) > 3) &
            (data['Stoch_K'] > data['Stoch_D'])
    )

    stoch_buy_signal = (
            stoch_deep_oversold_bounce |
            stoch_oversold_recovery |
            stoch_early_bounce
    )

    # === VOLUME ANALYSIS ===
    data['Volume_MA'] = data['Volume'].rolling(20).mean()
    volume_surge = data['Volume'] > (1.5 * data['Volume_MA'])
    volume_positive = data['Volume'] > data['Volume_MA']

    # === TREND FILTERS ===
    uptrend = (
            (data['EMA5'] > data['EMA20']) &
            (data['EMA20'] > data['EMA50']) &
            (data['ADX'] > 25)
    )

    short_term_bullish = data['Close'] > data['EMA5']

    momentum_positive = (
            (data['RSI'] > 45) &
            (data['RSI'] < 70)
    )

    # === STRATEGY BRANCHES ===
    branch1 = stoch_buy_signal & short_term_bullish
    branch2 = momentum_positive & stoch_buy_signal
    branch3 = stoch_oversold_recovery
    branch4 = stoch_buy_signal & (data['RSI'] > 30)

    recent_extreme_oversold = data['EMA5'] > data['EMA9']
    precondition = recent_extreme_oversold

    # === TRENDLINE TOUCH DETECTION ===
    data['touches'] = 0

    # print("\n" + "="*80)
    # print("TRENDLINE DETECTION (Extended Backwards)")
    # print("="*80)

    # Detect touches with backward extension
    touch_indices = detect_trendline_touches_for_strategy(
        data,
        swing_period=8,  # 8 candles = 24 minutes
        min_hl_points=2,  # Minimum 2 Higher Lows
        lookback_candles=100,  # Analyze last 100 candles (5 hours)
        tolerance=0.007,  # 0.7% tolerance
        extend_back=True  # EXTEND BACKWARDS to find all touches
    )

    # Mark all touches
    for touch_idx in touch_indices:
        if touch_idx < len(data):
            data.iloc[touch_idx, data.columns.get_loc('touches')] = 1

    # === COMBINE SIGNALS ===
    original_signal = branch1 | branch2 | branch3 | branch4

    data['st_sig'] = np.where(
        original_signal & (data['touches'] == 1),
        1,
        0
    )

    # === SIGNAL DETAILS ===
    data['stoch_pattern'] = np.select(
        [stoch_deep_oversold_bounce, stoch_oversold_recovery, stoch_early_bounce],
        ['Deep_Oversold_Bounce', 'Recovery_Momentum', 'Early_Bounce'],
        default='None'
    )

    data['signal_reason'] = np.select(
        [branch1, branch2, branch3, branch4],
        [
            f'Oversold_Bounce_{data["stoch_pattern"]}',
            'Momentum_Continuation',
            'Breakout_After_Oversold',
            'Support_Bounce'
        ],
        default=''
    )

    data['signal_reason'] = np.where(
        data['st_sig'] == 1,
        data['signal_reason']+'_TRENDLINE_TOUCH',
        data['signal_reason']
    )

    data['bounce_strength'] = np.where(
        data['st_sig'] == 1,
        np.select(
            [
                data['Stoch_K'].shift(1) < 10,
                data['Stoch_K'].shift(1) < 15,
                data['Stoch_K'].shift(1) < 20,
            ],
            ['Strong', 'Medium', 'Normal'],
            default='Weak'
        ),
        ''
    )

    # Print detailed summary (keep only this if you want)
    # print("\n" + "="*80)
    # print("SUMMARY")
    # print("="*80)
    # print(f"Total data points: {len(data)}")
    # print(f"Total touches detected: {data['touches'].sum()}")
    # print(f"Strategy signals: {original_signal.sum()}")
    # print(f"Confirmed signals (strategy + touch): {data['st_sig'].sum()}")

    # Show where touches occurred
    # if data['touches'].sum() > 0:
    #     touch_dates = data[data['touches'] == 1].index.tolist()
    #     print(f"\nTouch dates:")
    #     for dt in touch_dates[:10]:  # Show first 10
    #         print(f"  {dt}")
    #     if len(touch_dates) > 10:
    #         print(f"  ... and {len(touch_dates) - 10} more")

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
        res = requests.post("https://algotrading-1-dluo.onrender.com/update-ticker", json={
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
                data_fut = get_cash_market_data(i, '3m')
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
                if data_list[i]['st_sig'][-1] == 1:


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
                    if data_list[i]['st_sig'][-1] == -1:
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
                if data_list[i]['st_sig'][-1] == -1:

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
                    if data_list[i]['st_sig'][-1] == 1:
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
