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

    if len(df) > lookback_candles:
        start_idx = len(df) - lookback_candles
        df_recent = df.iloc[start_idx:].copy()
        offset = start_idx
    else:
        df_recent = df.copy()
        offset = 0

    # Find swing lows
    swing_lows = []
    for i in range(swing_period, len(df_recent)):
        if i + swing_period > len(df_recent):
            current_low = df_recent['Low'].iloc[i]
            left_lows = df_recent['Low'].iloc[max(0, i - swing_period):i]

            if len(left_lows) > 0 and current_low <= left_lows.min():
                swing_lows.append((i + offset, current_low))
        else:
            current_low = df_recent['Low'].iloc[i]
            left_lows = df_recent['Low'].iloc[i - swing_period:i]
            right_lows = df_recent['Low'].iloc[i + 1:min(len(df_recent), i + swing_period + 1)]

            if current_low <= left_lows.min() and current_low <= right_lows.min():
                swing_lows.append((i + offset, current_low))

    if len(swing_lows) < 2:
        return []

    # Higher Lows
    higher_lows = [swing_lows[0]]
    for i in range(1, len(swing_lows)):
        idx, low = swing_lows[i]
        prev_idx, prev_low = higher_lows[-1]
        if low >= prev_low * 0.998:
            higher_lows.append((idx, low))

    if len(higher_lows) < min_hl_points:
        return []

    # Trendline
    recent_hl = higher_lows[-min_hl_points:]
    idx1, price1 = recent_hl[0]
    idx2, price2 = recent_hl[-1]

    slope = (price2 - price1) / (idx2 - idx1)
    if slope < -0.0001:
        return []

    intercept = price1 - slope * idx1

    if extend_back:
        search_start = max(0, len(df) - lookback_candles)
    else:
        search_start = idx1

    touch_indices = []
    for i in range(search_start, len(df)):
        line_value = slope * i + intercept
        threshold = line_value * tolerance

        current_low = df['Low'].iloc[i]
        current_high = df['High'].iloc[i]
        current_close = df['Close'].iloc[i]

        is_anchor_point = any(i == hl_idx for hl_idx, _ in recent_hl)

        if i > 0 and not is_anchor_point:
            prev_close = df['Close'].iloc[i - 1]

            touch_from_above = (prev_close > line_value and current_low <= line_value + threshold)
            crosses_line = (current_low <= line_value + threshold and current_high >= line_value - threshold)
            close_near = abs(current_close - line_value) <= threshold
            low_touch = abs(current_low - line_value) <= threshold

            if touch_from_above or crosses_line or close_near or low_touch:
                if current_close > line_value - (threshold * 2):
                    touch_indices.append(i)

    return touch_indices


# ======================================================================
# 3. SUPER TREND STRATEGY
# ======================================================================
def super_trend(symbol, data):

    # ---------------- Core Indicators ----------------
    data['EMA5'] = ta.ema(data['Close'], 5)
    data['EMA9'] = ta.ema(data['Close'], 9)
    data['EMA15'] = ta.ema(data['Close'], 15)
    data['EMA20'] = ta.ema(data['Close'], 20)
    data['EMA50'] = ta.ema(data['Close'], 50)

    data['ADX'] = ta.adx(data['High'], data['Low'], data['Close'], 14)['ADX_14']
    data['RSI'] = ta.rsi(data['Close'], 14)
    data['VO'] = volume_oscillator(data, 10, 20)

    bb = ta.bbands(data['Close'], 20, 2)
    print(bb.columns)
    data['BB_upper'] = bb['BBU_20_2_2.0']
    data['BB_lower'] = bb['BBL_20_2_2.0']
    data['BB_middle'] = bb['BBM_20_2_2.0']
    data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle'] * 100

    stoch = ta.stoch(data['High'], data['Low'], data['Close'], k=10, d=3, smooth_k=3)
    data['Stoch_K'] = stoch['STOCHk_10_3_3']
    data['Stoch_D'] = stoch['STOCHd_10_3_3']

    # ---------------- ATR for Dynamic Tolerance ----------------
    data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], 14)
    data['ATR_pct'] = data['ATR'] / data['Close'] * 100
    dyn_tol = max(0.007, (data['ATR_pct'].tail(100).median() / 100.0) * 0.7)

    # ---------------- Trendline Touches ----------------
    data['touches'] = 0
    touches = detect_trendline_touches_for_strategy(
        data,
        swing_period=8,
        min_hl_points=2,
        lookback_candles=100,
        tolerance=dyn_tol
    )
    for i in touches:
        data.loc[data.index[i], 'touches'] = 1

    # ---------------- Regime Filters ----------------
    data['is_trending'] = data['ADX'] > 22
    data['is_ranging'] = data['ADX'] < 18

    data['Volume_MA'] = data['Volume'].rolling(20).mean()
    data['vol_ok_strong'] = (data['Volume'] > data['Volume_MA'] * 1.2) & (data['VO'] > 0)
    data['vol_ok_normal'] = (data['Volume'] > data['Volume_MA'] * 0.7)

    # ---------------- Stochastic Patterns ----------------
    stoch_deep = (
        (data['Stoch_K'].shift(1) < 25) &
        (data['Stoch_D'].shift(1) < 25) &
        ((data['Stoch_K'] - data['Stoch_K'].shift(1)) > 3)
    )
    stoch_recovery = (
        (data['Stoch_K'].shift(1) < 40) &
        (data['Stoch_K'] > 20) &
        (data['Stoch_K'] > data['Stoch_D'])
    )
    stoch_bounce = (
        (data['Stoch_K'] < 20) &
        ((data['Stoch_K'] - data['Stoch_K'].shift(1)) > 3)
    )

    stoch_signal = stoch_deep | stoch_recovery | stoch_bounce

    # ---------------- Branch 5: EMA + SAR ----------------
    data['SAR'] = ta.psar(data['High'], data['Low'], data['Close'])['PSARl_0.02_0.2']

    b5 = (
        (data['EMA9'] > data['EMA15']) &
        (data['Close'] > data['EMA9']) &
        (data['Close'] > data['SAR']) &
        stoch_signal &
        data['is_trending'] &
        data['vol_ok_strong']
    )

    data['ema_sar_branch'] = b5.astype(int)

    # ---------------- Branch 6: BB Reversal ----------------
    bb_cross_up = (
        (data['Close'].shift(1) < data['BB_lower'].shift(1)) &
        (data['Close'] > data['BB_lower'])
    )

    b6 = (
        bb_cross_up &
        (data['Stoch_K'] < 20) &
        data['is_ranging'] &
        (data['RSI'] > data['RSI'].shift(1)) &
        data['vol_ok_normal']
    )
    data['bb_reversal_branch'] = b6.astype(int)

    # ---------------- Branch 1–4 Need Trendline Touch ----------------
    b14 = stoch_signal & (data['touches'] == 1) & data['vol_ok_normal']

    # ---------------- Combine All Signals ----------------
    raw_sig = (b14 | b5 | b6).astype(int)

    # --------------- Cooldown ----------------
    cooldown = 4
    recent = pd.Series(raw_sig).shift(1).rolling(cooldown).sum().fillna(0)
    data['st_sig'] = np.where((raw_sig == 1) & (recent == 0), 1, 0)

    return data


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
