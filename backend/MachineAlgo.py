from py5paisa import FivePaisaClient
from py5paisa.order import Order, OrderType, Exchange
import pyotp
import os
import mibian as mb
working_dir = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
import datetime as dt
import auth
import pandas_ta as indi
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
import pandas_ta as ta

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
                                             From=dt.date.today()-dt.timedelta(4), To=dt.date.today()))

    df.set_index("Datetime", inplace=True)
    df["Option_Type"] = opttype
    df["Strike_Price"] = strike



    print(df)
    return df




def super_trend(data):
    import pandas_ta as ta
    import numpy as np
    import pandas as pd
    data['st_sig']=0
    data = data.copy()
    data.index = pd.to_datetime(data.index)

    # === Indicators ===
    data['EMA'] = ta.ema(data['Close'], length=5)
    data['RSI'] = ta.rsi(data['Close'], length=14)
    bb = ta.bbands(data['Close'], length=20, std=1)
    data['BB_upper'] = bb['BBU_20_1.0']
    data['BB_lower'] = bb['BBL_20_1.0']
    data['BB_width'] = data['BB_upper'] - data['BB_lower']

    # === Candle Anatomy ===
    data['body'] = data['Close'] - data['Open']
    data['range'] = data['High'] - data['Low']
    data['upper_wick'] = data['High'] - data[['Close', 'Open']].max(axis=1)
    data['lower_wick'] = data[['Close', 'Open']].min(axis=1) - data['Low']

    # === Strong Bullish Candle Logic ===
    strong_bullish_candle_logic = (
        (data['body'] > 0) &
        (data['body'] > 0.6 * data['range']) &
        (data['upper_wick'] < 0.3 * data['body']) &
        (data['lower_wick'] < 0.3 * data['body'])
    )

    # === Setup Conditions ===
    cond_bearish_candle = data['Close'].shift(1) < data['Open'].shift(1)
    cond_bullish_candle = data['Close'] > data['Open']
    cond_below_ema = (data['Close'].shift(1) < data['EMA'].shift(1)) & (data['Close'] < data['EMA'])
    cond_bearish_ema_below = (data['Close'].shift(1) < data['EMA'].shift(1)) & (data['Open'].shift(1) < data['EMA'].shift(1))
    cond_buy = (data['Close'] > data['Close'].shift(1)) & (data['Close'] > data['EMA'])
    cond_distance_from_ema = (data['EMA'] - data['Close']) > 1.5

    setup_raw = (
        (cond_bearish_candle & cond_bullish_candle & cond_below_ema & cond_distance_from_ema)
        | (cond_bearish_candle & cond_bullish_candle & cond_bearish_ema_below & cond_buy)
    )

    # === Time Filter ===
    times = data.index.time
    time_filter = ~(
        ((times >= pd.to_datetime("09:15").time()) & (times <= pd.to_datetime("09:25").time())) |
        ((times >= pd.to_datetime("15:15").time()) & (times <= pd.to_datetime("15:30").time()))
    )

    # === Setup Found (only one signal per window)
    setup_found = [0] * len(data)
    active_trade = False
    last_trade_index = -10

    for i in range(len(data)):
        if setup_raw.iloc[i] and not active_trade:
            setup_found[i] = 1
            last_trade_index = i
            active_trade = True
        if i - last_trade_index > 2:
            active_trade = False

    setup_found_series = pd.Series(setup_found, index=data.index)

    # === RSI Rising
    rsi_rising = data['RSI'] > data['RSI'].shift(1)

    # === Branch Conditions
    cond_not_touching_bb_upper = data['High'] < data['BB_upper']

    branch1 = (setup_found_series == 1) & strong_bullish_candle_logic & cond_not_touching_bb_upper & rsi_rising
    branch2 = strong_bullish_candle_logic & (data['RSI'] > 50) & (data['RSI'] < 65) & rsi_rising

    # === Final Signal and Reason
    signal_raw = np.where(branch1, 1, np.where(branch2, 1, 0))
    signal_final = np.where(time_filter, signal_raw, 0)

    reason = np.where(branch1, 'Branch1_StrongBullish_RSIUp_NoBBTouch',
              np.where(branch2, 'Branch2_StrongBullish_RSI>50_RSIUp', ''))
    reason = np.where(time_filter, reason, '')

    # === Output Columns
    data['st_sig'] = signal_final
    data['signal_reason'] = reason
    data['setup_found'] = setup_found_series

    data.to_excel("BuyOnlyTradeResults_Bollinger.xlsx", index=True)
    return data[['st_sig', 'signal_reason', 'setup_found', 'BB_width']]










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
    valid_dfs = []

    for file in files:
        if os.path.exists(file):
            df = pd.read_excel(file)
            if not df.empty:
                valid_dfs.append(df)
        else:
            print(f"File not found: {file}")

    if valid_dfs:
        merged_df = pd.concat(valid_dfs, ignore_index=True)
        merged_df.to_excel('All_Trades.xlsx', index=False)
        return 'All_Trades.xlsx saved successfully.'
    else:
        return 'No valid trade files found to merge.'


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
required_times = [
    (9, 20), (9, 21), (9, 22), (9, 23), (9, 24), (9, 25), (9, 26), (9, 27), (9, 28), (9, 29),
    (9, 30), (9, 31), (9, 32), (9, 33), (9, 34), (9, 35), (9, 36), (9, 37), (9, 38), (9, 39),
    (9, 40), (9, 41), (9, 42), (9, 43), (9, 44), (9, 45), (9, 46), (9, 47), (9, 48), (9, 49),
    (9, 50), (9, 51), (9, 52), (9, 53), (9, 54), (9, 55), (9, 56), (9, 57), (9, 58), (9, 59),
    (10, 0), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9),
    (10, 10), (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19),
    (10, 20), (10, 21), (10, 22), (10, 23), (10, 24), (10, 25), (10, 26), (10, 27), (10, 28), (10, 29),
    (10, 30), (10, 31), (10, 32), (10, 33), (10, 34), (10, 35), (10, 36), (10, 37), (10, 38), (10, 39),
    (10, 40), (10, 41), (10, 42), (10, 43), (10, 44), (10, 45), (10, 46), (10, 47), (10, 48), (10, 49),
    (10, 50), (10, 51), (10, 52), (10, 53), (10, 54), (10, 55), (10, 56), (10, 57), (10, 58), (10, 59),
    (11, 0), (11, 1), (11, 2), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (11, 8), (11, 9),
    (11, 10), (11, 11), (11, 12), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), (11, 18), (11, 19),
    (11, 20), (11, 21), (11, 22), (11, 23), (11, 24), (11, 25), (11, 26), (11, 27), (11, 28), (11, 29),
    (11, 30), (11, 31), (11, 32), (11, 33), (11, 34), (11, 35), (11, 36), (11, 37), (11, 38), (11, 39),
    (11, 40), (11, 41), (11, 42), (11, 43), (11, 44), (11, 45), (11, 46), (11, 47), (11, 48), (11, 49),
    (11, 50), (11, 51), (11, 52), (11, 53), (11, 54), (11, 55), (11, 56), (11, 57), (11, 58), (11, 59),
    (12, 0), (12, 1), (12, 2), (12, 3), (12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9),
    (12, 10), (12, 11), (12, 12), (12, 13), (12, 14), (12, 15), (12, 16), (12, 17), (12, 18), (12, 19),
    (12, 20), (12, 21), (12, 22), (12, 23), (12, 24), (12, 25), (12, 26), (12, 27), (12, 28), (12, 29),
    (12, 30), (12, 31), (12, 32), (12, 33), (12, 34), (12, 35), (12, 36), (12, 37), (12, 38), (12, 39),
    (12, 40), (12, 41), (12, 42), (12, 43), (12, 44), (12, 45), (12, 46), (12, 47), (12, 48), (12, 49),
    (12, 50), (12, 51), (12, 52), (12, 53), (12, 54), (12, 55), (12, 56), (12, 57), (12, 58), (12, 59),
    (13, 0), (13, 1), (13, 2), (13, 3), (13, 4), (13, 5), (13, 6), (13, 7), (13, 8), (13, 9),
    (13, 10), (13, 11), (13, 12), (13, 13), (13, 14), (13, 15), (13, 16), (13, 17), (13, 18), (13, 19),
    (13, 20), (13, 21), (13, 22), (13, 23), (13, 24), (13, 25), (13, 26), (13, 27), (13, 28), (13, 29),
    (13, 30), (13, 31), (13, 32), (13, 33), (13, 34), (13, 35), (13, 36), (13, 37), (13, 38), (13, 39),
    (13, 40), (13, 41), (13, 42), (13, 43), (13, 44), (13, 45), (13, 46), (13, 47), (13, 48), (13, 49),
    (13, 50), (13, 51), (13, 52), (13, 53), (13, 54), (13, 55), (13, 56), (13, 57), (13, 58), (13, 59),
    (14, 0), (14, 1), (14, 2), (14, 3), (14, 4), (14, 5), (14, 6), (14, 7), (14, 8), (14, 9),
    (14, 10), (14, 11), (14, 12), (14, 13), (14, 14), (14, 15), (14, 16), (14, 17), (14, 18), (14, 19),
    (14, 20), (14, 21), (14, 22), (14, 23), (14, 24), (14, 25), (14, 26), (14, 27), (14, 28), (14, 29),
    (14, 30), (14, 31), (14, 32), (14, 33), (14, 34), (14, 35), (14, 36), (14, 37), (14, 38), (14, 39),
    (14, 40), (14, 41), (14, 42), (14, 43), (14, 44), (14, 45), (14, 46), (14, 47), (14, 48), (14, 49),
    (14, 50), (14, 51), (14, 52), (14, 53), (14, 54), (14, 55), (14, 56), (14, 57), (14, 58), (14, 59),
    (15, 0), (15, 1), (15, 2), (15, 3), (15, 4), (15, 5), (15, 6), (15, 7), (15, 8), (15, 9),
    (15, 10), (15, 11), (15, 12), (15, 13), (15, 14), (15, 15), (15, 16), (15, 17), (15, 18), (15, 19),
    (15, 20), (15, 21), (15, 22), (15, 23), (15, 24), (15, 25), (15, 26), (15, 27), (15, 28), (15, 29),
    (15, 30)
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

    data_fut = get_cash_market_data(h, '5m')
    data_fut.drop(data_fut.tail(1).index, inplace=True)

    super_trend(data_fut)

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
            print("###################################################################")
            send_to_ui(i, spot_prices1[i])
            time.sleep(0.5)


            if is_required_time():
                data_fut = get_cash_market_data(i, '5m')
                data_fut.drop(data_fut.tail(1).index, inplace=True)
                super_trend(data_fut)

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

                        entry_time = dt.datetime.now().strftime("%d-%b-%Y %I:%M%p")

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

                        Exit_Time = dt.datetime.now().strftime("%d-%b-%Y %I:%M%p")

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
                        Exit_Time = dt.datetime.now().strftime("%d-%b-%Y %I:%M%p")
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
                        Exit_Time = dt.datetime.now().strftime("%d-%b-%Y %I:%M%p")
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
                        Exit_Time = dt.datetime.now().strftime("%d-%b-%Y %I:%M%p")
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

                        entry_time = dt.datetime.now().strftime("%d-%b-%Y %I:%M%p")

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

















