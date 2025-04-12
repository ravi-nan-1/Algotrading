from py5paisa import FivePaisaClient
from py5paisa.order import Order, OrderType, Exchange
import pyotp
import os
import mibian as mb
working_dir = os.chdir(r"C:\Users\nanda\PycharmProjects\AlgoTrading")
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
from globals import spot_prices

UTC = pytz.timezone('Asia/Kolkata')
import time

client = FivePaisaClient(cred=auth.cred)
print(pyotp.TOTP(auth.token).now())

# New TOTP based authentication
client.get_totp_session(auth.client_id, pyotp.TOTP(auth.token).now(), auth.pin)
with open("algo_log.txt", "a") as f:
    f.write("MachineAlgo.py started successfully\n")

# User Inputs
START_TIME = [9, 16, 0]  # Algo Start Time
EXIT_TIME = [15, 30, 0]  # Algo End Time

Total_Cash = 10000
Max_Position = 1
Total_Cash_per_position = int(Total_Cash / Max_Position)

Take_Profit = 20

#Tickers = ['NIFTY 17 APR 2025 CE 23000.00','NIFTY 17 APR 2025 PE 22650.00']

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
print(instrument_df[(instrument_df.Name == 'NIFTY 03 APR 2025 CE 23300.00')])

try:
    signal_strength_model = joblib.load('signal_strength_model.pkl')
    target_price_model = joblib.load('target_price_model.pkl')
except:
    signal_strength_model = RandomForestClassifier()
    target_price_model = RandomForestRegressor()


def train_models():
    """Train the signal strength and target price models."""
    try:
        df = pd.read_excel("MarketData17m.xlsx")
        features = df[['Open','Close','low','high', 'volume', 'macd', 'macd_signal', 'delta', 'gamma', 'theta','expected_move']]
        signal_labels = df['signal_type'].apply(lambda x: 1 if x == 'BUY' else 0)  # Binary classification
        target_values = df['expected_move']

        #signal_strength_model.fit(features, signal_labels)
        #target_price_model.fit(features, target_values)

        signal_strength_model.partial_fit(features, signal_labels,classes=np.unique(signal_labels))  # Include all possible classes
        target_price_model.partial_fit(features, target_values)

        joblib.dump(signal_strength_model, 'signal_strength_model.pkl')
        joblib.dump(target_price_model, 'target_price_model.pkl')
        print("Models trained and saved successfully.")
    except Exception as e:
        print("Error in training models:", e)


train_models()

def predict_signal_strength(Open,Close,Low,High, volume, macd, macd_signal, delta, gamma, theta):
    """Predict whether a signal is weak, medium, or strong."""
    if not hasattr(signal_strength_model, "estimators_"):
        raise ValueError("Error: The signal strength model is not trained yet.")

    features = np.array([[Open,Close,Low,High, volume, macd, macd_signal, delta, gamma, theta]])
    return signal_strength_model.predict(features)[0]

def estimate_target_price(Open,Close,Low,High, volume, macd, macd_signal, delta, gamma, theta):
    """Estimate the expected price movement after a signal."""
    features = np.array([[Open,Close,Low,High,volume, macd, macd_signal, delta, gamma, theta]])
    return target_price_model.predict(features)[0]

signal_data = []
def store_signal_data(Open,Close,Low,High, volume, macd, macd_signal, delta, gamma, theta, signal_type,optype,strike):
    """Store signals with relevant data and save to Excel."""
    signal_entry = {
        'timestamp': [dt.datetime.now()],  # ✅ Wrap in a list
        'OptionType':[optype],
        'Strike':[strike],
        'Open': [Open],
        'Close': [Close],
        'Low': [Low],
        'High': [High],
        'Volume': [volume],
        'MACD': [macd],
        'MACD_Signal': [macd_signal],
        'Delta': [delta],
        'Gamma': [gamma],
        'Theta': [theta],
        'Signal_Type': [signal_type]
    }
    df_new = pd.DataFrame(signal_entry)
    #signal_data.append(signal_entry)
    df_new = pd.DataFrame(signal_entry)
    file_path = "signal_data.xlsx"
    # Convert to DataFrame and save to Excel
    # Append data to existing Excel file
    if os.path.exists(file_path):
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
            df_new.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    else:
        df_new.to_excel(file_path, index=False)  # Create new file if it doesn't exist
# Getting Script Code
def scripcode_lookup(instrument=instrument_df, symbol='TCS'):
    ## This function is used to find the instrument token number
    try:
        return instrument[instrument.Name == symbol].ScripCode.values[0]
    except:
        return -1



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

    # Save to Excel
    file_name = "MarketData.xlsx"
    if os.path.exists(file_name):
        existing_data = pd.read_excel(file_name)  # Load existing data
        opt_data = pd.concat([existing_data, opt_data], ignore_index=True)  # Append new data
    else:
        print("Creating new MarketData.xlsx file")

    return opt_data


def get_cash_market_data(symbol, timeframe):
    scriptcode = scripcode_lookup(instrument_df, symbol)
    sym=symbol
    print(scriptcode)
    parts = symbol.split()
    ticker = parts[0]  # Extract ticker
    expiry = f"{parts[1]} {parts[2]} {parts[3]}"  # Extract expiry date
    opttype = parts[4]  # Extract option type (CE/PE)
    strike = float(parts[5])  # Extract strike price and convert to float
    print(strike)
    df = pd.DataFrame(client.historical_data(Exch='N', ExchangeSegment='D', ScripCode=scriptcode, time=timeframe,
                                             From=dt.date.today()-dt.timedelta(2), To=dt.date.today()))
    print(df)
    df.set_index("Datetime", inplace=True)
    df["Option_Type"] = opttype
    df["Strike_Price"] = strike

    option_data = fetch_option_data(symbol)
    print(option_data.columns)
    if opttype == 'CE':
        df["delta"] = option_data['CE_Delta'].iloc[0]
        df["gamma"] = option_data['CE_Gamma'].iloc[0]
        df["theta"] = option_data['CE_Theta'].iloc[0]
        df["Spot"] = option_data['SPOT'].iloc[0]
        df["OI"] = option_data['OI'].iloc[0]

    else:
        df["delta"] = option_data['PE_Delta'].iloc[0]
        df["gamma"] = option_data['PE_Gamma'].iloc[0]
        df["theta"] = option_data['PE_Theta'].iloc[0]
        df["Spot"] = option_data['SPOT'].iloc[0]
        df["OI"] = option_data['OI'].iloc[0]


    if opttype == 'CE':

        file_name = 'SuperTrend_Long_Trades.xlsx'

        if os.path.exists(file_name):
            existing_data = pd.read_excel(file_name)  # Load existing data
            df_c = pd.concat([existing_data, df], ignore_index=True)  # Append new data
        else:
            print("Creating new SuperTrend_Long_Trades.xlsx file")

    else:

        file_name = 'SuperTrend_Short_Trades.xlsx'

        # saving the excel
        #df.to_excel(file_name)
        #print('DataFrame is written to Excel File successfully.')
        if os.path.exists(file_name):
            existing_data = pd.read_excel(file_name)  # Load existing data
            df_p = pd.concat([existing_data, df], ignore_index=True)  # Append new data
        else:
            print("Creating new SuperTrend_Short_Trades.xlsx file")

    print(df)
    return df




def score_signal(row, lookback_data):
    score = 0
    total_points = 10  # Max score

    # MACD strength
    if row['macd'] > row['macd_signal'] and (row['macd'] - row['macd_signal']) > 0.3:
        score += 2

    # Delta average over last 5 candles
    if lookback_data['delta'].mean() > 0.25:
        score += 2

    # Gamma positive trend
    if (lookback_data['gamma'] > 0).sum() >= 3:
        score += 1.5

    # Theta decay not too fast
    if lookback_data['theta'].mean() > -10:
        score += 1

    # Volume spike (current > avg)
    if row['Volume'] > lookback_data['Volume'].mean():
        score += 1.5

    # OI increasing trend
    oi_changes = lookback_data['OI'].diff()
    if (oi_changes > 0).sum() >= 3:
        score += 1.5

    # Bonus for ATM or ITM (less than 100pt from spot)
    if abs(row['Strike_Price'] - row['Close']) < 100:
        score += 0.5

    return score, total_points

def super_trend(data, period=5, mul=1):

        import pandas_ta as ta
        import numpy as np

        # Indicator parameters
        fast, slow, signal = 5, 9, 9
        ema_period = 5
        box_window = 5

        # === Indicators ===
        macd = ta.macd(data['Close'], fast=fast, slow=slow, signal=signal)
        data['macd'] = macd['MACD_5_9_9']
        data['macd_signal'] = macd['MACDs_5_9_9']
        data['macd_rising'] = (data['macd']-data['macd_signal']) > 0.4

        data['EMA'] = ta.ema(data['Close'], length=ema_period)
        data['EMA20'] = ta.ema(data['Close'], length=20)
        data['EMA50'] = ta.ema(data['Close'], length=50)
        data['box_high'] = data['High'].rolling(window=box_window).max()
        data['box_low'] = data['Low'].rolling(window=box_window).min()

        # === Bullish Reversal Condition ===


        cond_bearish_candle = data['Close'].shift(1) < data['Open'].shift(1)
        cond_bullish_candle = data['Close'] > data['Open']
        cond_below_ema = (data['Close'].shift(1) < data['EMA'].shift(1)) & (data['Close'] < data['EMA'])
        cond_distance_from_ema = (data['EMA']-data['Close']) > 1.5
        condC1=data['Close'] < data['EMA20']
        condC2=data['Close'] < data['EMA50']
        condC3=data['EMA50']-data['EMA20'] < 40
        condP1=data['EMA50']-data['EMA20'] > 20
        #cond_ema_diff = (data['EMA50']-data['EMA20'] >= -15) & (data['EMA50']-data['EMA20'] <= 5)
        #cond_box_breakout = data['Close'] > data['box_high'].shift(1)

        # Combine all into final signal
        if data['Option_Type'].iloc[0] == 'PE':
            print('PE')
            data['st_sig'] = np.where(
            cond_bearish_candle & cond_bullish_candle
            #& cond_below_ema & cond_distance_from_ema & condP1
             ,
            1, 0
        )

        if data['Option_Type'].iloc[0] == 'CE':
            print('CE')
            data['st_sig'] = np.where(
                cond_bearish_candle & cond_bullish_candle &
                cond_below_ema #& cond_distance_from_ema & condC1 & condC2 & condC3
                ,
                1, 0
            )



        return data[['st_sig']]


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
required_times =  [(9, 15), (9, 18), (9, 21), (9, 24), (9, 27), (9, 30), (9, 33), (9, 36), (9, 39), (9, 42),
                                 (9, 45), (9, 48), (9, 51), (9, 54), (9, 57), (10, 0), (10, 3), (10, 6), (10, 9), (10, 12),
                                 (10, 15), (10, 18), (10, 21), (10, 24), (10, 27), (10, 30), (10, 33), (10, 36), (10, 39), (10, 42),
                                 (10, 45), (10, 48), (10, 51), (10, 54), (10, 57), (11, 0), (11, 3), (11, 6), (11, 9), (11, 12),
                                 (11, 15), (11, 18), (11, 21), (11, 24), (11, 27), (11, 30), (11, 33), (11, 36), (11, 39), (11, 42),
                                 (11, 45), (11, 48), (11, 51), (11, 54), (11, 57), (12, 0), (12, 3), (12, 6), (12, 9), (12, 12),
                                 (12, 15), (12, 18), (12, 21), (12, 24), (12, 27), (12, 30), (12, 33), (12, 36), (12, 39), (12, 42),
                                 (12, 45), (12, 48), (12, 51), (12, 54), (12, 57), (13, 0), (13, 3), (13, 6), (13, 9), (13, 12),
                                 (13, 15), (13, 18), (13, 21), (13, 24), (13, 27), (13, 30), (13, 33), (13, 36), (13, 39), (13, 42),
                                 (13, 45), (13, 48), (13, 51), (13, 54), (13, 57), (14, 0), (14, 3), (14, 6), (14, 9), (14, 12),
                                 (14, 15), (14, 18), (14, 21), (14, 24), (14, 27), (14, 30), (14, 33), (14, 36), (14, 39), (14, 42),
                                 (14, 45), (14, 48), (14, 51), (14, 54), (14, 57), (15, 0), (15, 3), (15, 6), (15, 9), (15, 12),
                                 (15, 15), (15, 18), (15, 21), (15, 24), (15, 27), (15, 30)]


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

spot_prices = {}
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
        res = requests.post("http://localhost:8000/update-ticker", json={
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
    print(h)
    data_fut = get_cash_market_data(h, '3m')
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
            #send_to_ui("NIFTY 17 APR 2025 CE 23400.00", 22400.00)
            time.sleep(0.5)

            if is_required_time():
                data_fut = get_cash_market_data(i, '3m')
                data_fut.drop(data_fut.tail(1).index, inplace=True)
                super_trend(data_fut)

                data_list[i] = data_fut

                super_Trend_Long = pd.read_excel(Long_Trade_File)
                Long_Open_Position = super_Trend_Long[(super_Trend_Long['Trade Status'] == 'OPEN')]

                super_Trend_Short = pd.read_excel(Short_Trade_File)
                Short_Open_Position = super_Trend_Short[(super_Trend_Short['Trade Status'] == 'OPEN')]

                # Checking For SuperTrend Long
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

                    current_price = float(spot_prices1[i])

                    Trade_quantity = 75#int(math.floor(Total_Cash_per_position / current_price))

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

                    update_long_trades(i, entry_time, BuyPrice, Target_Price, Sprice, Trade_quantity, Long_Trade_File)

                    tele_msg("Long Entry Taken For "+i+" Total Quantity "+str(
                        Trade_quantity)+" And the Target Price is "+str(Target_Price) + "And Buy Price is"+str(BuyPrice))

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
            if i in Long_Open_Position['Symbol'].values:

                print("Find the Target Price")
                trade_row = Long_Open_Position[Long_Open_Position['Symbol'] == i]
                Target_Price = trade_row['Target Price'].values[0]
                print(Target_Price)
                # Check if current price exceeds target price
                if spot_prices1[i] > Target_Price:
                    print(f"Long Entry Target Hit for {i}. Closing position.")

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
                #S_Price = max(S_Price + (float(spot_prices[i]) - trade_row['Buy Price']), S_Price)
                print(S_Price)


                # Check if current price exceeds target price
                if spot_prices1[i] < S_Price:
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
                    Trade_Status = "Target Hit"

                    close_long_trade(i, Exit_Time, Sell_Price, Points, Brokerage, Profit_Loss, Trade_Status,
                                     Long_Trade_File)

                    tele_msg(f"Long Entry Target Hit for {i}. Exit Price: {Sell_Price}, P/L: {Profit_Loss}")

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



    except Exception as e:
        print("Error:", e)
        print("Oops!", e.__class__, "occurred.")

        ct = dt.datetime.now().strftime("%d-%b-%Y %I:%M%p")
        error_message = f"{ct} - An error occurred: {e}"
        tele_msg(error_message)

        with open("error_log.txt", "a") as error_log_file:
            error_log_file.write(error_message+"\n")
        raise ValueError("I have raised an Exception in main")



















