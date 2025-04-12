from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import subprocess
import datetime as dt
import os
import sys
from auth import get_live_ltp
import json
from typing import List
import httpx

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔁 Global states
algo_process = None
algo_running = False

TICKERS_FILE = "tickers.json"
spot_prices = {}

# 🧾 Model for incoming request
class TickerUpdate(BaseModel):
    tickers: List[str]

# 📄 Load tickers from file
def load_tickers():
    if not os.path.exists(TICKERS_FILE):
        return []
    with open(TICKERS_FILE, "r") as f:
        return json.load(f)

# 💾 Save tickers to file
def save_tickers(tickers):
    with open(TICKERS_FILE, "w") as f:
        json.dump(tickers, f, indent=2)

# 🏷️ Dummy LTP fetch (replace with actual function)
def get_live_ltp(symbol: str) -> float:
    return spot_prices.get(symbol, 0.0)  # or your real-time fetch logic

# 📊 GET endpoint: fetch current ticker prices
@app.get("/ticker")
def get_ticker():
    tickers = load_tickers()
    data = {symbol: get_live_ltp(symbol) for symbol in tickers}
    return data

@app.get("/get-ticker")
async def get_ticker():
    return {"spot_prices": spot_prices}

# 📜 POST endpoint: update tickers list
@app.post("/tickers")
def update_tickers(ticker_update: TickerUpdate):
    with open(TICKERS_FILE, "w") as f:
        json.dump(ticker_update.tickers, f)
    return {"status": "updated"}

# 🧳 GET endpoint: get all tickers
@app.get("/tickers")
def get_tickers():
    try:
        with open(TICKERS_FILE, "r") as f:
            tickers = json.load(f)
    except FileNotFoundError:
        tickers = []
    return {"tickers": tickers}

@app.get("/status")
def get_status():
    return {"algo_status": "running" if algo_running else "stopped"}

@app.get("/trades")
def get_all_trades():
    try:
        df_long = pd.read_excel("SuperTrend_Long.xlsx")
        df_short = pd.read_excel("SuperTrend_Short.xlsx")
        df = pd.concat([df for df in [df_long, df_short] if not df.empty and not df.isna().all().all()], ignore_index=True)
        return df.fillna("").to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

@app.get("/open-trades")
def open_trades():
    try:
        df = pd.read_excel("SuperTrend_Long.xlsx")
        df_open = df[df["Trade Status"] == "OPEN"]
        return df_open.fillna("").to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Ticker update model
class TickerUpdateModel(BaseModel):
    symbol: str
    price: float

# Endpoint to update the ticker price
@app.post("/update-ticker")
async def update_ticker(payload: TickerUpdateModel):
    symbol = payload.symbol
    price = payload.price

    if price <= 0:
        raise HTTPException(status_code=400, detail="Price must be greater than zero")

    # Update the price in the dictionary
    spot_prices[symbol] = price
    return {"status": "ok", "message": f"Updated {symbol} to ₹{price}"}

@app.get("/pnl/current")
def get_current_pnl():
    df_long = pd.read_excel("SuperTrend_Long.xlsx")
    df_short = pd.read_excel("SuperTrend_Short.xlsx")
    df = pd.concat([df for df in [df_long, df_short] if not df.empty and not df.isna().all().all()], ignore_index=True)
    open_trades = df[df['Trade Status'].str.upper() == "OPEN"]
    total_pnl = 0
    details = []
    for _, trade in open_trades.iterrows():
        symbol = trade['Symbol']
        buy_price = float(trade['Buy Price'])
        qty = float(trade.get('Quantity', 1))
        ltp = get_live_ltp(symbol)
        pnl = (ltp - buy_price) * qty
        total_pnl += pnl
        details.append({"Symbol": symbol, "PnL": round(pnl, 2), "LTP": ltp})
    return {"current_pnl": round(total_pnl, 2), "details": details}

@app.get("/pnl/today")
def pnl_today():
    try:
        today = dt.datetime.now().strftime("%d-%b-%Y %I:%M%p")

        df_long = pd.read_excel("SuperTrend_Long.xlsx")
        df_short = pd.read_excel("SuperTrend_Short.xlsx")
        df_all = pd.concat([df_long, df_short], ignore_index=True)

        df_all.columns = df_all.columns.str.strip()

        df_all["Exit Time"] = pd.to_datetime(df_all["Exit Time"], errors="coerce")

        df_today = df_all[df_all["Exit Time"].dt.strftime('%Y-%m-%d') == today]

        total_pnl = df_today["Points"].sum()
        return {"pnl_today": round(total_pnl, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/spot-ticker")
async def get_spot_prices():
    try:
        async with httpx.AsyncClient() as client:
            headers = {"User-Agent": "Mozilla/5.0"}
            nifty_resp = await client.get("https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI", headers=headers)
            banknifty_resp = await client.get("https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEBANK", headers=headers)

            nifty = nifty_resp.json()["chart"]["result"][0]["meta"]["regularMarketPrice"]
            banknifty = banknifty_resp.json()["chart"]["result"][0]["meta"]["regularMarketPrice"]

            return {"NIFTY": nifty, "BANKNIFTY": banknifty}
    except Exception as e:
        return {"error": str(e)}

@app.post("/algo/start")
def start_algo(background_tasks: BackgroundTasks):
    global algo_process, algo_running
    if not algo_running:
        def run_algo():
            global algo_process
            algo_process = subprocess.Popen([sys.executable, "MachineAlgo.py"])
        background_tasks.add_task(run_algo)
        algo_running = True
        return {"status": "started"}
    return {"status": "already running"}

@app.post("/algo/stop")
def stop_algo():
    global algo_process, algo_running
    if algo_running and algo_process:
        algo_process.terminate()
        algo_process.wait()
        algo_running = False
        return {"status": "stopped"}
    return {"status": "not running"}

@app.get("/logs")
def read_logs():
    try:
        if os.path.exists("error_log.txt"):
            with open("error_log.txt", "r", encoding="utf-8") as f:
                lines = f.readlines()[-20:]
                return {"logs": lines}
        return {"logs": []}
    except Exception as e:
        return {"error": str(e)}
