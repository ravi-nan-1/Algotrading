import React, { useEffect, useState } from 'react';
import axios from 'axios';

//const API_URL = 'http://localhost:8000';
const API_URL = import.meta.env.VITE_API_URL;

export default function App() {
  const [status, setStatus] = useState('unknown');
  const [trades, setTrades] = useState([]);
  const [openTrades, setOpenTrades] = useState([]);
  const [logs, setLogs] = useState([]);
  const [ticker, setTicker] = useState({});
  const [currentPnL, setCurrentPnL] = useState(0);
  const [todayPnL, setTodayPnL] = useState(0);
  const [tickers, setTickers] = useState([]);
  const [newTickersText, setNewTickersText] = useState("");

  const fetchStatus = async () => {
    const res = await axios.get(`${API_URL}/status`);
    setStatus(res.data.algo_status);
  };

  const fetchTrades = async () => {
    const res = await axios.get(`${API_URL}/trades`);
    const today = new Date().toISOString().split('T')[0];
    const filtered = res.data.filter(trade => {
      const exitTime = trade["Exit Time"];
      if (!exitTime) return false;
      const tradeDate = new Date(exitTime).toISOString().split('T')[0];
      return tradeDate === today;
    });
    setTrades(filtered);
  };

  const fetchOpenTrades = async () => {
    const res = await axios.get(`${API_URL}/open-trades`);
    setOpenTrades(res.data);
  };

  const fetchLogs = async () => {
    const res = await axios.get(`${API_URL}/logs`);
    setLogs(res.data.logs);
  };

  const fetchTicker = async () => {
    const res = await axios.get(`${API_URL}/ticker`);
    console.log("Fetched ticker data:", res.data);  // Debugging
    setTicker(prev => ({ ...prev, ...res.data }));
  };

  const fetchYahooSpotPrices = async () => {
    try {
      const res = await axios.get(`${API_URL}/spot-ticker`);
      const { NIFTY, BANKNIFTY } = res.data;
      setTicker(prev => ({
        ...prev,
        NIFTY: NIFTY ?? 0,
        BANKNIFTY: BANKNIFTY ?? 0
      }));
    } catch (err) {
      console.error("Backend Yahoo Fetch Error", err);
    }
  };

  const fetchPnLs = async () => {
    const res1 = await axios.get(`${API_URL}/pnl/current`);
    const res2 = await axios.get(`${API_URL}/pnl/today`);
    setCurrentPnL(res1.data.current_pnl);
    setTodayPnL(res2.data.pnl_today);
  };

  const fetchTickers = async () => {
    const res = await axios.get(`${API_URL}/tickers`);
    setTickers(res.data.tickers);
    setNewTickersText(res.data.tickers.join("\n"));
  };

  const updateTickers = async () => {
    const updated = newTickersText.split("\n").map(t => t.trim()).filter(Boolean);
    await axios.post(`${API_URL}/tickers`, { tickers: updated });
    fetchTickers();
  };

  const startAlgo = async () => {
    await axios.post(`${API_URL}/algo/start`);
    fetchStatus();
  };

  const stopAlgo = async () => {
    await axios.post(`${API_URL}/algo/stop`);
    fetchStatus();
  };

  useEffect(() => {
    fetchStatus();
    fetchTrades();
    fetchOpenTrades();
    fetchLogs();
    fetchTicker();
    fetchYahooSpotPrices();
    fetchPnLs();
    fetchTickers();
    const interval = setInterval(() => {
      fetchTicker();
      fetchYahooSpotPrices();
      fetchPnLs();
      fetchLogs();
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
   // At the top: No changes to imports and hooks

<div style={{ padding: '20px', fontFamily: 'Arial, sans-serif', background: '#f9f9f9', color: '#333' }}>
  <h1 style={{ textAlign: 'center', marginBottom: 30 }}>📊 Algo Trading Dashboard</h1>

  {/* Status Section */}
  <div style={{ marginBottom: 20 }}>
    <strong>Status:</strong> <span style={{ color: status === 'running' ? 'green' : 'red' }}>{status}</span>
    <div style={{ marginTop: 10 }}>
      <button onClick={startAlgo} style={{ padding: '8px 16px', background: 'green', color: 'white', border: 'none', marginRight: 10 }}>Start Algo</button>
      <button onClick={stopAlgo} style={{ padding: '8px 16px', background: 'red', color: 'white', border: 'none' }}>Stop Algo</button>
    </div>
  </div>

  
  <div style={{ display: 'flex', gap: '20px', marginBottom: 30 }}>
    <div style={{ flex: 1, background: 'white', padding: 20, borderRadius: 8, boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
    <h2>Live NIFTY & BANKNIFTY</h2>
    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
    <thead>
      <tr style={{ background: '#eee' }}>
        <th style={{ textAlign: 'left', padding: 8 }}>Symbol</th>
        <th style={{ textAlign: 'right', padding: 8 }}>Price (₹)</th>
      </tr>
    </thead>
    </table>
    <div>
    <h3 style={{ display: 'inline' }}>NIFTY</h3>
<p style={{ fontSize: 20, color: ticker["NIFTY"] > 0 ? 'green' : 'red', display: 'inline', marginLeft: '79%' }}>
  {ticker["NIFTY"] ? `${ticker["NIFTY"]}` : '—'}
</p>
    </div>
    <h3 style={{ display: 'inline' }}>BANKNIFTY</h3>
    <p style={{ fontSize: 20, color: ticker["BANKNIFTY"] > 0 ? 'green' : 'red',display: 'inline',marginLeft: '71%' }}>
      {ticker["BANKNIFTY"] ? `${ticker["BANKNIFTY"]}` : '—'}
    </p>
  
      



    </div>
    <div style={{ flex: 1, background: 'white', padding: 20, borderRadius: 8, boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
       {/* Live Ticker Table */}
  <h2>Live Ticker</h2>
  <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 30 }}>
    <thead>
      <tr style={{ background: '#eee' }}>
        <th style={{ textAlign: 'left', padding: 8 }}>Symbol</th>
        <th style={{ textAlign: 'right', padding: 8 }}>Price (₹)</th>
      </tr>
    </thead>
    <tbody>
      {Object.entries(ticker).filter(([key]) => key !== "NIFTY" && key !== "BANKNIFTY").map(([key, value]) => (
        <tr key={key}>
          <td style={{ padding: 8 }}>{key}</td>
          <td style={{ padding: 8, textAlign: 'right', color: value > 0 ? 'green' : 'red' }}>{value}</td>
        </tr>
      ))}
    </tbody>
  </table>
    </div>
  </div>

 

  {/* PnL Section */}
  <div style={{ display: 'flex', gap: 20, marginBottom: 30 }}>
    <div style={{ background: 'white', padding: 20, borderRadius: 8, flex: 1, boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
      <h3>Current Trade PnL</h3>
      <p style={{ fontSize: 18 }}>₹{currentPnL}</p>
    </div>
    <div style={{ background: 'white', padding: 20, borderRadius: 8, flex: 1, boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
      <h3>Today's PnL</h3>
      <p style={{ fontSize: 18 }}>₹{todayPnL}</p>
    </div>
  </div>

  {/* Open Trades Table */}
  <h2>Open Trades</h2>
  <table border="1" cellPadding="8" style={{ borderCollapse: 'collapse', width: '100%', marginBottom: 30 }}>
    <thead style={{ background: '#f0f0f0' }}>
      <tr>
        <th>Symbol</th><th>Entry Time</th><th>Buy Price</th><th>Target</th><th>Status</th>
      </tr>
    </thead>
    <tbody>
      {openTrades.map((trade, i) => (
        <tr key={i}>
          <td>{trade.Symbol}</td>
          <td>{trade['Entry Time']}</td>
          <td>{trade['Buy Price']}</td>
          <td>{trade['Target Price']}</td>
          <td>{trade['Trade Status']}</td>
        </tr>
      ))}
    </tbody>
  </table>

  {/* All Trades Today */}
  <h2>All Trades Today</h2>
  <table border="1" cellPadding="8" style={{ borderCollapse: 'collapse', width: '100%', marginBottom: 30 }}>
    <thead style={{ background: '#f0f0f0' }}>
      <tr>
        {trades.length > 0 && Object.keys(trades[0]).map((key, i) => (
          <th key={i}>{key}</th>
        ))}
      </tr>
    </thead>
    <tbody>
      {trades.map((row, i) => (
        <tr key={i}>
          {Object.values(row).map((val, j) => (
            <td key={j}>{val}</td>
          ))}
        </tr>
      ))}
    </tbody>
  </table>

  {/* Ticker Textarea */}
  <h2>Tracked Tickers</h2>
  <textarea
    rows="5"
    style={{ width: '100%', padding: 10, fontFamily: 'monospace', marginBottom: 10 }}
    value={newTickersText}
    onChange={e => setNewTickersText(e.target.value)}
  />
  <button onClick={updateTickers} style={{ padding: '8px 16px', background: '#333', color: 'white', border: 'none' }}>
    Update Tickers
  </button>

  {/* Logs */}
  <h2 style={{ marginTop: 30 }}>Logs</h2>
  <pre style={{
    background: '#000',
    color: '#0f0',
    padding: 10,
    fontSize: 12,
    borderRadius: 4,
    height: 200,
    overflowY: 'scroll'
  }}>
    {logs.join('\n')}
  </pre>
</div>

  );
}
