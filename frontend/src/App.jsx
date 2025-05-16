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

  function parseCustomDate(dateStr) {
    // dateStr example: "16-May-2025 10:45AM"
  
    // Split date and time parts
    const [datePart, timePart] = dateStr.split(' ');
  
    // Parse date part: "16-May-2025"
    const [day, monthStr, year] = datePart.split('-');
  
    // Convert month short name to month number
    const months = {
      Jan: 0, Feb: 1, Mar: 2, Apr: 3, May: 4, Jun: 5,
      Jul: 6, Aug: 7, Sep: 8, Oct: 9, Nov: 10, Dec: 11
    };
  
    const month = months[monthStr];
    if (month === undefined) return null;
  
    // Parse time part: "10:45AM"
    const timeRegex = /(\d{1,2}):(\d{2})(AM|PM)/;
    const match = timePart.match(timeRegex);
    if (!match) return null;
  
    let hour = parseInt(match[1], 10);
    const minute = parseInt(match[2], 10);
    const meridian = match[3];
  
    if (meridian === 'PM' && hour !== 12) hour += 12;
    if (meridian === 'AM' && hour === 12) hour = 0;
  
    // Create date object
    return new Date(year, month, parseInt(day, 10), hour, minute);
  }
  

  const fetchTrades1 = async () => {
    const res = await axios.get(`${API_URL}/trades`);
    const now = new Date();
    const day = String(now.getDate()).padStart(2, '0');
    const month = now.toLocaleString('en-US', { month: 'short' });
    const year = now.getFullYear();
    const formatted = `${day}-${month}-${year}`;
    const today = `${day}-${month}-${year}`;
    const filtered = res.data.filter(trade => {
      const exitTime = trade["Entry Time"];
      if (!exitTime) return false;
      const tradeDate = `${day}-${month}-${year}`;//new Date(exitTime).toISOString().split('T')[0];
      return tradeDate === today;
    });
    setTrades(filtered);
  };


  const fetchTrades = async () => {
    const res = await axios.get(`${API_URL}/trades`);
  
    const now = new Date();
    const todayISO = now.toISOString().split('T')[0]; // "YYYY-MM-DD"
  
    const filtered = res.data.filter(trade => {
      const entryTimeStr = trade["Entry Time"];
      if (!entryTimeStr) return false;
  
      const tradeDateObj = parseCustomDate(entryTimeStr);
      if (!tradeDateObj) return false;
  
      const tradeISODate = tradeDateObj.toISOString().split('T')[0];
  
      return tradeISODate === todayISO;
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
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif', background: '#f9f9f9', color: '#333' }}>
      <h1 style={{ textAlign: 'center', marginBottom: 30 }}>📊 Algo Trading Dashboard</h1>

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
            <p style={{ fontSize: 20, color: ticker["NIFTY"] > 0 ? 'green' : 'red', display: 'inline', float: 'right' }}>{ticker["NIFTY"] || '—'}</p>
          </div>
          <div>
            <h3 style={{ display: 'inline' }}>BANKNIFTY</h3>
            <p style={{ fontSize: 20, color: ticker["BANKNIFTY"] > 0 ? 'green' : 'red', display: 'inline', float: 'right' }}>{ticker["BANKNIFTY"] || '—'}</p>
          </div>
        </div>

        <div style={{ flex: 1, background: 'white', padding: 20, borderRadius: 8, boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
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

      <h2>Open Trades</h2>
      <table border="1" cellPadding="8" style={{ borderCollapse: 'collapse', width: '100%', marginBottom: 30 }}>
        <thead style={{ background: '#f0f0f0' }}>
          <tr>
            <th>Symbol</th><th>Entry Time</th><th>Buy Price</th><th>SL Price</th><th>Target</th><th>Status</th>
          </tr>
        </thead>
        <tbody>
          {openTrades.map((trade, i) => (
            <tr key={i}>
              <td>{trade.Symbol}</td>
              <td>{trade['Entry Time']}</td>
              <td>{trade['Buy Price']}</td>
              <td>{trade['Sprice']}</td>
              <td>{trade['Target Price']}</td>
              <td>{trade['Trade Status']}</td>
            </tr>
          ))}
        </tbody>
      </table>

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
