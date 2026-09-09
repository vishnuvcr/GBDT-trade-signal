import os
import re
import html
import math
import requests
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import warnings

# Suppress pandas FutureWarnings from yfinance
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURATION ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
THRESHOLD = 0.55  # Minimum Buy Probability (Set to 0.00 to force a test alert)
MAX_WORKERS = 10  # Number of simultaneous shards
CHUNK_SIZE = 250  # Tickers per shard

def load_tickers():
    if os.path.exists("tickers.txt"):
        with open("tickers.txt", "r") as f:
            return [line.strip() + ".NS" for line in f.readlines() if line.strip()]
    return ["DIXON.NS", "LUPIN.NS"]

def load_gbdt_model():
    if not os.path.exists("pine_script.txt"):
        print("Error: pine_script.txt not found!")
        return []
    
    with open("pine_script.txt", "r") as f:
        model_text = f.read()
        
    tree_strings = []
    for line in model_text.split('\n'):
        line = line.strip()
        if line.startswith('score +='):
            expression = line.replace('score +=', '').strip()
            tree_strings.append(expression)
    return tree_strings

def evaluate_tree(pine_str, features):
    s = pine_str
    for k, v in features.items():
        s = re.sub(r'\b' + k + r'\b', str(v), s)
        
    def eval_node(node_str):
        node_str = node_str.strip()
        if node_str.startswith('(') and node_str.endswith(')'):
            node_str = node_str[1:-1].strip()
        
        depth = 0
        q_idx, c_idx = -1, -1
        for i, char in enumerate(node_str):
            if char == '(': depth += 1
            elif char == ')': depth -= 1
            elif char == '?' and depth == 0 and q_idx == -1: q_idx = i
            elif char == ':' and depth == 0 and c_idx == -1: c_idx = i
            
        if q_idx != -1 and c_idx != -1:
            cond_str = node_str[:q_idx].strip()
            true_str = node_str[q_idx+1:c_idx].strip()
            false_str = node_str[c_idx+1:].strip()
            
            if '<=' in cond_str:
                left, right = cond_str.split('<=')
                cond = float(left.strip()) <= float(right.strip())
            else:
                cond = False
            
            return eval_node(true_str) if cond else eval_node(false_str)
        else:
            return float(node_str)

    return eval_node(s)

def send_telegram(text):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Telegram credentials missing.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID, 
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    response = requests.post(url, data=payload, timeout=10)
    if response.status_code != 200:
        print(f"Telegram API Error: {response.text}")

def process_batch(batch, tree_strings):
    """Downloads and processes a shard (chunk) of tickers simultaneously."""
    batch_signals = []
    try:
        # Download data for this specific chunk
        data = yf.download(batch, period="3mo", interval="1d", group_by="ticker", threads=True, progress=False)
        
        for t in batch:
            try:
                # Handle DataFrame structure depending on batch size
                if len(batch) > 1 and isinstance(data.columns, pd.MultiIndex):
                    d = data[t].copy()
                else:
                    d = data.copy()
                    
                # Drop dates where the stock did not trade
                d = d.dropna(subset=['Close'])
                
                if d.empty or len(d) < 55:
                    continue

                d['return_1d'] = d['Close'].pct_change()
                d['volatility_20d'] = d['return_1d'].rolling(20).std()
                
                d['sma_20'] = ta.sma(d['Close'], length=20)
                d['sma_20_ratio'] = (d['Close'] / d['sma_20']) - 1.0
                
                d['sma_50'] = ta.sma(d['Close'], length=50)
                d['sma_50_ratio'] = (d['Close'] / d['sma_50']) - 1.0
                
                d['rsi_14'] = ta.rsi(d['Close'], length=14)
                d['atr_14'] = ta.atr(d['High'], d['Low'], d['Close'], length=14)
                d['atr_ratio'] = d['atr_14'] / d['Close']

                latest = d.iloc[-1]
                
                features = {
                    'return_1d': latest['return_1d'],
                    'volatility_20d': latest['volatility_20d'],
                    'sma_20_ratio': latest['sma_20_ratio'],
                    'sma_50_ratio': latest['sma_50_ratio'],
                    'rsi_14': latest['rsi_14'],
                    'atr_14': latest['atr_14'],
                    'atr_ratio': latest['atr_ratio']
                }
                
                if pd.isna(list(features.values())).any():
                    continue

                score = 0.0
                for tree in tree_strings:
                    score += evaluate_tree(tree, features)
                    
                baseline = -3.70558681
                raw_margin = baseline + score
                prob = 1.0 / (1.0 + math.exp(-raw_margin))

                if prob >= THRESHOLD:
                    batch_signals.append({
                        "ticker": t.replace(".NS", ""), 
                        "price": latest['Close'], 
                        "prob": prob * 100 
                    })
            except Exception:
                continue
    except Exception as e:
        print(f"Error processing batch: {e}")
        
    return batch_signals

def format_row(ticker, price, prob):
    safe_ticker = html.escape(ticker)
    tv_url = f"https://in.tradingview.com/chart/?symbol=NSE:{ticker}"
    padding = " " * max(0, 10 - len(ticker))
    return f'<a href="{tv_url}">{safe_ticker}</a>{padding} | {price:<8.2f} | {prob:>6.1f}%\n'

def scan():
    tree_strings = load_gbdt_model()
    if not tree_strings:
        return

    tickers = load_tickers()
    signals = []
    
    # Chunk the tickers into smaller lists (shards)
    shards = [tickers[i:i + CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]
    
    print(f"📡 Processing {len(tickers)} tickers across {len(shards)} shards...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit each shard to a thread worker
        futures = [executor.submit(process_batch, shard, tree_strings) for shard in shards]
        
        for future in as_completed(futures):
            res = future.result()
            if res:
                signals.extend(res) # Combine the signals from each shard

    if not signals:
        print("🏁 Scan complete. No GBDT Long signals generated today.")
        return

    ist_time = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%d %b %Y, %I:%M %p")
    signals = sorted(signals, key=lambda x: x['prob'], reverse=True)

    msg = f"<b>🤖 GBDT Model Signals</b>\n<i>{ist_time}</i>\n\n"
    msg += f"<b>🟢 LONG TRIGGERS (>={int(THRESHOLD*100)}%)</b>\n<pre>\n"
    msg += f"{'TICKER':<10} | {'PRICE':<8} | {'PROB':<7}\n"
    msg += "-" * 31 + "\n"
    
    for s in signals:
        msg += format_row(s['ticker'], s['price'], s['prob'])
    
    msg += "</pre>"

    send_telegram(msg)
    print(f"✅ {len(signals)} signals found. Alert report compiled and sent to Telegram!")

if __name__ == "__main__":
    scan()
