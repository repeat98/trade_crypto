import os
import sys
import joblib
import requests
from datetime import datetime, timezone
from typing import List, Dict
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio
import html
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import AverageTrueRange

from main import prepare_features, get_feature_columns

# Configuration via environment variables or defaults
MODEL_DIR       = os.getenv('MODELS_DIR', './models')
SYMBOLS         = os.getenv('SYMBOLS', 'BNB-USD BTC-USD NEAR-USD AVAX-USD MATIC-USD SOL-USD').split()
TRADE_SIZE_USD  = float(os.getenv('TRADE_SIZE_USD', '100'))
ADVICE_TIME_UTC = os.getenv('ADVICE_TIME_UTC', '00:00')  # HH:MM

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

TELEGRAM_TOKEN  = os.getenv('TELEGRAM_TOKEN')
CHAT_ID         = os.getenv('TELEGRAM_CHAT_ID')

# Ensure Telegram credentials are provided
if not TELEGRAM_TOKEN or not CHAT_ID:
    print("Error: TELEGRAM_TOKEN and TELEGRAM_CHAT_ID must be set in environment variables.")
    sys.exit(1)

scheduler = AsyncIOScheduler(timezone=timezone.utc)
models: Dict[str, Dict[str, object]] = {}

# Track last known probabilities to avoid duplicate alerts
last_probs: Dict[str, Dict[str, float]] = {}

# Track last seen prices to detect significant moves
last_prices: Dict[str, float] = {}

# Threshold for significant price change (fractional, e.g., 0.02 for 2%)
PRICE_CHANGE_THRESHOLD = float(os.getenv('PRICE_CHANGE_THRESHOLD', '0.02'))

# --- Helper: Notification via Telegram ---
def send_telegram(text: str) -> None:
    """
    Send a MarkdownV2-formatted message to the configured Telegram chat.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(url, json={
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    })
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"Failed to send Telegram message: {e}. Response: {resp.text}")

# --- Load Models ---
def load_models(symbols: List[str]) -> Dict[str, Dict[str, object]]:
    loaded: Dict[str, Dict[str, object]] = {}
    for sym in symbols:
        loaded[sym] = {}
        files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith(f"{sym}_") and f.endswith('.joblib')])
        by_type: Dict[str, str] = {}
        for f in files:
            parts = f.replace('.joblib', '').split('_')
            if len(parts) >= 3:
                _, model_name, *_ = parts
                by_type[model_name] = f
        for name, fn in by_type.items():
            loaded[sym][name] = joblib.load(os.path.join(MODEL_DIR, fn))
    return loaded

# --- Prediction & Notification ---
def make_prediction(sym: str) -> Dict[str, Dict[str, str]]:
    try:
        # Always fetch the latest price data from yfinance to avoid using stale cache
        df_raw = yf.download(sym, period="1d", interval="1m", progress=False)
        
        # Check if we got any data
        if df_raw.empty:
            print(f"Warning: No data received for {sym}")
            send_telegram(f"⚠️ Warning: No data available for {html.escape(sym)}. The symbol might be delisted or temporarily unavailable.")
            return {}
        
        # Print column names for debugging
        print(f"Columns for {sym}: {df_raw.columns.tolist()}")
        
        # Handle the case where yfinance returns columns with just the symbol name
        if isinstance(df_raw.columns, pd.MultiIndex):
            # For MultiIndex, first level holds the field names
            df_raw.columns = [level1.lower() for level1, _ in df_raw.columns]
        else:
            # For single-level index, just lowercase the column names
            df_raw.columns = [col.lower() for col in df_raw.columns]
        
        # --- Ensure presence of 'close' column ---------------------------------
        # When yfinance is called with auto_adjust=True (the new default as of
        # April 2025), the API may return an 'adjclose' column instead of 'close'.
        # For downstream feature engineering we always need a 'close' column, so
        # we create one if it's missing.
        if 'close' not in df_raw.columns:
            if 'adjclose' in df_raw.columns:
                df_raw['close'] = df_raw['adjclose']
            else:
                print(f"Error: Neither 'close' nor 'adjclose' column found for {sym}")
                print(f"Available columns: {df_raw.columns.tolist()}")
                return {}
        
        # Create a copy for feature preparation
        df_feat = df_raw.copy()
        
        # Now prepare features with the DataFrame that has a 'close' column
        df_feat = prepare_features(df_feat)
        
        # Get feature columns and latest data point
        feats = get_feature_columns(df_feat)
        latest = df_feat.iloc[-1:]
        
        price = df_raw['close'].iloc[-1]

        # Check for significant price move
        prev_price = last_prices.get(sym)
        if prev_price is None:
            last_prices[sym] = price
        else:
            change_pct = (price - prev_price) / prev_price
            if abs(change_pct) >= PRICE_CHANGE_THRESHOLD:
                direction = "📈" if change_pct > 0 else "📉"
                pct_display = f"{change_pct:.2%}"
                send_telegram(
                    f"<b>{html.escape(sym)} Significant Move {direction}</b>\n"
                    f"Price changed by {pct_display} to ${price:.2f}\n"
                    f"<i>Time: {datetime.now(timezone.utc).isoformat()} UTC</i>"
                )
                last_prices[sym] = price
                # Skip further alerts for this run
                return {}
            # Update price for next comparison
            last_prices[sym] = price
        
        # Initialize per-symbol last_probs if missing
        if sym not in last_probs:
            last_probs[sym] = {}
        
        advice: Dict[str, Dict[str, str]] = {}
        for name, mdl in models[sym].items():
            prob = mdl.predict_proba(latest[feats].values)[0, 1]
            prob_val = prob
            signal = 'HOLD'
            if prob > 0.55:
                signal = 'BUY'
            elif prob < 0.45:
                signal = 'SELL'
            
            qty   = TRADE_SIZE_USD / price if price > 0 else 0
            advice[name] = {
                'signal':      signal,
                'quantity':    f"{qty:.4f}",
                'probability': f"{prob:.2f}"
            }
            
            # Only send alert if high confidence and probability changed
            # last = last_probs[sym].get(name)
            # if (prob_val >= 0.75 or prob_val <= 0.25) and (last is None or prob_val != last):
            #     emoji = '🚀' if signal == 'BUY' else '🔻'
            #     text  = (
            #         f"<b>{html.escape(sym)} {html.escape(signal)} Alert</b> {emoji}\n"
            #         f"Model: <code>{html.escape(name)}</code>\n"
            #         f"Prob: <b>{prob:.2%}</b>\n"
            #         f"Qty: <code>{qty:.4f}</code> @ ${price:.2f}\n"
            #         f"<i>Time: {datetime.now(timezone.utc).isoformat()} UTC</i>"
            #     )
            #     send_telegram(text)
            # Update last known probability
            last_probs[sym][name] = prob_val
        
        return advice
    except Exception as e:
        print(f"Error processing {sym}: {str(e)}")
        send_telegram(f"⚠️ Error processing {html.escape(sym)}: {html.escape(str(e))}")
        return {}

# --- Daily Job & Scheduler ---
async def daily_job():
    print(f"[{datetime.now(timezone.utc).isoformat()}] Generating advice and summary...")
    global models
    global last_probs
    # Reset last known probabilities at each daily retrain
    last_probs = {sym: {} for sym in SYMBOLS}
    # Retrain models
    models = load_models(SYMBOLS)
    # Gather advice for summary
    all_advice = {}
    for sym in SYMBOLS:
        all_advice[sym] = make_prediction(sym)
    # Build and send daily summary report
    lines = ["<b>Daily Summary Report</b>"]
    for sym, adv in all_advice.items():
        for model_name, details in adv.items():
            lines.append(
                f"{sym} - {model_name}: {details['signal']} "
                f"(Prob: {details['probability']}, Qty: {details['quantity']})"
            )
    summary_text = "\n".join(lines)
    send_telegram(html.escape(summary_text).replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>'))

async def interval_job():
    """
    Poll every 10 minutes, using the latest loaded models to send alerts.
    """
    for sym in SYMBOLS:
        make_prediction(sym)

def schedule_jobs():
    h, m = map(int, ADVICE_TIME_UTC.split(':'))
    # Schedule polling for alerts every 1 minute as a coroutine
    scheduler.add_job(interval_job, 'interval', minutes=1)
    # Schedule daily retrain and summary as a coroutine
    scheduler.add_job(daily_job, 'cron', hour=h, minute=m)

async def main():
    # Schedule and start the daily job
    schedule_jobs()
    scheduler.start()
    print("Telegram alert bot started. Waiting for scheduled jobs...")
    # Send initial report on launch
    await daily_job()
    # Keep the event loop running indefinitely
    await asyncio.Event().wait()

if __name__ == '__main__':
    # Run the scheduler within an asyncio event loop
    asyncio.run(main())
