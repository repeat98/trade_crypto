import os
import sys
import joblib
import requests
from datetime import datetime, timezone
from typing import List, Dict
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio
import html

from main import prepare_features, get_feature_columns, get_data

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
    df_raw  = get_data(sym)
    df_feat = prepare_features(df_raw)
    feats   = get_feature_columns(df_feat)
    latest  = df_feat.iloc[-1:]

    advice: Dict[str, Dict[str, str]] = {}
    for name, mdl in models[sym].items():
        prob = mdl.predict_proba(latest[feats].values)[0, 1]
        signal = 'HOLD'
        if prob > 0.55:
            signal = 'BUY'
        elif prob < 0.45:
            signal = 'SELL'

        price = (df_raw['Close'] if 'Close' in df_raw.columns 
                 else df_raw.select_dtypes(include='number').iloc[:, -1]).iloc[-1]
        qty   = TRADE_SIZE_USD / price if price > 0 else 0
        advice[name] = {
            'signal':      signal,
            'quantity':    f"{qty:.4f}",
            'probability': f"{prob:.2f}"
        }

        # Send Telegram alert on high confidence
        if prob >= 0.75 or prob <= 0.25:
            emoji = '🚀' if signal == 'BUY' else '🔻'
            text  = (
                f"<b>{html.escape(sym)} {html.escape(signal)} Alert</b> {emoji}\n"
                f"Model: <code>{html.escape(name)}</code>\n"
                f"Prob: <b>{prob:.2%}</b>\n"
                f"Qty: <code>{qty:.4f}</code> @ ${price:.2f}\n"
                f"<i>Time: {datetime.now(timezone.utc).isoformat()} UTC</i>"
            )
            send_telegram(text)

    return advice

# --- Daily Job & Scheduler ---
async def daily_job():
    print(f"[{datetime.now(timezone.utc).isoformat()}] Generating advice and summary...")
    global models
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
    # Schedule 10-minute polling for alerts
    scheduler.add_job(lambda: asyncio.create_task(interval_job()), 'cron', minute='*/10')
    # Schedule daily retrain and summary
    scheduler.add_job(lambda: asyncio.create_task(daily_job()), 'cron', hour=h, minute=m)

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
