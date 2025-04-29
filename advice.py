import os
import joblib
import asyncio
from datetime import datetime, time, timezone
from typing import List, Dict

import pandas as pd
import yfinance as yf
from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from main import prepare_features, get_feature_columns, get_data

# Configuration
MODEL_DIR = os.getenv('MODELS_DIR', './models')
# Recommended symbols based on backtest performance:
# 1. BNB-USD (STACK)    Sharpe=1.75, Accuracy=52.0%, Win Rate=52.0%
# 2. BTC-USD (STACK)    Sharpe=1.55, Accuracy=51.5%, Win Rate=51.4%
# 3. NEAR-USD (RF)      Sharpe=1.51, Accuracy=52.1%, Win Rate=51.8%
# 4. AVAX-USD (RF)      Sharpe=1.40, Accuracy=52.5%, Win Rate=52.5%
# 5. MATIC-USD (XGB)    Sharpe=1.15, Accuracy=55.3%, Win Rate=55.1%
# 6. SOL-USD (XGB)      Sharpe=1.13, Accuracy=52.6%, Win Rate=52.3%
SYMBOLS = os.getenv('SYMBOLS', 'BNB-USD BTC-USD NEAR-USD AVAX-USD MATIC-USD SOL-USD').split()
ADVICE_TIME_UTC = os.getenv('ADVICE_TIME_UTC', '00:00')  # HH:MM

app = FastAPI(title="Crypto Trading Advice Service")
scheduler = AsyncIOScheduler(timezone=timezone.utc)

# Load latest model files
models = {}  # symbol -> {name: model}


def load_models(symbols: List[str]) -> Dict[str, Dict[str, object]]:
    loaded = {}
    for sym in symbols:
        loaded[sym] = {}
        pattern = f"{sym}_"
        files = sorted(
            [f for f in os.listdir(MODEL_DIR) if f.startswith(pattern) and f.endswith('.joblib')]
        )
        # pick the latest timestamped file for each model type
        by_type = {}
        for f in files:
            parts = f.replace('.joblib','').split('_')
            # filename symbol_MODEL_TS.joblib
            if len(parts) >= 3:
                _, model_name, *_ = parts
                by_type[model_name] = f
        for name, fn in by_type.items():
            loaded[sym][name] = joblib.load(os.path.join(MODEL_DIR, fn))
    return loaded


def make_prediction(sym: str) -> Dict[str, str]:
    """Fetch latest data, compute features, apply models, and generate advice."""
    # fetch and prepare
    df_raw = get_data(sym)
    df_feat = prepare_features(df_raw)
    feats = get_feature_columns(df_feat)
    latest = df_feat.iloc[-1:]

    advice = {}
    for name, mdl in models[sym].items():
        prob = mdl.predict_proba(latest[feats].values)[0,1]
        # simple threshold
        if prob > 0.55:
            signal = 'BUY'
        elif prob < 0.45:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        advice[name] = f"{signal} (p={prob:.2f})"
    return advice


async def daily_job():
    print(f"Generating advice for {datetime.now(timezone.utc)} UTC...")
    global models
    models = load_models(SYMBOLS)
    results = {}
    for sym in SYMBOLS:
        results[sym] = make_prediction(sym)
    # Here you could send the results via email, webhook, or save to DB
    print(results)


@app.on_event("startup")
async def startup_event():
    # schedule daily advice
    h, m = map(int, ADVICE_TIME_UTC.split(':'))
    scheduler.add_job(daily_job, 'cron', hour=h, minute=m)
    scheduler.start()
    # run once at startup
    asyncio.create_task(daily_job())


@app.get("/advice/{symbol}")
def get_advice(symbol: str):
    """Get the latest advice for a single symbol on demand."""
    if symbol not in models:
        return {"error": "Symbol not loaded"}
    return {symbol: make_prediction(symbol)}

@app.get("/advice")
def get_all_advice():
    """Get latest advice for all symbols."""
    return {sym: make_prediction(sym) for sym in SYMBOLS}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
