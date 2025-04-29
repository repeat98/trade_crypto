import os
import joblib
from datetime import datetime, timezone
from typing import List, Dict

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from main import prepare_features, get_feature_columns, get_data

# Configuration
MODEL_DIR = os.getenv('MODELS_DIR', './models')
SYMBOLS = os.getenv('SYMBOLS', 'BNB-USD BTC-USD NEAR-USD AVAX-USD MATIC-USD SOL-USD').split()
TRADE_SIZE_USD = float(os.getenv('TRADE_SIZE_USD', '100'))
ADVICE_TIME_UTC = os.getenv('ADVICE_TIME_UTC', '00:00')  # HH:MM

scheduler = AsyncIOScheduler(timezone=timezone.utc)
models: Dict[str, Dict[str, object]] = {}

def load_models(symbols: List[str]) -> Dict[str, Dict[str, object]]:
    loaded: Dict[str, Dict[str, object]] = {}
    for sym in symbols:
        loaded[sym] = {}
        files = sorted(
            [f for f in os.listdir(MODEL_DIR)
             if f.startswith(f"{sym}_") and f.endswith('.joblib')]
        )
        by_type: Dict[str, str] = {}
        for f in files:
            parts = f.replace('.joblib', '').split('_')
            if len(parts) >= 3:
                _, model_name, *_ = parts
                by_type[model_name] = f
        for name, fn in by_type.items():
            loaded[sym][name] = joblib.load(os.path.join(MODEL_DIR, fn))
    return loaded

def make_prediction(sym: str) -> Dict[str, str]:
    df_raw = get_data(sym)
    df_feat = prepare_features(df_raw)
    feats = get_feature_columns(df_feat)
    latest = df_feat.iloc[-1:]

    advice: Dict[str, str] = {}
    for name, mdl in models[sym].items():
        prob = mdl.predict_proba(latest[feats].values)[0, 1]
        if prob > 0.55:
            signal = 'BUY'
        elif prob < 0.45:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        # get last price
        price_col = 'Close' if 'Close' in df_raw.columns else df_raw.select_dtypes(include='number').columns[-1]
        price = df_raw[price_col].iloc[-1]
        qty = TRADE_SIZE_USD / price if price > 0 else 0

        advice[name] = {
            'signal': signal,
            'quantity': f"{qty:.4f}",
            'probability': f"{prob:.2f}"
        }
    return advice

def get_curve_data(sym: str, days: int = 30) -> Dict[str, List]:
    """
    Return the last `days` closing prices and ISO timestamps for symbol.
    """
    df = get_data(sym)
    df = df.sort_index()  # ensure chronological
    df = df.tail(days)
    # ensure timestamp strings
    timestamps = [ts.isoformat() for ts in df.index]
    closes = df['Close'].tolist() if 'Close' in df.columns else df.select_dtypes(include='number').iloc[:, -1].tolist()
    return {'timestamps': timestamps, 'closes': closes}

async def daily_job():
    print(f"Generating advice for {datetime.now(timezone.utc)} UTC...")
    global models
    models = load_models(SYMBOLS)
    for sym in SYMBOLS:
        _ = make_prediction(sym)
        # Here you could log or persist the advice if desired

def schedule_daily():
    h, m = map(int, ADVICE_TIME_UTC.split(':'))
    scheduler.add_job(daily_job, 'cron', hour=h, minute=m)