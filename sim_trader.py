#!/usr/bin/env python3
"""
sim_trader.py

A simulated trading script that loads all models from the models directory,
retrieves recent price data for a specified duration, and runs a backtest
on each symbol using real-time (historical streaming) data.
Supports continuous mode on a Raspberry Pi (or any Linux) with a configurable interval.
"""
import os
import argparse
import logging
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf
import ta
import matplotlib.pyplot as plt

# === Configuration ===
MODELS_DIR = Path(os.getenv('MODELS_DIR', 'models'))
PLOTS_DIR = Path(os.getenv('PLOTS_DIR', 'sim_plots'))

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === Utility Functions ===

def map_duration_to_yf(period_str):
    mapping = {
        '1 week': '1wk',
        '1 month': '1mo',
        '3 months': '3mo',
        '6 months': '6mo',
        '1 year': '1y',
    }
    return mapping.get(period_str, '1wk')


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2['returns'] = df2['close'].pct_change()
    df2['sma20'] = ta.trend.sma_indicator(df2['close'], window=20)
    df2['ema20'] = ta.trend.ema_indicator(df2['close'], window=20)
    macd = ta.trend.MACD(df2['close'])
    df2['macd'] = macd.macd()
    df2['macd_signal'] = macd.macd_signal()
    df2['rsi'] = ta.momentum.rsi(df2['close'], window=14)
    bb = ta.volatility.BollingerBands(df2['close'])
    df2['bb_h'] = bb.bollinger_hband()
    df2['bb_l'] = bb.bollinger_lband()
    df2 = df2.ffill().bfill().dropna()
    return df2


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {'returns', 'open', 'high', 'low', 'close', 'volume'}
    return [c for c in df.columns if c not in exclude]


def generate_signals(preds: pd.Series, thr: float) -> pd.Series:
    return preds.apply(lambda x: 1 if x > thr else -1 if x < -thr else 0)


def calculate_returns(prices: pd.Series, positions: pd.Series) -> pd.Series:
    rets = prices.pct_change()
    return positions.shift(1).fillna(0) * rets


def calculate_metrics(returns: pd.Series, rf: float = 0.0) -> dict:
    total = (1 + returns).prod() - 1
    ann = (1 + total) ** (252 / len(returns)) - 1
    vol = returns.std() * np.sqrt(252)
    sr = (ann - rf) / vol if vol > 0 else np.nan
    mdd = (returns + 1).cumprod().div((returns + 1).cumprod().cummax()).min() - 1
    win = (returns > 0).mean()
    return dict(Total=total, Annual=ann, Vol=vol, Sharpe=sr, MaxDD=mdd, WinRate=win)


def backtest(df: pd.DataFrame, model, features: list, capital: float, cost: float, thr: float):
    X = df[features].values
    preds = pd.Series(model.predict(X), index=df.index)
    positions = generate_signals(preds, thr)
    rets = calculate_returns(df['close'], positions)
    tx_costs = positions.diff().abs() * cost * df['close']
    net_rets = rets - tx_costs
    equity = (1 + net_rets).cumprod() * capital
    return {
        'equity_curve': equity,
        'metrics': calculate_metrics(rets),
        'net_metrics': calculate_metrics(net_rets),
        'tx_costs': tx_costs.sum()
    }

# === Core Simulation ===


# === Duration Parsing ===
def parse_duration_to_timedelta(period_str: str) -> pd.Timedelta:
    mapping = {
        '1 week': '7D',
        '1 month': '30D',
        '3 months': '90D',
        '6 months': '180D',
        '1 year': '365D',
    }
    return pd.to_timedelta(mapping.get(period_str, '7D'))


def run_once(args):
    # Determine lookback period needed for indicators + user duration
    user_delta = parse_duration_to_timedelta(args.duration)
    indicator_lookback = pd.Timedelta('30D')  # covers 20-day SMA + buffer
    total_lookback = user_delta + indicator_lookback

    # Fetch ample data for indicators
    interval = args.interval
    yf_period = None  # use start/end instead of period string
    # Get current UTC time and drop timezone to create a naive timestamp
    end = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    start = end - total_lookback

    logging.info(f"Fetching data from {start.date()} to {end.date()} (interval={interval})")

    model_files = list(MODELS_DIR.glob("*.pkl"))
    symbols = sorted({f.name.split('_')[0] for f in model_files})
    summary = []

    for sym in symbols:
        logging.info(f"Simulation start: {sym}")
        # Fetch full lookback range
        df_raw = yf.Ticker(sym).history(start=start, end=end, interval=interval)
        if df_raw.empty:
            logging.warning(f"No data for {sym}, skipping.")
            continue
        df_raw = df_raw.rename(columns={
            'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'
        })
        # Ensure index is tz-naive for slicing
        if hasattr(df_raw.index, 'tz') and df_raw.index.tz is not None:
            df_raw.index = df_raw.index.tz_localize(None)

        # Apply indicators on the full lookback
        df_full = df_raw[['open','high','low','close','volume']]
        df_feat_full = add_technical_indicators(df_full)

        # Now slice to user-specified duration from the feature set
        df_feat = df_feat_full.loc[df_feat_full.index >= (end - user_delta)]
        if df_feat.empty:
            logging.warning(f"No data after applying indicators in the last {args.duration} for {sym}, skipping.")
            continue

        features = get_feature_columns(df_feat)

        for mf in [f for f in model_files if f.name.startswith(sym + '_')]:
            model_type = mf.stem.split('_')[1]
            ts = mf.stem.split('_')[-1]
            with open(mf, 'rb') as f:
                model = pickle.load(f)
            result = backtest(df_feat, model, features, args.capital, args.cost, args.thr)
            # Plot
            plt.figure(figsize=(10,6))
            result['equity_curve'].plot(title=f"{sym} {model_type} Equity")
            plt.xlabel('Date'); plt.ylabel('Value'); plt.grid(True)
            plot_file = PLOTS_DIR / f"{sym}_{model_type}_equity_{ts}.png"
            plot_file.parent.mkdir(exist_ok=True)
            plt.savefig(plot_file); plt.close()
            logging.info(f"Plot saved: {plot_file}")
            row = {'Symbol': sym, 'Model': model_type, **result['metrics'],
                   **{f'net_{k}': v for k,v in result['net_metrics'].items()},
                   'Costs': result['tx_costs']}
            summary.append(row)

    if summary:
        df = pd.DataFrame(summary).sort_values(by='Sharpe', ascending=False)
        print("\n---- Simulation Summary ----")
        print(df.to_string(index=False))
    else:
        print("No simulations run.")

# === Entrypoint ===

def main():
    parser = argparse.ArgumentParser(description='Simulated Trader Continuous')
    parser.add_argument('--thr', type=float, default=0.0, help='Signal threshold')
    parser.add_argument('--duration', type=str, default='1 week',
                        choices=['1 week','1 month','3 months','6 months','1 year'])
    parser.add_argument('--capital', type=float, default=10000.0)
    parser.add_argument('--cost', type=float, default=0.001)
    parser.add_argument('--interval', type=str, default='1d',
                        help="Data interval (e.g. '1d','1h','1m')")
    parser.add_argument('--continuous', action='store_true',
                        help='Run continuously with sleep interval')
    parser.add_argument('--sleep', type=int, default=3600,
                        help='Seconds to sleep between runs when continuous')
    args = parser.parse_args()

    # For live-duration control
    start_time = datetime.now()
    end_time = start_time + parse_duration_to_timedelta(args.duration)

    if args.continuous:
        logging.info("Starting continuous simulation mode")
        try:
            while datetime.now() < end_time:
                run_once(args)
                logging.info(f"Sleeping for {args.sleep} seconds")
                time.sleep(args.sleep)
            logging.info("Live simulation duration reached; stopping.")
            print("Live simulation completed for duration: ", args.duration)
            return
        # except KeyboardInterrupt:
        #     logging.info("Continuous simulation stopped by user")
        except KeyboardInterrupt:
            pass
    else:
        run_once(args)

if __name__ == '__main__':
    main()
