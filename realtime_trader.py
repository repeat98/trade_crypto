import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, date, timedelta, time as dtime, timezone

import joblib
import yfinance as yf
import pandas as pd

# Import feature and backtest utilities from main pipeline
from main import prepare_features, get_feature_columns, get_data, Config

# --------------------------- State Management --------------------------- #
STATE_FILE = BASE_DIR / 'trading_state.json'

def load_state():
    """Load the current trading state from file."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Corrupted state file, starting fresh")
            return {}
    return {}

def save_state(state):
    """Save the current trading state to file."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

def update_state(last_run=None, status=None, error=None):
    """Update the trading state with new information."""
    state = load_state()
    if last_run:
        state['last_run'] = last_run
    if status:
        state['status'] = status
    if error:
        state['error'] = error
    save_state(state)

def load_best_models():
    """Load the latest metrics summary and return best model name by Sharpe for each symbol."""
    metrics_dir = Config.METRICS_DIR
    csvs = sorted(metrics_dir.glob('summary_*.csv'))
    if not csvs:
        logger.warning("No metrics summary CSV found; defaulting to all models")
        return {}
    df = pd.read_csv(csvs[-1])
    # pick best model per symbol by Sharpe Ratio
    best = df.loc[df.groupby('symbol')['Sharpe Ratio'].idxmax()]
    return dict(zip(best['symbol'], best['model']))

# --------------------------- Logging Setup --------------------------- #
BASE_DIR = Path(__file__).parent.resolve()
LOG_FILE = BASE_DIR / 'realtime_trading.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger('realtime')

# -------------------------- Dashboard Utils -------------------------- #
DASHBOARD_PATH = BASE_DIR / 'dashboard.json'

def load_dashboard():
    if DASHBOARD_PATH.exists():
        with open(DASHBOARD_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_dashboard(dashboard):
    with open(DASHBOARD_PATH, 'w') as f:
        json.dump(dashboard, f, indent=2, default=str)


def get_last_sharpe(dashboard, model_key):
    for run_date in sorted(dashboard.keys(), reverse=True):
        info = dashboard[run_date].get('models', {}).get(model_key)
        if info and 'sharpe' in info:
            return info['sharpe']
    return None

# ------------------------- Trading Components ------------------------ #
def find_models(models_dir: Path, symbols=None):
    files = list(models_dir.glob('*.joblib'))
    latest = {}
    for f in files:
        parts = f.stem.split('_')  # symbol_modelname_timestamp
        if len(parts) < 3:
            continue
        symbol, model_name = parts[0], parts[1]
        if symbols and symbol not in symbols:
            continue
        key = f"{symbol}_{model_name}"
        ts = parts[2]
        if key not in latest or ts > latest[key].stem.split('_')[2]:
            latest[key] = f
    return latest


def select_qualified(latest_models, dashboard, min_sharpe=1.0):
    qualified = {}
    for key, path in latest_models.items():
        last_sharpe = get_last_sharpe(dashboard, key) or 0.0
        if last_sharpe >= min_sharpe:
            qualified[key] = path
        else:
            logger.info(f"Skipping {key}, last Sharpe={last_sharpe:.2f} < {min_sharpe}")
    return qualified


def fetch_daily_finance(symbol: str):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period='1d')
    if df.empty:
        logger.warning(f"No daily data for {symbol}")
        return {}
    row = df.iloc[0]
    return {
        'open': float(row['Open']),
        'high': float(row['High']),
        'low': float(row['Low']),
        'close': float(row['Close']),
        'volume': int(row['Volume'])
    }


def retrain_and_evaluate(model_path: Path, symbol: str, prev_sharpe: float):
    model = joblib.load(model_path)
    # load and prepare full data
    df_raw = get_data(symbol)
    df_feat = prepare_features(df_raw)
    feats = get_feature_columns(df_feat)
    # train and evaluate like in main.py
    models, metrics = train_and_evaluate(df_feat, feats, symbol)
    # select best model by Sharpe Ratio
    best_idx = metrics['Sharpe Ratio'].idxmax()
    best_row = metrics.loc[best_idx]
    best_name = best_row['model']
    best_sharpe = best_row['Sharpe Ratio']
    best_model = models[best_name]
    # skip retraining if no improvement
    if best_sharpe <= prev_sharpe:
        logger.info(f"Skipping retrain for {model_path.name}: previous Sharpe {prev_sharpe:.2f}, best {best_sharpe:.2f}")
        return model_path, prev_sharpe, None
    # save (overwrite) new best model
    joblib.dump(best_model, model_path)
    logger.info(f"Overwrote {model_path.name}: Sharpe {prev_sharpe:.2f} -> {best_sharpe:.2f}")
    # backtest best model on test set (80/20 split)
    split = int(len(df_feat) * 0.8)
    test = df_feat.iloc[split:]
    eq, _, _ = backtest_model(df_feat, feats, best_model, test, 0.0)
    return model_path, best_sharpe, eq


def generate_signal(model_path: Path, symbol: str):
    model = joblib.load(model_path)
    df_raw = get_data(symbol)
    df_feat = prepare_features(df_raw)
    feats = get_feature_columns(df_feat)
    latest = df_feat.iloc[-1:]
    prob = model.predict_proba(latest[feats].values)[0,1]
    if prob > 0.55:
        return 1
    if prob < 0.45:
        return -1
    return 0


def allocate_capital(sharpes: dict, total_capital: float):
    total = sum(sharpes.values())
    if total == 0:
        return {k: 0.0 for k in sharpes}
    return {k: total_capital * (v/total) for k, v in sharpes.items()}


def execute_orders(signals: dict, allocations: dict):
    trades = {}
    for key, sig in signals.items():
        action = 'buy' if sig > 0 else 'sell' if sig < 0 else 'hold'
        amt = allocations.get(key, 0)
        trades[key] = {'action': action, 'amount': amt}
        logger.info(f"Order: {key} -> {action} ${amt:.2f}")
    return trades

# -------------------------- Main Routine ---------------------------- #
def run_day(capital: float, symbols: list):
    today = date.today().isoformat()
    dashboard = load_dashboard()
    dashboard.setdefault(today, {'models': {}, 'equity': {}, 'actions': {}, 'prices': {}})

    # select best pre-trained model per symbol based on metrics
    best_models = load_best_models()
    latest = find_models(Config.MODELS_DIR, symbols)
    selected_models = {}
    for symbol in symbols:
        model_name = best_models.get(symbol)
        if model_name:
            key = f"{symbol}_{model_name}"
            path = latest.get(key)
            if path:
                selected_models[symbol] = path
            else:
                logger.warning(f"No model file for {key}")
        else:
            logger.warning(f"No metrics entry for {symbol}")

    # generate signals and track equity
    signals = {}
    day_models = {}
    equity = {}
    actions = {}
    prices = {}
    
    for symbol, path in selected_models.items():
        # Generate signal
        sig = generate_signal(path, symbol)
        signals[symbol] = sig
        
        # Store model info
        day_models[symbol] = {
            'model': path.name,
            'signal': sig
        }
        
        # Get latest price data for equity calculation
        price_data = fetch_daily_finance(symbol)
        if not price_data:
            continue
            
        # Store price data
        prices[symbol] = {
            'open': price_data['open'],
            'high': price_data['high'],
            'low': price_data['low'],
            'close': price_data['close']
        }
            
        # Calculate equity based on previous day's position and today's price change
        prev_equity = dashboard.get(today, {}).get('equity', {}).get(symbol, capital/len(symbols))
        price_change = (price_data['close'] - price_data['open']) / price_data['open']
        
        # Update equity based on signal
        if sig > 0:  # Buy signal
            equity[symbol] = prev_equity * (1 + price_change)
            actions[symbol] = 'buy'
        elif sig < 0:  # Sell signal
            equity[symbol] = prev_equity * (1 - price_change)
            actions[symbol] = 'sell'
        else:  # Hold signal
            equity[symbol] = prev_equity
            actions[symbol] = 'hold'

    # Update dashboard
    dashboard[today]['models'] = day_models
    dashboard[today]['equity'] = equity
    dashboard[today]['actions'] = actions
    dashboard[today]['prices'] = prices

    save_dashboard(dashboard)
    logger.info(f"Completed run for {today}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+', required=True, help='List of symbols to trade')
    parser.add_argument('--capital', type=float, default=10000, help='Total capital to allocate')
    args = parser.parse_args()
    
    logger.info(f"Starting realtime trading for symbols {args.symbols} with capital ${args.capital:.2f}")
    
    # Load previous state
    state = load_state()
    last_run = state.get('last_run')
    if last_run:
        last_run = datetime.fromisoformat(last_run)
        logger.info(f"Resuming from last run at {last_run.isoformat()}")
    
    while True:
        try:
            now = datetime.now(timezone.utc)
            today = now.date()
            
            # Check if we need to run today
            if last_run and last_run.date() == today:
                logger.info("Already ran today, waiting for next day")
                next_run = datetime.combine(today + timedelta(days=1), dtime(0,5), tzinfo=timezone.utc)
            else:
                # Run trading for today
                update_state(status='running')
                run_day(args.capital, args.symbols)
                update_state(last_run=now.isoformat(), status='completed')
                next_run = datetime.combine(today + timedelta(days=1), dtime(0,5), tzinfo=timezone.utc)
            
            # Calculate sleep time
            sleep_secs = (next_run - now).total_seconds()
            if sleep_secs > 0:
                logger.info(f"Sleeping {sleep_secs/3600:.2f}h until next run at {next_run.isoformat()}")
                time.sleep(sleep_secs)
            else:
                logger.warning("Next run time is in the past, running immediately")
                
        except Exception as e:
            error_msg = f"Error in daily run: {str(e)}"
            logger.exception(error_msg)
            update_state(error=error_msg)
            # Sleep for 5 minutes before retrying
            time.sleep(300)

if __name__ == '__main__':
    main()