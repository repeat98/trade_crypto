"""
main.py

A pipeline for fetching, caching, feature-engineering, modeling, backtesting, and plotting equity curves for cryptocurrency symbols.
Leverages yfinance for data, SQLAlchemy for caching, ta for technical indicators, and scikit-learn/XGBoost for machine learning.
"""

import os
import sys
import logging
from logging.config import dictConfig
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import multiprocessing

import pandas as pd
import numpy as np
import sqlalchemy as sa
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from joblib import Memory
import joblib
import yfinance as yf
import ta
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.base import ClassifierMixin
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from dataclasses import dataclass

# Force multiprocessing to use 'fork' to avoid ResourceTracker errors with loky
try:
    multiprocessing.set_start_method('fork')
except RuntimeError:
    pass

__version__ = "1.0.0"

def now_utc() -> datetime:
    """Return current UTC datetime with timezone."""
    return datetime.now(timezone.utc)

@dataclass
class Config:
    DB_URL: str = os.getenv('DB_URL', f"sqlite:///{Path(__file__).parent.resolve() / 'crypto_data.db'}")
    MODELS_DIR: Path = Path(os.getenv('MODELS_DIR', Path(__file__).parent.resolve() / 'models'))
    PLOTS_DIR: Path = Path(os.getenv('PLOTS_DIR', Path(__file__).parent.resolve() / 'plots'))
    METRICS_DIR: Path = Path(os.getenv('METRICS_DIR', Path(__file__).parent.resolve() / 'metrics'))
    CACHE_REFRESH_DAYS: int = int(os.getenv('CACHE_REFRESH_DAYS', '1'))

BASE_DIR = Path(__file__).parent.resolve()
CACHE_REFRESH = timedelta(days=Config.CACHE_REFRESH_DAYS)
CACHE_DIR = BASE_DIR / 'cache'
memory = Memory(location=CACHE_DIR, verbose=0, mmap_mode=None)

# ------------------------- Logging Setup ------------------------- #
LOG_CONFIG = {
    'version': 1,
    'formatters': {
        'default': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        }
    },
    'handlers': {
        'console': {'class': 'logging.StreamHandler', 'formatter': 'default'},
        'file': {
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'crypto_pipeline.log',
            'formatter': 'default'
        }
    },
    'root': {'handlers': ['console', 'file'], 'level': 'INFO'}
}

dictConfig(LOG_CONFIG)
logger = logging.getLogger(__name__)

# ------------------------ Database Utilities ------------------------ #
engine = sa.create_engine(Config.DB_URL, echo=False)
metadata = sa.MetaData()
crypto_table = sa.Table(
    'crypto_data', metadata,
    sa.Column('date', sa.Date, primary_key=True),
    sa.Column('symbol', sa.String, primary_key=True),
    sa.Column('open', sa.Float), sa.Column('high', sa.Float),
    sa.Column('low', sa.Float), sa.Column('close', sa.Float),
    sa.Column('volume', sa.Float)
)
update_table = sa.Table(
    'last_update', metadata,
    sa.Column('symbol', sa.String, primary_key=True),
    sa.Column('last_update', sa.DateTime)
)
metadata.create_all(engine)

# ------------------------ Data Fetching & Caching ------------------------ #
@memory.cache
def fetch_data(symbol: str) -> pd.DataFrame:
    """Fetch the full OHLCV price history for a symbol from introduction to today using Yahoo Finance and clean it.
    Args:
        symbol: Cryptocurrency ticker string.
    Returns:
        DataFrame with open, high, low, close, volume columns covering the entire market history."""
    logger.info(f"Downloading {symbol}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start="1900-01-01", interval='1d', actions=False)
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    df.rename(columns=str.lower, inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.ffill().bfill()

def get_data(symbol: str) -> pd.DataFrame:
    """Load data for a symbol from DB if fresh, else fetch & store.
    Args:
        symbol: Cryptocurrency ticker string.
    Returns:
        DataFrame of OHLCV data indexed by date."""
    with engine.connect() as conn:
        last = None
        try:
            last = conn.execute(
                sa.select(update_table.c.last_update)
                    .where(update_table.c.symbol == symbol)
            ).scalar()
        except OperationalError as e:
            logger.error(f"DB error checking cache for {symbol}: {e}")
            last = None
        # ensure last_update is timezone-aware (assume stored in UTC)
        if last and last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        if last and now_utc() - last < CACHE_REFRESH:
            logger.info(f"Loading cached {symbol} (updated {last})")
            try:
                df = pd.read_sql(
                    sa.select(crypto_table)
                      .where(crypto_table.c.symbol == symbol)
                      .order_by(crypto_table.c.date),
                    conn, index_col='date', parse_dates=['date']
                )
                return df
            except OperationalError as e:
                logger.error(f"DB error reading cache for {symbol}: {e}")
                last = None

    df = fetch_data(symbol)
    save_data(df, symbol)
    return df

def save_data(df: pd.DataFrame, symbol: str) -> None:
    """Persist new data to DB.
    Args:
        df: DataFrame with OHLCV data indexed by date.
        symbol: Cryptocurrency ticker string."""
    df = df.copy()
    df.index.name = 'date'
    df['symbol'] = symbol
    records = df.reset_index().to_dict(orient='records')
    try:
        with engine.begin() as conn:
            conn.execute(sa.delete(crypto_table).where(
                crypto_table.c.symbol == symbol
            ))
            conn.execute(crypto_table.insert(), records)
            # Use SQLite dialect insert for upsert
            insert_stmt = sqlite_insert(update_table).values(
                symbol=symbol,
                last_update=now_utc()
            )
            do_update_stmt = insert_stmt.on_conflict_do_update(
                index_elements=['symbol'],
                set_={'last_update': now_utc()}
            )
            conn.execute(do_update_stmt)
        logger.info(f"Saved {len(records)} rows for {symbol}")
    except SQLAlchemyError as e:
        logger.error(f"DB error: {e}")
        raise

# ------------------------ Feature Engineering ------------------------ #
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators, drop non-numeric columns, and create target.
    Args:
        df: Raw OHLCV DataFrame.
    Returns:
        DataFrame with features and target column."""
    df_feat = df.copy()
    df_feat['returns'] = df_feat['close'].pct_change()
    # drop non-numeric
    non_numeric = df_feat.select_dtypes(exclude=[np.number]).columns.tolist()
    df_feat.drop(columns=non_numeric, inplace=True, errors='ignore')

    # standard indicators
    indicators = {
        'sma20':    lambda x: ta.trend.sma_indicator(x, window=20),
        'ema20':    lambda x: ta.trend.ema_indicator(x, window=20),
        'rsi14':    lambda x: ta.momentum.rsi(x, window=14),
        'macd':     lambda x: ta.trend.MACD(x).macd(),
        'macd_sig': lambda x: ta.trend.MACD(x).macd_signal(),
        'bb_h':     lambda x: ta.volatility.BollingerBands(x).bollinger_hband(),
        'bb_l':     lambda x: ta.volatility.BollingerBands(x).bollinger_lband()
    }
    for name, func in indicators.items():
        df_feat[name] = func(df_feat['close'])
    # ATR and OBV
    df_feat['atr14'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df_feat['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    # multi-horizon momentum
    for w in [5, 10, 50]:
        df_feat[f'ret_{w}'] = df['close'].pct_change(w)

    df_feat.ffill(inplace=True)
    df_feat.bfill(inplace=True)
    df_feat['target'] = np.sign(df_feat['returns'].shift(-1))
    return df_feat.dropna()

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return columns suitable for model input.
    Args:
        df: DataFrame with features and target.
    Returns:
        List of feature column names."""
    exclude = {'target', 'returns', 'symbol'} | {'open','high','low','close','volume'}
    return [c for c in df.columns if c not in exclude]

# ------------------------ Modeling & Backtest ------------------------ #

# --- Helper for randomized search ---
def build_search(pipeline: Pipeline, param_distributions: dict) -> RandomizedSearchCV:
    """Build a RandomizedSearchCV for hyperparameter tuning.
    Args:
        pipeline: sklearn Pipeline with model.
        param_distributions: parameter search space.
    Returns:
        Configured RandomizedSearchCV object."""
    return RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        cv=TimeSeriesSplit(n_splits=5),
        scoring='accuracy',
        n_iter=30,
        n_jobs=-1,
        error_score='raise',
        random_state=42
    )

def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: List[str],
    symbol: str
) -> Tuple[Dict[str, Pipeline], pd.DataFrame]:
    """Train models with hyperparameter tuning and evaluate on test set.
    Args:
        df: DataFrame with features and target.
        feature_cols: List of feature column names.
        symbol: Cryptocurrency ticker string.
    Returns:
        Tuple of trained models dict and metrics DataFrame."""
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    y_train = (train['target'] > 0).astype(int)
    y_test  = (test['target'] > 0).astype(int)

    models = {}
    metrics = []

    # define base estimators and search spaces
    configs = {
        'RF': (
            Pipeline([('scale', StandardScaler()), ('model', RandomForestClassifier(random_state=42))]),
            {'model__n_estimators': (50, 500), 'model__max_depth': (3, 20)}
        ),
        'XGB': (
            Pipeline([('scale', StandardScaler()), ('model', XGBClassifier(random_state=42))]),
            {'model__n_estimators': (50, 500), 'model__max_depth': (3, 20)}
        )
    }

    fitted = {}
    for name, (pipeline, spaces) in configs.items():
        try:
            logger.info(f"Optimizing {name} via Bayesian search...")
            search = build_search(pipeline, spaces)
            search.fit(train[feature_cols].values, y_train)
            best = search.best_estimator_
            models[name] = best
            fitted[name] = best
        except Exception as e:
            logger.error(f"Error optimizing model {name}: {e}")
            raise

    # stacking ensemble (uses KFold for final CV)
    try:
        estimators = [(n, fitted[n]) for n in ['RF', 'XGB']]
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=KFold(n_splits=5)
        )
        stack_pipe = Pipeline([('scale', StandardScaler()), ('model', stack)])
        stack_pipe.fit(train[feature_cols].values, y_train)
        models['STACK'] = stack_pipe
    except Exception as e:
        logger.error(f"Error fitting stacking model: {e}")
        raise

    # backtest each model
    from sklearn.base import clone
    for name, model in models.items():
        mdl = clone(model)
        eq, sharpe, rets = backtest_model(df, feature_cols, mdl, test, 0.0)
        # classification accuracy on the test set
        y_pred = mdl.predict(test[feature_cols].values)
        accuracy = accuracy_score(y_test, y_pred)
        # trade win rate as percentage of positive returns
        win_rate = (rets > 0).sum() / len(rets) * 100
        metrics.append({
            'symbol': symbol,
            'model': name,
            'Sharpe': sharpe,
            'Accuracy': accuracy,
            'Win Rate [%]': win_rate
        })
        # plot equity
        plot_equity(eq, name, symbol, now_utc().strftime('%Y%m%d_%H%M%S'))

    return models, pd.DataFrame(metrics)

def calculate_metrics(rets: pd.Series) -> Dict[str, float]:
    """Calculate performance metrics from returns.
    Args:
        rets: Series of returns.
    Returns:
        Dictionary with performance metrics."""
    # Annualized Sharpe ratio assuming 252 trading days
    mean_ret = rets.mean()
    std_ret = rets.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret != 0 else 0.0
    return {'Sharpe': sharpe}

# --- Dedicated backtest function ---
def backtest_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    model: ClassifierMixin,
    test: pd.DataFrame,
    threshold: float
) -> Tuple[pd.Series, float, pd.Series]:
    """Backtest the provided model on the test set, fitting it on the training portion prior to prediction.
    Args:
        df: Full DataFrame with features and target.
        feature_cols: List of feature columns.
        model: Trained or untrained classifier.
        test: Test DataFrame.
        threshold: Threshold for signal (unused here).
    Returns:
        Tuple of equity curve series, Sharpe ratio, and returns series.
    """
    # Fit the model on the training portion up to the start of the test set
    split_date = test.index[0]
    train_mask = df.index < split_date
    X_train = df.loc[train_mask, feature_cols].values
    y_train = (df.loc[train_mask, 'target'] > 0).astype(int)
    model.fit(X_train, y_train)
    # predict probabilities
    probs = model.predict_proba(test[feature_cols].values)[:,1]
    # volatility estimate
    vol = df['close'].pct_change().rolling(20).std().reindex(test.index).fillna(method='bfill')
    # raw signal and dynamic sizing
    signal = (probs - 0.5) * 2
    position = (signal / vol).clip(-1,1)
    # returns
    rets = test['close'].pct_change().fillna(0) * position.shift(1).fillna(0)
    eq = (1 + rets).cumprod() * 1.0  # capital normalized
    metrics = calculate_metrics(rets)
    return eq, metrics['Sharpe'], rets

# ------------------------ Visualization ------------------------ #
def plot_equity(
    equity: pd.Series,
    name: str,
    symbol: str,
    ts: str
) -> None:
    """Plot and save equity curve.
    Args:
        equity: Series of equity values.
        name: Model name.
        symbol: Cryptocurrency ticker.
        ts: Timestamp string."""
    Config.PLOTS_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(10,6))
    plt.plot(equity)
    plt.title(f"{symbol} Equity ({name}) @ {ts}")
    plt.xlabel('Date'); plt.ylabel('Value')
    plt.grid(True)
    file = Config.PLOTS_DIR / f"{symbol}_{name}_eq_{ts}.png"
    plt.savefig(file, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot {file}")

# ------------------------ Main Pipeline ------------------------ #

def process(
    symbol: str,
    capital: float,
    cost: float,
    threshold: float
) -> pd.DataFrame:
    """Run full pipeline for a given symbol.
    Args:
        symbol: Cryptocurrency ticker string.
        capital: Initial capital (unused currently).
        cost: Transaction cost (unused currently).
        threshold: Signal threshold (unused currently).
    Returns:
        DataFrame with performance metrics."""
    ts = now_utc().strftime('%Y%m%d_%H%M%S')
    df_raw = get_data(symbol)
    df_feat = prepare_features(df_raw)
    feats = get_feature_columns(df_feat)
    models, metrics_df = train_and_evaluate(df_feat, feats, symbol)

    # Save trained model objects
    Config.MODELS_DIR.mkdir(exist_ok=True)
    for name, model in models.items():
        model_path = Config.MODELS_DIR / f"{symbol}_{name}_{ts}.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Saved model {name} to {model_path}")

    return metrics_df

def main() -> None:
    """Command line interface entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+', required=True)
    parser.add_argument('--capital', type=float, default=10000)
    parser.add_argument('--cost', type=float, default=0.001)
    parser.add_argument('--threshold', type=float, default=0.0)
    args = parser.parse_args()

    all_metrics = []
    for sym in args.symbols:
        try:
            dfm = process(sym, args.capital, args.cost, args.threshold)
            all_metrics.append(dfm)
        except Exception as e:
            logger.exception(f"Error processing {sym}: {e}")

    if all_metrics:
        summary = pd.concat(all_metrics).sort_values('Sharpe', ascending=False)
        Config.METRICS_DIR.mkdir(exist_ok=True)
        summary.to_csv(Config.METRICS_DIR / f"summary_{now_utc():%Y%m%d_%H%M%S}.csv", index=False)
        print(summary)

if __name__ == '__main__':
    """
    CLI entry point for main2.py.

    Parameters:
        --symbols: List of cryptocurrency ticker symbols to process (required).
        --capital: Initial capital for backtesting (default 10000).
        --cost: Transaction cost per trade (default 0.001).
        --threshold: Signal threshold for trading (default 0.0).
    """
    main()

__all__ = ['process', 'main']
