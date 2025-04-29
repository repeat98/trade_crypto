
#!/usr/bin/env python3
"""
update_models.py

A script to update existing cryptocurrency trading models, only saving new versions
if they perform better than the previous ones based on Sharpe ratio.
"""

import os
import sys
import logging
import pickle
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

# Import from main.py
from main import (
    get_data, add_technical_indicators, get_feature_columns,
    train_models, backtest, calculate_metrics
)

# === Configuration ===
MODELS_DIR = Path(os.getenv('MODELS_DIR', 'models'))
METRICS_DIR = Path(os.getenv('METRICS_DIR', 'metrics'))

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_updates.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_latest_model_and_metrics(symbol: str, model_type: str) -> Tuple[Optional[object], Optional[float], Optional[Path]]:
    """
    Get the latest model and its Sharpe ratio for a given symbol and model type.
    Returns (model, sharpe_ratio, model_file_path) tuple.
    """
    # Find latest model file
    model_files = list(MODELS_DIR.glob(f"{symbol}_{model_type}_*.pkl"))
    if not model_files:
        return None, None, None
    
    latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
    
    # Find corresponding metrics file
    timestamp = latest_model_file.stem.split('_')[-1]
    metrics_file = METRICS_DIR / f"{symbol}_metrics_{timestamp}.csv"
    
    if not metrics_file.exists():
        logging.warning(f"No metrics file found for {latest_model_file}")
        return None, None, None
    
    # Load model and metrics
    with open(latest_model_file, 'rb') as f:
        model = pickle.load(f)
    
    metrics = pd.read_csv(metrics_file)
    sharpe_col = f"{model_type}_Sharpe"
    if sharpe_col not in metrics.columns:
        logging.warning(f"No Sharpe ratio found for {model_type} in {metrics_file}")
        return None, None, None
    
    sharpe = metrics[sharpe_col].iloc[0]
    return model, sharpe, latest_model_file

def update_models(symbols: list, capital: float = 10000, cost: float = 0.001, thr: float = 0):
    """
    Update models for given symbols, only saving new versions if they perform better.
    """
    for symbol in symbols:
        logging.info(f"Processing {symbol}...")
        
        # Get data and prepare features
        df = get_data(symbol)
        df_feat = add_technical_indicators(df)
        features = get_feature_columns(df_feat)
        
        # Train new models
        new_models, _ = train_models(df_feat, features)
        
        # Process each model type
        for model_type, new_model in new_models.items():
            # Get existing model and its performance
            old_model, old_sharpe, old_model_file = get_latest_model_and_metrics(symbol, model_type)
            
            # Backtest new model
            bt_results = backtest(df_feat, new_model, features, capital, cost, thr)
            new_sharpe = bt_results['metrics']['Sharpe']
            
            # Compare performance
            if old_sharpe is None or new_sharpe > old_sharpe:
                # Save new model and metrics
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # If we have an old model file, use its name to overwrite
                if old_model_file:
                    model_file = old_model_file
                    logging.info(f"Overwriting existing model file: {model_file}")
                else:
                    # Create a new file if no previous model exists
                    model_file = MODELS_DIR / f"{symbol}_{model_type}.pkl"
                    logging.info(f"Creating new model file: {model_file}")
                
                with open(model_file, 'wb') as f:
                    pickle.dump(new_model, f)
                
                # Save metrics
                metrics_file = METRICS_DIR / f"{symbol}_metrics_{timestamp}.csv"
                metrics_df = pd.DataFrame([{
                    f"{model_type}_Total": bt_results['metrics']['Total'],
                    f"{model_type}_Annual": bt_results['metrics']['Annual'],
                    f"{model_type}_Vol": bt_results['metrics']['Vol'],
                    f"{model_type}_Sharpe": bt_results['metrics']['Sharpe'],
                    f"{model_type}_MaxDD": bt_results['metrics']['MaxDD'],
                    f"{model_type}_WinRate": bt_results['metrics']['WinRate'],
                    f"{model_type}_net_Total": bt_results['net_metrics']['Total'],
                    f"{model_type}_net_Annual": bt_results['net_metrics']['Annual'],
                    f"{model_type}_net_Vol": bt_results['net_metrics']['Vol'],
                    f"{model_type}_net_Sharpe": bt_results['net_metrics']['Sharpe'],
                    f"{model_type}_net_MaxDD": bt_results['net_metrics']['MaxDD'],
                    f"{model_type}_net_WinRate": bt_results['net_metrics']['WinRate']
                }])
                metrics_df.to_csv(metrics_file, index=False)
                
                logging.info(f"Saved new {model_type} model for {symbol} with Sharpe {new_sharpe:.4f}")
                if old_sharpe is not None:
                    logging.info(f"Improvement: {new_sharpe - old_sharpe:.4f}")
            else:
                logging.info(f"Skipped saving {model_type} model for {symbol}: new Sharpe {new_sharpe:.4f} <= old {old_sharpe:.4f}")

def main():
    # Read symbols from file
    with open('symbols.txt', 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]
    
    # Update models
    update_models(symbols)

if __name__ == '__main__':
    main() 