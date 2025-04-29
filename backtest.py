#!/usr/bin/env python3
"""
Script to backtest selected cryptocurrency models using previously trained model files.
"""
import glob
import joblib
import pandas as pd
from main import get_data, prepare_features, get_feature_columns, backtest_model, plot_equity, now_utc, Config

def main():
    # List of (symbol, model name) pairs to backtest
    symbols_models = [
        ("BNB-USD", "STACK"),
        ("BTC-USD", "STACK"),
        ("NEAR-USD", "RF"),
        ("AVAX-USD", "RF"),
        ("SAND-USD", "RF"),
        ("XMR-USD", "XGB"),
        ("MATIC-USD", "XGB"),
        ("SOL-USD", "XGB"),
    ]

    ts = now_utc().strftime('%Y%m%d_%H%M%S')
    for symbol, model_name in symbols_models:
        print(f"=== Backtesting {symbol} with model {model_name} ===")
        # Fetch and prepare data
        df_raw = get_data(symbol)
        df_feat = prepare_features(df_raw)
        feature_cols = get_feature_columns(df_feat)

        # Locate the latest model file
        pattern = f"{symbol}_{model_name}_*.joblib"
        model_files = glob.glob(str(Config.MODELS_DIR / pattern))
        if not model_files:
            print(f"No model file found for {symbol} ({model_name})")
            continue
        model_file = max(model_files)  # pick the most recent by filename
        print(f"Loading model from {model_file}")
        model = joblib.load(model_file)

        # Split into train/test as in original pipeline
        split_idx = int(len(df_feat) * 0.8)
        test_df = df_feat.iloc[split_idx:]

        # Backtest and compute Sharpe
        equity_curve, sharpe_ratio = backtest_model(
            df_feat, feature_cols, model, test_df, threshold=0.0
        )
        print(f"Sharpe ratio: {sharpe_ratio:.4f}\n")

        # Plot and save equity curve
        plot_equity(equity_curve, model_name, symbol, ts)

if __name__ == '__main__':
    main()
