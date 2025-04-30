import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Tuple

# Import pipeline functions
from main import get_data, prepare_features, get_feature_columns, backtest_model, now_utc, Config

import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class TestBroker:
    """
    A simple test broker that:
      - Accepts initial capital
      - Loads pre-trained models from the models directory
      - Selects top-performing models by Sharpe ratio
      - Allocates capital evenly across selected models
      - Executes a simulated backtest combining all strategies
    """
    def __init__(
        self,
        symbols: List[str],
        capital: float,
        models_dir: Path = Config.MODELS_DIR,
        threshold: float = 0.0,
        weight_by_sharpe: bool = False
    ):
        self.symbols = symbols
        self.initial_capital = capital
        self.models_dir = models_dir
        self.signal_threshold = threshold
        self.weight_by_sharpe = weight_by_sharpe
        self.results: List[Tuple[str, str, float, pd.Series]] = []  # (symbol, model_name, sharpe, equity)

    def load_models(self) -> List[Tuple[str, str, object]]:
        """
        Load serialized models from the models directory.
        Expects filenames like SYMBOL_MODEL_TIMESTAMP.joblib
        Returns list of tuples (symbol, model_name, model_obj)
        """
        loaded = []
        for file in self.models_dir.glob('*.joblib'):
            parts = file.stem.split('_')
            if len(parts) < 3:
                continue
            symbol, model_name = parts[0], parts[1]
            if symbol not in self.symbols:
                continue
            try:
                model = joblib.load(file)
                logger.info(f"Loaded model {model_name} for symbol {symbol} from {file.name}")
            except Exception as e:
                logger.warning(f"Failed to load model {file.name}: {e}")
                continue
            loaded.append((symbol, model_name, model))
        return loaded

    def run(self) -> pd.DataFrame:
        logger.info(f"Starting backtest for symbols: {self.symbols} with capital {self.initial_capital}")
        # 1. Load models
        models = self.load_models()
        if not models:
            raise ValueError("No models loaded. Check models_dir and symbols.")

        # 2. Backtest each model on its symbol
        for symbol, name, model in tqdm(models, desc="Backtesting models"):
            df = get_data(symbol)
            df_feat = prepare_features(df)
            feats = get_feature_columns(df_feat)
            split_idx = int(len(df_feat) * 0.8)
            test_df = df_feat.iloc[split_idx:]

            # Perform backtest
            equity, sharpe, _ = backtest_model(
                df_feat, feats, model, test_df, self.signal_threshold
            )
            self.results.append((symbol, name, sharpe, equity))
            logger.info(f"Backtested {name} on {symbol}: Sharpe {sharpe:.2f}")

        # 3. For each symbol, select the model with the highest Sharpe ratio
        selected = []
        for sym in self.symbols:
            # gather results for this symbol
            sym_results = [res for res in self.results if res[0] == sym]
            if not sym_results:
                logger.warning(f"No models backtested for symbol {sym}")
                continue
            # pick the best by Sharpe
            best = max(sym_results, key=lambda x: x[2])
            logger.info(f"Selected best model {best[1]} for symbol {sym} with Sharpe {best[2]:.2f}")
            selected.append(best)

        # 4. Allocate capital equally and combine equity curves
        combined = pd.DataFrame({})
        if self.weight_by_sharpe:
            total_sharpe = sum(sharpe for _, _, sharpe, _ in selected)
            allocations = [self.initial_capital * (sharpe / total_sharpe) for _, _, sharpe, _ in selected]
        else:
            allocations = [self.initial_capital / len(selected)] * len(selected)
        for (symbol, name, sharpe, eq), alloc in zip(selected, allocations):
            scaled_eq = eq * alloc
            combined[f"{symbol}_{name}"] = scaled_eq
        combined['Total_Portfolio'] = combined.sum(axis=1)
        return combined

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+', required=True)
    parser.add_argument('--capital', type=float, default=10000)
    parser.add_argument('--models_dir', type=Path, default=Config.MODELS_DIR)
    parser.add_argument(
        '--weight_by_sharpe',
        action='store_true',
        help='Allocate initial capital proportionally based on each model\'s Sharpe ratio'
    )
    args = parser.parse_args()

    broker = TestBroker(
        symbols=args.symbols,
        capital=args.capital,
        models_dir=args.models_dir,
        weight_by_sharpe=args.weight_by_sharpe
    )
    portfolio_eq = broker.run()
    print(portfolio_eq.tail())

    # Dynamic performance summary
    import numpy as np

    total = portfolio_eq['Total_Portfolio']
    daily_returns = total.pct_change().dropna()

    start = total.index[0]
    end = total.index[-1]
    duration = end - start
    exposure = daily_returns.count() / daily_returns.size * 100

    annual_factor = 252  # trading days per year

    performance = {
        "Start": start,
        "End": end,
        "Duration": duration,
        "Exposure Time [%]": f"{exposure:.4f}",
        "Equity Final [$]": f"{total.iloc[-1]:.2f}",
        "Equity Peak [$]": f"{total.max():.2f}",
        "Return [%]": f"{(total.iloc[-1]/total.iloc[0]-1)*100:.4f}",
        "Return (Ann.) [%]": f"{((total.iloc[-1]/total.iloc[0]) ** (annual_factor / duration.days)-1)*100:.4f}",
        "Volatility (Ann.) [%]": f"{daily_returns.std() * np.sqrt(annual_factor) * 100:.4f}",
        "Sharpe Ratio": f"{(daily_returns.mean()/daily_returns.std()) * np.sqrt(annual_factor):.4f}",
        "Max. Drawdown [%]": f"{((total / total.cummax() - 1).min())*100:.4f}",
        "# Trades": len([c for c in portfolio_eq.columns if c != 'Total_Portfolio']),
    }

    print("\nPerformance Summary:")
    for k, v in performance.items():
        print(f"{k:30} {v}")

    # Save to CSV with UTC timestamp
    ts = now_utc().strftime('%Y%m%d_%H%M%S')
    portfolio_eq.to_csv(f"portfolio_equity_{ts}.csv")
