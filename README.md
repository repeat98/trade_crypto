# Cryptocurrency Trading Pipeline

A comprehensive pipeline for cryptocurrency trading that includes data fetching, feature engineering, machine learning modeling, backtesting, and performance visualization.

## Features

- **Data Collection**: Fetches historical cryptocurrency data using yfinance
- **Data Caching**: Efficient data storage using SQLite and SQLAlchemy
- **Feature Engineering**: Technical indicators and advanced features using the `ta` library
- **Machine Learning**: Implements scikit-learn and XGBoost models for trading signals
- **Backtesting**: Comprehensive backtesting framework with transaction costs
- **Performance Analysis**: Detailed metrics calculation and equity curve visualization
- **Multi-Cryptocurrency Support**: Handles multiple cryptocurrencies including:
  - Bitcoin (BTC-USD)
  - Ethereum (ETH-USD)
  - And many other major cryptocurrencies

## Project Structure

```
.
├── main.py              # Main pipeline implementation
├── backtest.py          # Backtesting functionality
├── sim_trader.py        # Trading simulation
├── update_models.py     # Model updating utilities
├── symbols.txt          # List of supported cryptocurrencies
├── crypto_data.db       # SQLite database for data caching
├── models/             # Directory for saved models
├── plots/              # Directory for performance plots
├── metrics/            # Directory for performance metrics
└── cache/              # Cache directory for joblib
```

## Requirements

- Python 3.x
- pandas
- numpy
- sqlalchemy
- yfinance
- ta (Technical Analysis library)
- scikit-learn
- xgboost
- matplotlib
- joblib

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The main pipeline can be run using:

```bash
python main.py
```

This will:
1. Fetch and cache cryptocurrency data
2. Generate technical indicators and features
3. Train machine learning models
4. Perform backtesting
5. Generate performance metrics and plots

## Configuration

The system can be configured using environment variables:
- `DB_URL`: Database connection string
- `MODELS_DIR`: Directory for saved models
- `PLOTS_DIR`: Directory for performance plots
- `METRICS_DIR`: Directory for performance metrics
- `CACHE_REFRESH_DAYS`: Number of days before refreshing cached data

## Performance Metrics

The system calculates various performance metrics including:
- Returns
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- And more

## License

[Your chosen license]

## Contributing

[Your contribution guidelines] 