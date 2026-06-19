import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config
from data.market import MarketData
from backtest.engine import BacktestEngine
from backtest.results import print_backtest_results, save_results_csv

# Setup logging
config.LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_DIR / "backtest.log"),
    ],
)
logger = logging.getLogger(__name__)


def run_backtest(ticker: str = "AAPL", initial_capital: float = 100_000):
    """Run a single-ticker backtest."""
    print(f"\nRunning backtest for {ticker}...")

    # Fetch data
    market = MarketData()
    price_data = market.get_historical_prices(ticker, period="5y")

    if price_data.empty:
        print(f"No data available for {ticker}")
        return None

    print(f"Loaded {len(price_data)} days of price data")

    # Run backtest
    engine = BacktestEngine(initial_capital=initial_capital)
    results = engine.run(price_data, ticker)

    # Print results
    print_backtest_results(results, ticker)

    # Save results
    output_dir = config.DATA_DIR / "backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results_csv(results, str(output_dir / ticker))

    return results


def run_multi_backtest(tickers: list = None, initial_capital: float = 100_000):
    """Run backtests for multiple tickers."""
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

    all_results = {}
    for ticker in tickers:
        try:
            results = run_backtest(ticker, initial_capital / len(tickers))
            all_results[ticker] = results
        except Exception as e:
            logger.error(f"Backtest failed for {ticker}: {e}")

    # Summary
    print(f"\n{'='*50}")
    print("  PORTFOLIO SUMMARY")
    print(f"{'='*50}")

    passing = 0
    for ticker, results in all_results.items():
        if results and "error" not in results:
            sharpe = results["sharpe_ratio"]
            dd = results["max_drawdown_pct"]
            status = "PASS" if sharpe >= 1.0 and dd <= 20 else "REVIEW"
            if status == "PASS":
                passing += 1
            print(f"  {ticker:>6s}: Sharpe={sharpe:.2f}  MaxDD={dd:.1f}%  {status}")

    print(f"\n  {passing}/{len(all_results)} tickers pass criteria")
    print(f"{'='*50}")

    return all_results


if __name__ == "__main__":
    # Run single backtest first
    run_backtest("AAPL")
