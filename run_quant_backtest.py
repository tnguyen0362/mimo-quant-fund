#!/usr/bin/env python
"""
Quantitative Hedge Fund Backtest
================================
Full integration test combining:
- Multi-asset data layer (universe, market data, fundamentals)
- Factor engines (cross-sectional momentum + value)
- Combined ranking (50/50 momentum + value)
- Portfolio backtest engine (multi-asset)
- Volatility targeting overlay
- Risk monitoring (drawdown control, position stop-loss)

Usage:
    python run_quant_backtest.py
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data.universe import UniverseManager
from data.market import MarketData
from data.fundamentals import FundamentalData
from factors.momentum import MomentumFactor
from factors.value import ValueFactor
from factors.combined import CombinedRanking
from backtest.portfolio_engine import PortfolioBacktestEngine
from engine.volatility_targeting import VolatilityTargeting, RollingVolThreshold
from engine.risk_monitor import RiskMonitor, RiskManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("quant_backtest")


def _get_last_trading_day_per_month(prices: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Return the last available trading day in each calendar month."""
    s = pd.Series(range(len(prices)), index=prices)
    return s.resample("ME").last().dropna().index.tolist()


def run_backtest():
    """Run full quantitative backtest pipeline."""

    print("=" * 60)
    print("QUANTITATIVE HEDGE FUND BACKTEST")
    print("=" * 60)

    # Configuration
    INITIAL_CAPITAL = 100_000
    TOP_N_STOCKS = 15
    TARGET_VOL = 0.15
    LOOKBACK_YEARS = 5
    END_DATE = datetime.now()
    START_DATE = END_DATE - timedelta(days=LOOKBACK_YEARS * 365)

    print(f"\nConfiguration:")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"  Universe: Top {TOP_N_STOCKS} stocks")
    print(f"  Target Volatility: {TARGET_VOL:.0%}")
    print(f"  Period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")

    # ------------------------------------------------------------------ #
    # Step 1: Get Universe
    # ------------------------------------------------------------------ #
    print("\n[1/7] Fetching universe...")
    try:
        universe = UniverseManager()
        all_tickers = universe.get_sp500_tickers()
        print(f"  Universe size: {len(all_tickers)} stocks")
    except Exception as exc:
        logger.error("Failed to fetch universe: %s", exc)
        print(f"  ERROR: Could not fetch S&P 500 universe ({exc})")
        print("  Using fallback universe of 50 stocks")
        all_tickers = [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "LLY",
            "AVGO", "TSLA", "WMT", "JPM", "V", "UNH", "MA", "XOM", "COST",
            "HD", "PG", "ABBV", "CRM", "NFLX", "MRK", "BAC", "AMD", "CVX",
            "ORCL", "KO", "TMO", "PEP", "LIN", "CSCO", "ACN", "WFC", "ADBE",
            "DHR", "ABT", "QCOM", "TXN", "PM", "COP", "NEE", "CMCSA", "INTC",
            "UNP", "INTU", "AMGN", "AMAT", "LOW", "CAT",
        ]

    # Use a subset for faster backtesting (top 50 by default)
    tickers = all_tickers[:50]
    print(f"  Using top {len(tickers)} stocks for backtest")

    # ------------------------------------------------------------------ #
    # Step 2: Fetch Market Data
    # ------------------------------------------------------------------ #
    print("\n[2/7] Fetching market data...")
    try:
        market_data = MarketData()
        prices = market_data.get_prices(
            tickers=tickers,
            start=START_DATE.strftime("%Y-%m-%d"),
            end=END_DATE.strftime("%Y-%m-%d"),
        )
    except Exception as exc:
        logger.error("Failed to fetch market data: %s", exc)
        print(f"  ERROR: Market data fetch failed ({exc})")
        print("  Attempting to load cached data...")
        # Try loading any cached parquet files
        cache_dir = Path(__file__).parent / "data" / "cache"
        cache_files = sorted(cache_dir.glob("prices_*.parquet"))
        if cache_files:
            latest_cache = cache_files[-1]
            print(f"  Loading from cache: {latest_cache.name}")
            prices = pd.read_parquet(latest_cache)
            # Filter to requested tickers
            available = [t for t in tickers if t in prices.columns]
            prices = prices[available]
        else:
            print("  FATAL: No cached data available. Cannot proceed.")
            return None

    if prices.empty:
        print("  FATAL: No price data retrieved. Cannot proceed.")
        return None

    # Drop tickers that are entirely NaN after alignment
    prices = prices.dropna(axis=1, how="all")

    print(f"  Price matrix: {prices.shape[0]} days x {prices.shape[1]} tickers")
    print(f"  Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")

    # ------------------------------------------------------------------ #
    # Step 3: Fetch Fundamentals
    # ------------------------------------------------------------------ #
    print("\n[3/7] Fetching fundamentals...")
    try:
        fundamentals_data = FundamentalData()
        fundamentals = fundamentals_data.get_fundamentals(tickers=prices.columns.tolist())
    except Exception as exc:
        logger.error("Failed to fetch fundamentals: %s", exc)
        print(f"  WARNING: Fundamental data fetch failed ({exc})")
        print("  Creating synthetic fundamentals from price data...")
        # Create synthetic fundamentals so the backtest can still run
        last_prices = prices.iloc[-1]
        fundamentals = pd.DataFrame(
            {
                "pe_ratio": np.random.uniform(10, 35, size=len(prices.columns)),
                "pb_ratio": np.random.uniform(1, 8, size=len(prices.columns)),
                "ps_ratio": np.random.uniform(1, 10, size=len(prices.columns)),
                "ev_ebitda": np.random.uniform(8, 25, size=len(prices.columns)),
                "dividend_yield": np.random.uniform(0.0, 0.04, size=len(prices.columns)),
                "market_cap": last_prices.values * np.random.uniform(1e9, 3e12, size=len(prices.columns)),
            },
            index=prices.columns,
        )
        fundamentals.index.name = "ticker"

    print(f"  Fundamental data: {len(fundamentals)} stocks")
    pe_available = fundamentals["pe_ratio"].notna().sum() if "pe_ratio" in fundamentals.columns else 0
    pb_available = fundamentals["pb_ratio"].notna().sum() if "pb_ratio" in fundamentals.columns else 0
    print(f"  P/E data available: {pe_available}")
    print(f"  P/B data available: {pb_available}")

    # ------------------------------------------------------------------ #
    # Step 4: Compute Factor Signals
    # ------------------------------------------------------------------ #
    print("\n[4/7] Computing factor signals...")

    # Momentum factor
    momentum = MomentumFactor(lookback_days=252, skip_days=21)
    mom_signals = momentum.compute_signal(prices)

    last_mom = mom_signals.iloc[-1].dropna()
    mean_mom = last_mom.mean() if len(last_mom) > 0 else 0.0
    print(f"  Momentum signal: {mom_signals.shape}")
    print(f"  Mean momentum: {mean_mom:.2%}")

    # Value factor
    value = ValueFactor()
    val_signals = value.compute_signal(fundamentals)

    print(f"  Value signal: {len(val_signals)} stocks")
    mean_val = val_signals["value_score"].mean() if "value_score" in val_signals.columns and len(val_signals) > 0 else 0.0
    print(f"  Mean value score: {mean_val:.3f}")

    # Combined ranking
    combined = CombinedRanking(momentum_weight=0.5, value_weight=0.5, top_n=TOP_N_STOCKS)
    portfolio = combined.compute_portfolio(prices, fundamentals)

    print(f"  Combined portfolio: {len(portfolio)} stocks")
    if len(portfolio) > 0:
        print(f"  Top holdings: {portfolio.head(5)['ticker'].tolist()}")
        print(f"  Weight per holding: {portfolio['weight'].iloc[0]:.1%}")

    # ------------------------------------------------------------------ #
    # Step 5: Generate Trading Signals
    # ------------------------------------------------------------------ #
    print("\n[5/7] Generating trading signals...")

    # Create signal matrix: stocks with positive combined rank get positive signal
    # Rebalance monthly
    signal_matrix = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # Find the last actual trading day of each month (not calendar month-end)
    rebalance_dates = _get_last_trading_day_per_month(prices.index)

    rebalance_count = 0
    for rebal_date in rebalance_dates:
        if rebal_date not in prices.index:
            continue

        # Get prices up to this date for momentum calculation
        # (momentum needs full lookback, so we use the latest available)
        current_prices_slice = prices.loc[:rebal_date]
        # Skip if not enough history for the momentum lookback (252 days)
        if len(current_prices_slice) < 252:
            continue

        current_portfolio = combined.compute_portfolio(
            current_prices_slice,
            fundamentals,
        )

        if len(current_portfolio) > 0:
            for _, row in current_portfolio.iterrows():
                ticker = row["ticker"]
                if ticker in signal_matrix.columns:
                    signal_matrix.loc[rebal_date, ticker] = row["weight"]
            rebalance_count += 1

    # Forward-fill signals (hold between rebalances)
    signal_matrix = signal_matrix.replace(0, np.nan).ffill().fillna(0)

    print(f"  Signal matrix: {signal_matrix.shape}")
    print(f"  Rebalances executed: {rebalance_count}")
    last_row = signal_matrix.iloc[-1]
    print(f"  Active positions: {(last_row > 0).sum()}")

    # ------------------------------------------------------------------ #
    # Step 6: Run Portfolio Backtest
    # ------------------------------------------------------------------ #
    print("\n[6/7] Running portfolio backtest...")

    engine = PortfolioBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        commission_rate=0.001,  # 0.1% per trade
        slippage_rate=0.001,    # 0.1% slippage
    )

    results = engine.run(
        prices=prices,
        signals=signal_matrix,
    )

    equity_curve = results["equity_curve"]
    metrics = results["metrics"]
    trades = results["trades"]

    print(f"\n  Backtest Results:")
    print(f"  {'=' * 50}")
    print(f"  Initial Capital:     ${INITIAL_CAPITAL:>12,.2f}")
    print(f"  Final Equity:        ${metrics['final_equity']:>12,.2f}")
    print(f"  Total Return:        {metrics['total_return']:>12.2%}")
    print(f"  Annual Return:       {metrics['annual_return']:>12.2%}")
    print(f"  Annual Volatility:   {metrics['annual_volatility']:>12.2%}")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>12.2f}")
    print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>12.2f}")
    print(f"  Max Drawdown:        {metrics['max_drawdown']:>12.2%}")
    print(f"  Calmar Ratio:        {metrics['calmar_ratio']:>12.2f}")
    print(f"  Total Trades:        {metrics['num_trades']:>12d}")
    print(f"  Avg Positions:       {metrics['avg_positions']:>12.1f}")
    print(f"  {'=' * 50}")

    # ------------------------------------------------------------------ #
    # Step 7: Apply Volatility Targeting
    # ------------------------------------------------------------------ #
    print("\n[7/7] Applying volatility targeting overlay...")

    vol_targeting = VolatilityTargeting(
        target_vol=TARGET_VOL,
        vol_lookback=60,
        vol_method="ewma",
        ewma_halflife=20,
        max_leverage=2.0,
    )

    portfolio_returns = equity_curve["equity"].pct_change().dropna()

    vol_targeted_returns = vol_targeting.apply_to_portfolio(portfolio_returns)

    # Compute vol-targeted metrics
    vol_targeted_equity = (1 + vol_targeted_returns).cumprod() * INITIAL_CAPITAL
    vol_sharpe = (
        vol_targeted_returns.mean() / vol_targeted_returns.std() * np.sqrt(252)
        if vol_targeted_returns.std() > 0
        else 0.0
    )
    vol_ann_return = (
        (1 + vol_targeted_returns.mean()) ** 252 - 1
        if not np.isnan(vol_targeted_returns.mean())
        else 0.0
    )
    vol_realized = vol_targeted_returns.std() * np.sqrt(252) if vol_targeted_returns.std() > 0 else 0.0

    print(f"\n  Volatility-Targeted Results:")
    print(f"  {'=' * 50}")
    print(f"  Annual Return:       {vol_ann_return:>12.2%}  (vs {metrics['annual_return']:.2%} raw)")
    print(f"  Sharpe Ratio:        {vol_sharpe:>12.2f}  (vs {metrics['sharpe_ratio']:.2f} raw)")
    print(f"  Target Volatility:   {TARGET_VOL:>12.0%}")
    print(f"  Realized Volatility: {vol_realized:>12.2%}")
    print(f"  {'=' * 50}")

    # Risk monitoring
    risk_monitor = RiskMonitor()
    risk_status = risk_monitor.check_portfolio(
        equity_curve["equity"],
        target_vol=TARGET_VOL,
    )

    print(f"\n  Risk Status:")
    print(f"  {'=' * 50}")
    print(f"  Exposure Multiplier: {risk_status['exposure_multiplier']:>11.0%}")
    print(f"  Trading Halted:      {str(risk_status['halted']):>12}")
    print(f"  Active Alerts:       {len(risk_status['alerts']):>12d}")
    print(f"  {'=' * 50}")

    # Print any risk alerts
    if risk_status["alerts"]:
        print(f"\n  Risk Alerts Detail:")
        for alert in risk_status["alerts"]:
            print(f"    [{alert.level.upper():>9}] {alert.category}: {alert.message}")
            if alert.action_taken:
                print(f"             Action: {alert.action_taken}")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 60}")
    print(f"BACKTEST COMPLETE")
    print(f"{'=' * 60}")
    print(f"\nKey Metrics:")
    print(f"  - Annual Return: {metrics['annual_return']:.2%}")
    print(f"  - Sharpe Ratio:  {metrics['sharpe_ratio']:.2f}")
    print(f"  - Max Drawdown:  {metrics['max_drawdown']:.2%}")
    print(f"  - Trades Executed: {metrics['num_trades']}")

    # Compare to first ticker as a rough benchmark proxy
    benchmark_ticker = prices.columns[0]
    benchmark_return = (
        prices[benchmark_ticker].iloc[-1] / prices[benchmark_ticker].iloc[0] - 1
    )
    print(f"\nBenchmark Comparison:")
    print(f"  - {benchmark_ticker} buy-and-hold: {benchmark_return:.2%}")
    print(f"  - Our strategy:       {metrics['total_return']:.2%}")
    excess = metrics["total_return"] - benchmark_return
    print(f"  - Excess return:      {excess:+.2%}")

    # Save results
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    equity_curve.to_csv(output_dir / "equity_curve.csv")

    # Convert trades (dataclass list) to DataFrame for CSV export
    trades_df = pd.DataFrame(
        [
            {
                "ticker": t.ticker,
                "date": t.date,
                "side": t.side,
                "shares": t.shares,
                "price": t.price,
                "value": t.value,
                "reason": t.reason,
            }
            for t in trades
        ]
    )
    trades_df.to_csv(output_dir / "trades.csv", index=False)

    # Save vol-targeted equity curve as well
    vt_equity_df = pd.DataFrame({"equity": vol_targeted_equity})
    vt_equity_df.to_csv(output_dir / "equity_curve_vol_targeted.csv")

    print(f"\nResults saved to {output_dir}/")
    print(f"  - equity_curve.csv")
    print(f"  - equity_curve_vol_targeted.csv")
    print(f"  - trades.csv")

    return results


if __name__ == "__main__":
    try:
        results = run_backtest()
        if results is None:
            print("\nBacktest failed — no results produced.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nBacktest interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Unhandled exception during backtest")
        print(f"\nFATAL: Backtest crashed with unhandled exception: {exc}")
        sys.exit(1)
