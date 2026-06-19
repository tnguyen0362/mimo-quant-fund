#!/usr/bin/env python
"""
LLM-Driven Quantitative Hedge Fund Backtest
============================================
Three-Factor System: Momentum + Value + MiMo 2.5 Sentiment

The LLM (MiMo 2.5) analyzes news and earnings to generate sentiment signals.
Combined with rule-based momentum and value factors.
Code handles data, risk management, and execution.

Usage:
    python run_llm_backtest.py
    
Environment:
    OPENROUTER_API_KEY=your_key_here  (for MiMo 2.5 access)
    Or runs in fallback mode (keyword sentiment) if no key.
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
from factors.llm_combined import LLMCombinedRanking
from factors.combined import CombinedRanking  # Fallback (no LLM)
from backtest.portfolio_engine import PortfolioBacktestEngine
from engine.volatility_targeting import VolatilityTargeting
from engine.risk_monitor import RiskMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("llm_backtest")


def get_news_data(tickers: list[str]) -> dict[str, list[str]]:
    """
    Fetch recent news headlines for each stock.
    
    In production, this would use a news API (NewsAPI, Finnhub, etc.)
    For now, we generate synthetic headlines based on price performance
    to simulate what the LLM would analyze.
    
    TODO: Replace with real news API integration
    """
    import yfinance as yf
    
    news_data = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            # Get recent news from yfinance
            news = stock.news if hasattr(stock, 'news') else []
            
            headlines = []
            if news:
                for item in news[:10]:
                    if isinstance(item, dict) and 'title' in item:
                        headlines.append(item['title'])
            
            if not headlines:
                # Generate synthetic headline based on recent performance
                hist = stock.history(period="1mo")
                if len(hist) > 0:
                    ret = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1)
                    if ret > 0.05:
                        headlines.append(f"{ticker} shows strong momentum with {ret:.1%} monthly gain")
                    elif ret < -0.05:
                        headlines.append(f"{ticker} faces selling pressure with {ret:.1%} monthly decline")
                    else:
                        headlines.append(f"{ticker} trades flat with {ret:.1%} monthly change")
            
            news_data[ticker] = headlines if headlines else [f"{ticker} - no recent news available"]
            
        except Exception as e:
            news_data[ticker] = [f"{ticker} - unable to fetch news"]
    
    return news_data


def _get_last_trading_day_per_month(prices: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Return the last available trading day in each calendar month."""
    s = pd.Series(range(len(prices)), index=prices)
    return s.resample("ME").last().dropna().index.tolist()


def run_llm_backtest():
    """Run LLM-driven quantitative backtest."""
    
    print("=" * 70)
    print("LLM-DRIVEN QUANTITATIVE HEDGE FUND BACKTEST")
    print("Three-Factor: Momentum + Value + MiMo 2.5 Sentiment")
    print("=" * 70)
    
    # Configuration
    INITIAL_CAPITAL = 100_000
    TOP_N_STOCKS = 15
    TARGET_VOL = 0.15
    LOOKBACK_YEARS = 5
    END_DATE = datetime.now()
    START_DATE = END_DATE - timedelta(days=LOOKBACK_YEARS * 365)
    
    # Check for API key
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if api_key:
        print(f"\n[OK] MiMo 2.5 API key detected -- using LLM sentiment")
    else:
        print(f"\n[WARN] No OPENROUTER_API_KEY set -- using keyword fallback")
        print(f"       Set OPENROUTER_API_KEY to enable MiMo 2.5 sentiment")
    
    print(f"\nConfiguration:")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"  Universe: Top {TOP_N_STOCKS} stocks by combined rank")
    print(f"  Factor Weights: 40% Momentum + 30% Value + 30% LLM Sentiment")
    print(f"  Target Volatility: {TARGET_VOL:.0%}")
    print(f"  Period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")

    # ------------------------------------------------------------------ #
    # Step 1: Get Universe
    # ------------------------------------------------------------------ #
    print("\n[1/8] Fetching universe...")
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

    tickers = all_tickers[:50]
    print(f"  Using top {len(tickers)} stocks for backtest")

    # ------------------------------------------------------------------ #
    # Step 2: Fetch Market Data
    # ------------------------------------------------------------------ #
    print("\n[2/8] Fetching market data...")
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
        cache_dir = Path(__file__).parent / "data" / "cache"
        cache_files = sorted(cache_dir.glob("prices_*.parquet"))
        if cache_files:
            latest_cache = cache_files[-1]
            print(f"  Loading from cache: {latest_cache.name}")
            prices = pd.read_parquet(latest_cache)
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
    print("\n[3/8] Fetching fundamentals...")
    try:
        fundamentals_data = FundamentalData()
        fundamentals = fundamentals_data.get_fundamentals(tickers=prices.columns.tolist())
    except Exception as exc:
        logger.error("Failed to fetch fundamentals: %s", exc)
        print(f"  WARNING: Fundamental data fetch failed ({exc})")
        print("  Creating synthetic fundamentals from price data...")
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
    # Step 4: Fetch News for LLM Analysis
    # ------------------------------------------------------------------ #
    print("\n[4/8] Fetching news headlines for LLM analysis...")
    try:
        news_data = get_news_data(prices.columns.tolist())
        avg_headlines = np.mean([len(v) for v in news_data.values()])
        print(f"  News fetched for {len(news_data)} stocks (avg {avg_headlines:.1f} headlines each)")
    except Exception as exc:
        logger.error("Failed to fetch news: %s", exc)
        print(f"  WARNING: News fetch failed ({exc})")
        print("  Using synthetic headlines for all stocks")
        news_data = {ticker: [f"{ticker} - no recent news available"] for ticker in prices.columns}

    # ------------------------------------------------------------------ #
    # Step 5: Initialize Three-Factor Ranking
    # ------------------------------------------------------------------ #
    print("\n[5/8] Initializing three-factor ranking engine...")
    
    # Try LLM version first, fall back to 2-factor if needed
    use_llm = False
    try:
        three_factor = LLMCombinedRanking(
            momentum_weight=0.40,
            value_weight=0.30,
            llm_weight=0.30,
            top_n=TOP_N_STOCKS,
            api_key=api_key if api_key else None,
        )
        
        # Test a single analysis to verify it works
        test_result = three_factor.llm_factor.llm.analyze_stock(
            "AAPL", ["Apple reports record quarterly revenue"]
        )
        print(f"  LLM engine initialized (test: sentiment={test_result.sentiment:.2f}, source={test_result.source})")
        
        use_llm = True
        
    except Exception as e:
        logger.warning("LLM engine init failed: %s", e)
        print(f"  [WARN] LLM engine failed: {e}")
        print(f"  Falling back to 2-factor (momentum + value)")
        three_factor = CombinedRanking(momentum_weight=0.5, value_weight=0.5, top_n=TOP_N_STOCKS)
        use_llm = False

    # ------------------------------------------------------------------ #
    # Step 6: Generate Trading Signals (Monthly Rebalance)
    # ------------------------------------------------------------------ #
    print("\n[6/8] Generating trading signals with LLM sentiment...")
    
    signal_matrix = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    # Monthly rebalancing using last trading day of each month
    rebalance_dates = _get_last_trading_day_per_month(prices.index)

    rebal_count = 0
    for rebal_date in rebalance_dates:
        if rebal_date not in prices.index:
            continue
        
        # Only rebalance after we have enough history for momentum (252 days)
        current_prices_slice = prices.loc[:rebal_date]
        if len(current_prices_slice) < 252:
            continue
        
        try:
            if use_llm:
                # Three-factor: Momentum + Value + LLM
                portfolio = three_factor.compute_portfolio(
                    current_prices_slice,
                    fundamentals,
                    news_data,
                )
            else:
                # Two-factor: Momentum + Value only
                portfolio = three_factor.compute_portfolio(
                    current_prices_slice,
                    fundamentals,
                )
            
            if len(portfolio) > 0:
                for _, row in portfolio.iterrows():
                    ticker = row["ticker"]
                    if ticker in signal_matrix.columns:
                        signal_matrix.loc[rebal_date, ticker] = row["weight"]
                rebal_count += 1
                
        except Exception as e:
            logger.warning("Rebalance failed at %s: %s", rebal_date, e)
            print(f"  Warning: Rebalance failed at {rebal_date}: {e}")
    
    # Forward-fill signals (hold between rebalances)
    signal_matrix = signal_matrix.replace(0, np.nan).ffill().fillna(0)
    
    print(f"  Rebalances executed: {rebal_count}")
    print(f"  Active positions: {(signal_matrix.iloc[-1] > 0).sum()}")

    # ------------------------------------------------------------------ #
    # Step 7: Run Portfolio Backtest
    # ------------------------------------------------------------------ #
    print("\n[7/8] Running portfolio backtest...")
    
    engine = PortfolioBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        commission_rate=0.001,
        slippage_rate=0.001,
    )
    
    results = engine.run(prices=prices, signals=signal_matrix)
    
    equity_curve = results["equity_curve"]
    metrics = results["metrics"]
    trades = results["trades"]
    
    print(f"\n  Backtest Results:")
    print(f"  {'=' * 60}")
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
    print(f"  {'=' * 60}")

    # ------------------------------------------------------------------ #
    # Step 8: Volatility Targeting
    # ------------------------------------------------------------------ #
    print("\n[8/8] Applying volatility targeting overlay...")
    
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
    vol_realized = (
        vol_targeted_returns.std() * np.sqrt(252)
        if vol_targeted_returns.std() > 0
        else 0.0
    )

    print(f"\n  Volatility-Targeted Results:")
    print(f"  {'=' * 60}")
    print(f"  Annual Return:       {vol_ann_return:>12.2%}  (vs {metrics['annual_return']:.2%} raw)")
    print(f"  Sharpe Ratio:        {vol_sharpe:>12.2f}  (vs {metrics['sharpe_ratio']:.2f} raw)")
    print(f"  Target Volatility:   {TARGET_VOL:>12.0%}")
    print(f"  Realized Volatility: {vol_realized:>12.2%}")
    print(f"  {'=' * 60}")

    # Risk monitoring
    risk_monitor = RiskMonitor()
    risk_status = risk_monitor.check_portfolio(
        equity_curve["equity"],
        target_vol=TARGET_VOL,
    )

    print(f"\n  Risk Status:")
    print(f"  {'=' * 60}")
    print(f"  Exposure Multiplier: {risk_status['exposure_multiplier']:>11.0%}")
    print(f"  Trading Halted:      {str(risk_status['halted']):>12}")
    print(f"  Active Alerts:       {len(risk_status['alerts']):>12d}")
    print(f"  {'=' * 60}")

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
    print(f"\n{'=' * 70}")
    print(f"LLM-DRIVEN QUANT FUND -- COMPLETE")
    print(f"{'=' * 70}")
    print(f"\n  Annual Return:  {metrics['annual_return']:.2%}")
    print(f"  Sharpe Ratio:   {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:   {metrics['max_drawdown']:.2%}")
    print(f"  LLM Sentiment:  {'Active (MiMo 2.5)' if use_llm else 'Fallback (keywords)'}")
    print(f"  Factor Blend:   40% Momentum + 30% Value + 30% LLM Sentiment")

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

    equity_curve.to_csv(output_dir / "llm_equity_curve.csv")

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
    trades_df.to_csv(output_dir / "llm_trades.csv", index=False)

    # Save vol-targeted equity curve as well
    vt_equity_df = pd.DataFrame({"equity": vol_targeted_equity})
    vt_equity_df.to_csv(output_dir / "llm_equity_curve_vol_targeted.csv")

    print(f"\n  Results saved to {output_dir}/")
    print(f"    - llm_equity_curve.csv")
    print(f"    - llm_equity_curve_vol_targeted.csv")
    print(f"    - llm_trades.csv")

    return results


if __name__ == "__main__":
    try:
        results = run_llm_backtest()
        if results is None:
            print("\nBacktest failed -- no results produced.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nBacktest interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Unhandled exception during backtest")
        print(f"\nFATAL: Backtest crashed with unhandled exception: {exc}")
        sys.exit(1)
