#!/usr/bin/env python
"""
Council-Driven Stock Picker Backtest
=====================================
The council ANALYZES real financial data and PICKS stocks.
Momentum + value are secondary ranking factors within council picks.

Flow:
1. Universe → 50 stocks
2. Pre-filter → top 20 by momentum+value
3. Council → analyzes each with real data, votes BUY/HOLD/SELL
4. Filter → only BUY stocks with majority agreement
5. Final rank → composite of council conviction + momentum + value
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("council_backtest")

from data.universe import UniverseManager
from data.market import MarketData
from data.fundamentals import FundamentalData
from factors.council_driven import CouncilStockPicker
from backtest.portfolio_engine import PortfolioBacktestEngine
from engine.volatility_targeting import VolatilityTargeting
from engine.risk_monitor import RiskMonitor


def _get_last_trading_day_per_month(index):
    s = pd.Series(range(len(index)), index=index)
    return s.resample("ME").last().dropna().index.tolist()


def run_council_backtest():
    print("=" * 70)
    print("COUNCIL-DRIVEN STOCK PICKER BACKTEST")
    print("LLM picks stocks, momentum+value rank within picks")
    print("=" * 70)

    # Config
    INITIAL_CAPITAL = 100_000
    FINAL_TOP_N = 15
    LOOKBACK_YEARS = 5
    TARGET_VOL = 0.15
    END_DATE = datetime.now()
    START_DATE = END_DATE - timedelta(days=LOOKBACK_YEARS * 365)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    print(f"\n  Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"  Period:  {START_DATE.date()} to {END_DATE.date()}")
    print(f"  Top N:   {FINAL_TOP_N}")
    print(f"  LLM:     {'Council (4 models)' if api_key else 'NO API KEY - will fail'}")

    if not api_key:
        print("\n  ERROR: No OPENROUTER_API_KEY set. Council needs API access.")
        print("  Set it in .env or environment variable.")
        return

    # Step 1: Universe
    print("\n[1/6] Fetching universe...")
    try:
        universe = UniverseManager()
        all_tickers = universe.get_sp500_tickers()
    except Exception:
        all_tickers = [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "LLY",
            "AVGO", "TSLA", "WMT", "JPM", "V", "UNH", "MA", "XOM", "COST",
            "HD", "PG", "ABBV", "CRM", "NFLX", "MRK", "BAC", "AMD", "CVX",
            "ORCL", "KO", "TMO", "PEP", "LIN", "CSCO", "ACN", "WFC", "ADBE",
            "DHR", "ABT", "QCOM", "TXN", "PM", "COP", "NEE", "CMCSA", "INTC",
            "UNP", "INTU", "AMGN", "AMAT", "LOW", "CAT",
        ]
    tickers = all_tickers[:50]
    print(f"  Universe: {len(tickers)} stocks")

    # Step 2: Market data
    print("\n[2/6] Fetching market data...")
    market_data = MarketData()
    prices = market_data.get_prices(
        tickers=tickers,
        start=START_DATE.strftime("%Y-%m-%d"),
        end=END_DATE.strftime("%Y-%m-%d"),
    )
    prices = prices.dropna(axis=1, how="all")
    print(f"  Matrix: {prices.shape[0]} days x {prices.shape[1]} tickers")

    # Step 3: Fundamentals
    print("\n[3/6] Fetching fundamentals...")
    fundamentals_data = FundamentalData()
    fundamentals = fundamentals_data.get_fundamentals(tickers=prices.columns.tolist())
    print(f"  Fundamentals: {len(fundamentals)} stocks")

    # Step 4: Initialize council picker
    print("\n[4/6] Initializing council stock picker...")
    picker = CouncilStockPicker(
        api_key=api_key,
        prefilter_top_n=20,   # Top 20 by momentum+value → council analyzes these
        final_top_n=FINAL_TOP_N,
        momentum_weight=0.35,
        value_weight=0.25,
        council_weight=0.40,  # Council has most weight
    )
    print("  Council initialized with 4 reasoning models")

    # Step 5: Generate signals via council
    print("\n[5/6] Running council stock picks (monthly rebalance)...")
    print("  This will make ~80 API calls (20 stocks x 4 models x 4 months)")

    signal_matrix = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    rebalance_dates = _get_last_trading_day_per_month(prices.index)

    pick_log = []
    rebal_count = 0

    for rd in rebalance_dates:
        if rd not in prices.index:
            continue
        slc = prices.loc[:rd]
        if len(slc) < 252:
            continue

        print(f"\n  Rebalancing {rd.date()}...")

        try:
            picks = picker.pick_stocks(
                prices=slc,
                fundamentals=fundamentals,
                rebalance_date=rd,
            )

            if picks:
                print(f"    Council picked {len(picks)} stocks:")
                for p in picks:
                    # Sanitize reasoning for Windows console (remove non-ASCII)
                    safe_reasoning = p.reasoning.encode("ascii", "replace").decode("ascii")[:60]
                    print(f"      {p.ticker:6s} | {p.action:4s} | sent={p.sentiment:+.3f} | "
                              f"conv={p.conviction:.3f} | score={p.composite_score:.3f} | "
                              f"votes={p.num_votes} | {safe_reasoning}")
                    # Set signal
                    if p.ticker in signal_matrix.columns:
                        signal_matrix.loc[rd, p.ticker] = 1.0
                    pick_log.append({
                        "date": rd, "ticker": p.ticker, "action": p.action,
                        "sentiment": p.sentiment, "conviction": p.conviction,
                        "composite_score": p.composite_score,
                        "num_votes": p.num_votes, "reasoning": p.reasoning,
                    })
                rebal_count += 1
            else:
                print("    No stocks picked by council")

        except Exception as e:
            print(f"    ERROR: {e}")
            logger.exception("Council pick failed")

    # Forward-fill signals (hold between rebalances)
    signal_matrix = signal_matrix.replace(0, np.nan).ffill().fillna(0)

    print(f"\n  Rebalances: {rebal_count}")
    print(f"  Active positions at end: {(signal_matrix.iloc[-1] > 0).sum()}")

    # Save pick log
    if pick_log:
        pick_df = pd.DataFrame(pick_log)
        pick_df.to_csv("output/council_picks.csv", index=False)
        print(f"  Pick log saved to output/council_picks.csv")

    # Step 6: Run backtest
    print("\n[6/6] Running portfolio backtest...")
    engine = PortfolioBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        commission_rate=0.001,
        slippage_rate=0.001,
    )
    result = engine.run(prices, signal_matrix)
    m = result["metrics"]

    print(f"\n  Results:")
    print(f"  {'=' * 50}")
    print(f"  Initial Capital:     ${INITIAL_CAPITAL:>12,.2f}")
    print(f"  Final Equity:        ${m['final_equity']:>12,.2f}")
    print(f"  Total Return:        {m['total_return']:>12.2%}")
    print(f"  Annual Return:       {m['annual_return']:>12.2%}")
    print(f"  Annual Volatility:   {m['annual_volatility']:>12.2%}")
    print(f"  Sharpe Ratio:        {m['sharpe_ratio']:>12.2f}")
    print(f"  Sortino Ratio:       {m['sortino_ratio']:>12.2f}")
    print(f"  Max Drawdown:        {m['max_drawdown']:>12.2%}")
    print(f"  Calmar Ratio:        {m['calmar_ratio']:>12.2f}")
    print(f"  Total Trades:        {m['num_trades']:>12d}")
    print(f"  {'=' * 50}")

    # Vol targeting
    print("\n  Applying volatility targeting...")
    vt = VolatilityTargeting(target_vol=TARGET_VOL)
    eq = result["equity_curve"]
    returns = eq["equity"].pct_change().dropna()
    vol_adj = vt.apply_to_portfolio(returns)
    vol_ret = (1 + vol_adj).prod() - 1
    print(f"  Vol-targeted return: {vol_ret:.2%} (vs raw {m['total_return']:.2%})")

    # Benchmark
    print("\n  Benchmark comparison:")
    bench = prices.iloc[:, 0]
    bench_ret = (bench.iloc[-1] / bench.iloc[252] - 1) if len(bench) > 252 else 0
    print(f"    {prices.columns[0]} buy-and-hold: {bench_ret:.2%}")
    print(f"    Our strategy:       {m['total_return']:.2%}")
    print(f"    Excess return:      {m['total_return'] - bench_ret:+.2%}")

    # Save
    result["equity_curve"].to_csv("output/council_equity.csv")
    print(f"\n  Results saved to output/council_equity.csv")

    print(f"\n{'=' * 70}")
    print(f"COUNCIL-DRIVEN BACKTEST COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_council_backtest()
