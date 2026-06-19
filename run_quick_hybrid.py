#!/usr/bin/env python
"""
Quick Hybrid Comparison — reduced API calls
============================================
Rules: full 50 stocks, 5 years (fast)
Council/Hybrid: top 10 stocks, 5 years (manageable API calls)
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
logger = logging.getLogger("quick_hybrid")

from data.market import MarketData
from data.fundamentals import FundamentalData
from factors.combined import CombinedRanking
from factors.council_driven import CouncilStockPicker
from factors.hybrid_picker import HybridStockPicker
from backtest.portfolio_engine import PortfolioBacktestEngine
from engine.volatility_targeting import VolatilityTargeting


def _get_last_trading_day_per_month(index):
    s = pd.Series(range(len(index)), index=index)
    return s.resample("ME").last().dropna().index.tolist()


TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "LLY",
    "AVGO", "TSLA", "WMT", "JPM", "V", "UNH", "MA", "XOM", "COST",
    "HD", "PG", "ABBV", "CRM", "NFLX", "MRK", "BAC", "AMD", "CVX",
    "ORCL", "KO", "TMO", "PEP", "LIN", "CSCO", "ACN", "WFC", "ADBE",
    "DHR", "ABT", "QCOM", "TXN", "PM", "COP", "NEE", "CMCSA", "INTC",
    "UNP", "INTU", "AMGN", "AMAT", "LOW", "CAT",
]


def main():
    print("=" * 70)
    print("QUICK HYBRID COMPARISON")
    print("=" * 70)

    INITIAL_CAPITAL = 100_000
    TOP_N = 10  # Reduced from 15 to cut API calls
    LOOKBACK_YEARS = 5
    END_DATE = datetime.now()
    START_DATE = END_DATE - timedelta(days=LOOKBACK_YEARS * 365)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    print(f"\n  Top N: {TOP_N} (reduced for speed)")
    print(f"  LLM: {'Yes (2 models)' if api_key else 'No'}")

    # Fetch data
    print("\n[1/3] Fetching data...")
    market_data = MarketData()
    prices = market_data.get_prices(
        tickers=TICKERS,
        start=START_DATE.strftime("%Y-%m-%d"),
        end=END_DATE.strftime("%Y-%m-%d"),
    )
    prices = prices.dropna(axis=1, how="all")
    print(f"  {prices.shape[0]} days x {prices.shape[1]} tickers")

    fundamentals_data = FundamentalData()
    fundamentals = fundamentals_data.get_fundamentals(tickers=prices.columns.tolist())
    print(f"  {len(fundamentals)} fundamentals")

    engine = PortfolioBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        commission_rate=0.001,
        slippage_rate=0.001,
    )
    rebalance_dates = _get_last_trading_day_per_month(prices.index)

    results = {}

    # --- RULES ONLY ---
    print("\n[2/3] Running RULES ONLY...")
    rule_signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    ranking = CombinedRanking(momentum_weight=0.6, value_weight=0.4, top_n=TOP_N)

    for rd in rebalance_dates:
        if rd not in prices.index:
            continue
        slc = prices.loc[:rd]
        if len(slc) < 252:
            continue
        portfolio = ranking.compute_portfolio(slc, fundamentals)
        if portfolio is not None and len(portfolio) > 0:
            for _, row in portfolio.iterrows():
                if row["ticker"] in rule_signals.columns:
                    rule_signals.loc[rd, row["ticker"]] = 1.0

    rule_signals = rule_signals.replace(0, np.nan).ffill().fillna(0)
    rule_result = engine.run(prices, rule_signals)
    results["Rules"] = rule_result["metrics"]
    rule_equity = rule_result["equity_curve"]
    print(f"  Annual: {results['Rules']['annual_return']:.2%} | Sharpe: {results['Rules']['sharpe_ratio']:.2f}")

    if api_key:
        # --- HYBRID ---
        print("\n[3/3] Running HYBRID (rules + council)...")
        hybrid = HybridStockPicker(
            council_picker=CouncilStockPicker(
                api_key=api_key,
                prefilter_top_n=TOP_N,
                final_top_n=TOP_N,
            ),
            momentum_weight=0.6,
            value_weight=0.4,
            top_n=TOP_N,
        )

        hybrid_signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        count = 0
        for rd in rebalance_dates:
            if rd not in prices.index:
                continue
            slc = prices.loc[:rd]
            if len(slc) < 252:
                continue
            try:
                portfolio = hybrid.compute_portfolio(prices=slc, fundamentals=fundamentals)
                if portfolio is not None and len(portfolio) > 0:
                    for _, row in portfolio.iterrows():
                        if row["ticker"] in hybrid_signals.columns:
                            hybrid_signals.loc[rd, row["ticker"]] = 1.0
                    count += 1
                    boosted = len(portfolio[portfolio["council_action"] == "BUY"])
                    cut = len(portfolio[portfolio["council_action"] == "SELL"])
                    print(f"    {rd.date()}: {len(portfolio)} stocks (+{boosted} -{cut})")
            except Exception as e:
                print(f"    {rd.date()}: ERROR {e}")

        hybrid_signals = hybrid_signals.replace(0, np.nan).ffill().fillna(0)
        hybrid_result = engine.run(prices, hybrid_signals)
        results["Hybrid"] = hybrid_result["metrics"]
        hybrid_equity = hybrid_result["equity_curve"]
        print(f"  Annual: {results['Hybrid']['annual_return']:.2%} | Sharpe: {results['Hybrid']['sharpe_ratio']:.2f}")
    else:
        results["Hybrid"] = results["Rules"]

    # --- COMPARISON ---
    print(f"\n{'='*70}")
    print(f"RESULTS COMPARISON (Top {TOP_N} stocks)")
    print(f"{'='*70}")

    headers = ["Metric", "Rules", "Hybrid"]
    print(f"  {headers[0]:<25s} {headers[1]:>15s} {headers[2]:>15s}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")

    for name, key in [
        ("Annual Return", "annual_return"),
        ("Sharpe Ratio", "sharpe_ratio"),
        ("Sortino Ratio", "sortino_ratio"),
        ("Max Drawdown", "max_drawdown"),
        ("Total Return", "total_return"),
        ("Trades", "num_trades"),
    ]:
        v1 = results["Rules"].get(key, 0)
        v2 = results["Hybrid"].get(key, 0)
        if key in ("annual_return", "max_drawdown", "total_return"):
            print(f"  {name:<25s} {v1:>14.2%} {v2:>14.2%}")
        elif key == "num_trades":
            print(f"  {name:<25s} {v1:>14d} {v2:>14d}")
        else:
            print(f"  {name:<25s} {v1:>14.2f} {v2:>14.2f}")

    # Benchmark
    bench = prices.iloc[:, 0]
    bench_ret = (bench.iloc[-1] / bench.iloc[252] - 1) if len(bench) > 252 else 0
    print(f"\n  Benchmark ({prices.columns[0]} buy-and-hold): {bench_ret:.2%}")

    # Save
    rule_equity.to_csv("output/quick_rules_equity.csv")
    if api_key:
        hybrid_equity.to_csv("output/quick_hybrid_equity.csv")
    print(f"\n  Saved to output/quick_*.csv")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
