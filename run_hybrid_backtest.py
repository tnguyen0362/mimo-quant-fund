#!/usr/bin/env python
"""
Hybrid Council + Rules Backtest
================================
Compares three strategies:
1. Rule-based (momentum+value, no LLM)
2. Council-only (LLM picks stocks)
3. Hybrid (rules for base, council for conviction)

The hybrid is the main event: rules always produce a portfolio,
council boosts winners and cuts losers.
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
logger = logging.getLogger("hybrid_backtest")

from data.universe import UniverseManager
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


def run_rule_based(prices, fundamentals, top_n=15):
    """Run rule-based strategy (no LLM)."""
    print("\n  [Rules] Computing momentum+value signals...")
    ranking = CombinedRanking(momentum_weight=0.6, value_weight=0.4, top_n=top_n)
    
    signal_matrix = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    rebalance_dates = _get_last_trading_day_per_month(prices.index)
    
    for rd in rebalance_dates:
        if rd not in prices.index:
            continue
        slc = prices.loc[:rd]
        if len(slc) < 252:
            continue
        
        portfolio = ranking.compute_portfolio(slc, fundamentals)
        if portfolio is not None and len(portfolio) > 0:
            for _, row in portfolio.iterrows():
                ticker = row["ticker"]
                if ticker in signal_matrix.columns:
                    signal_matrix.loc[rd, ticker] = 1.0
    
    signal_matrix = signal_matrix.replace(0, np.nan).ffill().fillna(0)
    return signal_matrix


def run_council_only(prices, fundamentals, api_key, top_n=15):
    """Run council-only strategy (LLM picks everything)."""
    print("\n  [Council] Running LLM council picks...")
    picker = CouncilStockPicker(
        api_key=api_key,
        prefilter_top_n=20,
        final_top_n=top_n,
        momentum_weight=0.35,
        value_weight=0.25,
        council_weight=0.40,
    )
    
    signal_matrix = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    rebalance_dates = _get_last_trading_day_per_month(prices.index)
    pick_count = 0
    
    for rd in rebalance_dates:
        if rd not in prices.index:
            continue
        slc = prices.loc[:rd]
        if len(slc) < 252:
            continue
        
        try:
            picks = picker.pick_stocks(prices=slc, fundamentals=fundamentals, rebalance_date=rd)
            if picks:
                for p in picks:
                    if p.ticker in signal_matrix.columns:
                        signal_matrix.loc[rd, p.ticker] = 1.0
                pick_count += 1
                print(f"    {rd.date()}: {len(picks)} stocks picked")
        except Exception as e:
            print(f"    {rd.date()}: ERROR - {e}")
    
    signal_matrix = signal_matrix.replace(0, np.nan).ffill().fillna(0)
    print(f"  [Council] {pick_count} rebalances completed")
    return signal_matrix


def run_hybrid(prices, fundamentals, api_key, top_n=15):
    """Run hybrid strategy (rules + council conviction)."""
    print("\n  [Hybrid] Running hybrid rules+council...")
    hybrid = HybridStockPicker(
        council_picker=CouncilStockPicker(
            api_key=api_key,
            prefilter_top_n=top_n,
            final_top_n=top_n,
        ),
        momentum_weight=0.6,
        value_weight=0.4,
        top_n=top_n,
    )
    
    signal_matrix = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    rebalance_dates = _get_last_trading_day_per_month(prices.index)
    pick_count = 0
    
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
                    ticker = row["ticker"]
                    if ticker in signal_matrix.columns:
                        signal_matrix.loc[rd, ticker] = 1.0
                pick_count += 1
                
                # Show council adjustments
                boosted = portfolio[portfolio["council_action"] == "BUY"]
                cut = portfolio[portfolio["council_action"] == "SELL"]
                print(f"    {rd.date()}: {len(portfolio)} stocks | "
                      f"+{len(boosted)} boosted | -{len(cut)} cut")
        except Exception as e:
            print(f"    {rd.date()}: ERROR - {e}")
    
    signal_matrix = signal_matrix.replace(0, np.nan).ffill().fillna(0)
    print(f"  [Hybrid] {pick_count} rebalances completed")
    return signal_matrix


def run_backtest(prices, signal_matrix, initial_capital=100_000):
    """Run portfolio backtest on signals."""
    engine = PortfolioBacktestEngine(
        initial_capital=initial_capital,
        commission_rate=0.001,
        slippage_rate=0.001,
    )
    result = engine.run(prices, signal_matrix)
    return result["metrics"], result["equity_curve"]


def print_comparison(results):
    """Print side-by-side comparison."""
    print(f"\n{'='*80}")
    print(f"STRATEGY COMPARISON")
    print(f"{'='*80}")
    
    headers = ["Metric", "Rules Only", "Council Only", "Hybrid"]
    print(f"  {headers[0]:<25s} {headers[1]:>15s} {headers[2]:>15s} {headers[3]:>15s}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15}")
    
    metrics = [
        ("Annual Return", "annual_return", "%"),
        ("Sharpe Ratio", "sharpe_ratio", ""),
        ("Sortino Ratio", "sortino_ratio", ""),
        ("Max Drawdown", "max_drawdown", "%"),
        ("Calmar Ratio", "calmar_ratio", ""),
        ("Total Return", "total_return", "%"),
        ("Annual Volatility", "annual_volatility", "%"),
        ("Total Trades", "num_trades", ""),
    ]
    
    for name, key, fmt in metrics:
        vals = []
        for label in ["Rules", "Council", "Hybrid"]:
            m = results[label]["metrics"]
            v = m.get(key, 0)
            if fmt == "%":
                vals.append(f"{v:>14.2%}")
            elif fmt == "":
                vals.append(f"{v:>14.2f}")
            else:
                vals.append(f"{v:>14d}")
        print(f"  {name:<25s} {vals[0]} {vals[1]} {vals[2]}")
    
    print(f"  {'='*25} {'='*15} {'='*15} {'='*15}")


def main():
    print("=" * 80)
    print("HYBRID COUNCIL + RULES BACKTEST")
    print("Rules = floor (always diversified), Council = ceiling (boost/cut)")
    print("=" * 80)
    
    # Config
    INITIAL_CAPITAL = 100_000
    LOOKBACK_YEARS = 5
    TARGET_VOL = 0.15
    TOP_N = 15
    END_DATE = datetime.now()
    START_DATE = END_DATE - timedelta(days=LOOKBACK_YEARS * 365)
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    print(f"\n  Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"  Period:  {START_DATE.date()} to {END_DATE.date()}")
    print(f"  Top N:   {TOP_N}")
    print(f"  LLM:     {'Council (2 models)' if api_key else 'RULES ONLY (no API key)'}")
    
    # Step 1: Universe
    print("\n[1/4] Fetching universe...")
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
    print("\n[2/4] Fetching market data...")
    market_data = MarketData()
    prices = market_data.get_prices(
        tickers=tickers,
        start=START_DATE.strftime("%Y-%m-%d"),
        end=END_DATE.strftime("%Y-%m-%d"),
    )
    prices = prices.dropna(axis=1, how="all")
    print(f"  Matrix: {prices.shape[0]} days x {prices.shape[1]} tickers")
    
    # Step 3: Fundamentals
    print("\n[3/4] Fetching fundamentals...")
    fundamentals_data = FundamentalData()
    fundamentals = fundamentals_data.get_fundamentals(tickers=prices.columns.tolist())
    print(f"  Fundamentals: {len(fundamentals)} stocks")
    
    # Step 4: Run all three strategies
    print("\n[4/4] Running strategies...")
    
    results = {}
    
    # Strategy 1: Rules only
    rule_signals = run_rule_based(prices, fundamentals, TOP_N)
    rule_metrics, rule_equity = run_backtest(prices, rule_signals, INITIAL_CAPITAL)
    results["Rules"] = {"metrics": rule_metrics, "equity": rule_equity}
    print(f"\n  Rules: {rule_metrics['annual_return']:.2%} annual, Sharpe {rule_metrics['sharpe_ratio']:.2f}")
    
    # Strategy 2: Council only (if API key available)
    if api_key:
        council_signals = run_council_only(prices, fundamentals, api_key, TOP_N)
        council_metrics, council_equity = run_backtest(prices, council_signals, INITIAL_CAPITAL)
        results["Council"] = {"metrics": council_metrics, "equity": council_equity}
        print(f"  Council: {council_metrics['annual_return']:.2%} annual, Sharpe {council_metrics['sharpe_ratio']:.2f}")
        
        # Strategy 3: Hybrid
        hybrid_signals = run_hybrid(prices, fundamentals, api_key, TOP_N)
        hybrid_metrics, hybrid_equity = run_backtest(prices, hybrid_signals, INITIAL_CAPITAL)
        results["Hybrid"] = {"metrics": hybrid_metrics, "equity": hybrid_equity}
        print(f"  Hybrid: {hybrid_metrics['annual_return']:.2%} annual, Sharpe {hybrid_metrics['sharpe_ratio']:.2f}")
    else:
        print("\n  Skipping Council + Hybrid (no API key)")
        results["Council"] = results["Rules"]
        results["Hybrid"] = results["Rules"]
    
    # Print comparison
    print_comparison(results)
    
    # Vol targeting
    print(f"\n  Volatility targeting ({TARGET_VOL:.0%} target):")
    vt = VolatilityTargeting(target_vol=TARGET_VOL)
    for label in ["Rules", "Council", "Hybrid"]:
        eq = results[label]["equity"]
        returns = eq["equity"].pct_change().dropna()
        vol_adj = vt.apply_to_portfolio(returns)
        vol_ret = (1 + vol_adj).prod() - 1
        raw_ret = results[label]["metrics"]["total_return"]
        print(f"    {label:10s}: {raw_ret:.2%} raw -> {vol_ret:.2%} vol-targeted")
    
    # Benchmark
    bench = prices.iloc[:, 0]
    bench_ret = (bench.iloc[-1] / bench.iloc[252] - 1) if len(bench) > 252 else 0
    print(f"\n  Benchmark: {prices.columns[0]} buy-and-hold = {bench_ret:.2%}")
    
    # Save
    for label in ["Rules", "Council", "Hybrid"]:
        results[label]["equity"].to_csv(f"output/hybrid_{label.lower()}_equity.csv")
    
    print(f"\n  Results saved to output/hybrid_*.csv")
    
    print(f"\n{'='*80}")
    print(f"HYBRID BACKTEST COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
