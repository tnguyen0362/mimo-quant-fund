#!/usr/bin/env python
"""
Validation: Compare 2-factor vs 3-factor with IDENTICAL parameters.
Runs both backtests on the exact same data with the same factor weights.
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

from data.universe import UniverseManager
from data.market import MarketData
from data.fundamentals import FundamentalData
from factors.momentum import MomentumFactor
from factors.value import ValueFactor
from factors.combined import CombinedRanking
from backtest.portfolio_engine import PortfolioBacktestEngine
from engine.volatility_targeting import VolatilityTargeting


def _get_last_trading_day_per_month(index):
    s = pd.Series(range(len(index)), index=index)
    return s.resample("ME").last().dropna().index.tolist()


def run_comparison():
    print("=" * 70)
    print("VALIDATION: 2-FACTOR vs 3-FACTOR COMPARISON")
    print("Same data, same universe, same dates, same everything")
    print("=" * 70)

    # Identical config for both
    INITIAL_CAPITAL = 100_000
    TOP_N_STOCKS = 15
    LOOKBACK_YEARS = 5
    MOMENTUM_LOOKBACK = 252
    MOMENTUM_SKIP = 21
    END_DATE = datetime.now()
    START_DATE = END_DATE - timedelta(days=LOOKBACK_YEARS * 365)

    print(f"\n  Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"  Period:  {START_DATE.date()} to {END_DATE.date()}")
    print(f"  Top N:   {TOP_N_STOCKS}")

    # Step 1: Fetch universe
    print("\n[1] Fetching universe...")
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

    # Step 2: Fetch market data (ONCE)
    print("\n[2] Fetching market data...")
    market_data = MarketData()
    prices = market_data.get_prices(
        tickers=tickers,
        start=START_DATE.strftime("%Y-%m-%d"),
        end=END_DATE.strftime("%Y-%m-%d"),
    )
    prices = prices.dropna(axis=1, how="all")
    print(f"  Matrix: {prices.shape[0]} days x {prices.shape[1]} tickers")
    print(f"  Range:  {prices.index[0].date()} to {prices.index[-1].date()}")

    # Step 3: Fetch fundamentals (ONCE)
    print("\n[3] Fetching fundamentals...")
    fundamentals_data = FundamentalData()
    fundamentals = fundamentals_data.get_fundamentals(tickers=prices.columns.tolist())

    # Step 4: Compute factors (ONCE)
    print("\n[4] Computing factors...")
    momentum = MomentumFactor(lookback_days=MOMENTUM_LOOKBACK, skip_days=MOMENTUM_SKIP)
    mom_signals = momentum.compute_signal(prices)

    value = ValueFactor()
    val_signals = value.compute_signal(fundamentals)
    print(f"  Momentum signals: {mom_signals.shape}")
    print(f"  Value signals: {len(val_signals)} stocks")

    # Step 5: Build signal matrices for BOTH strategies
    print("\n[5] Building signal matrices...")

    # Monthly rebalance dates
    rebalance_dates = _get_last_trading_day_per_month(prices.index)

    # --- STRATEGY A: 50/50 momentum + value (the 2-factor version) ---
    print("  Strategy A: 50/50 momentum + value")
    sig_a = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    combined_a = CombinedRanking(momentum_weight=0.5, value_weight=0.5, top_n=TOP_N_STOCKS)

    for rd in rebalance_dates:
        if rd not in prices.index:
            continue
        slc = prices.loc[:rd]
        if len(slc) < 252:
            continue
        try:
            port = combined_a.compute_portfolio(slc, fundamentals)
            if len(port) > 0:
                for _, row in port.iterrows():
                    t = row["ticker"]
                    if t in sig_a.columns:
                        sig_a.loc[rd, t] = 1.0
        except Exception as e:
            print(f"    [ERR] Strategy A at {rd.date()}: {e}")

    # --- STRATEGY B: 40/30/30 with council factor = 0 (simulates council returning neutral) ---
    print("  Strategy B: 40/30/30 (council=0 for all, so effective ~57/43)")
    sig_b = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    combined_b = CombinedRanking(momentum_weight=0.40, value_weight=0.30, top_n=TOP_N_STOCKS)

    for rd in rebalance_dates:
        if rd not in prices.index:
            continue
        slc = prices.loc[:rd]
        if len(slc) < 252:
            continue
        try:
            # Same as A but different weights
            port = combined_b.compute_portfolio(slc, fundamentals)
            if len(port) > 0:
                for _, row in port.iterrows():
                    t = row["ticker"]
                    if t in sig_b.columns:
                        sig_b.loc[rd, t] = 1.0
        except Exception as e:
            print(f"    [ERR] Strategy B at {rd.date()}: {e}")

    # Forward-fill signals (hold between rebalances) — CRITICAL
    sig_a = sig_a.replace(0, np.nan).ffill().fillna(0)
    sig_b = sig_b.replace(0, np.nan).ffill().fillna(0)

    trades_a = int(sig_a.sum().sum())
    trades_b = int(sig_b.sum().sum())
    print(f"  Signal A total: {trades_a} position-days")
    print(f"  Signal B total: {trades_b} position-days")

    # Step 6: Run backtests (ONCE each, identical engine config)
    print("\n[6] Running backtests...")
    engine = PortfolioBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        commission_rate=0.001,
        slippage_rate=0.001,
    )

    print("  Running Strategy A...")
    result_a = engine.run(prices, sig_a)

    print("  Running Strategy B...")
    result_b = engine.run(prices, sig_b)

    # Step 7: Compare
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    metrics = [
        ("Total Return", "total_return", "%"),
        ("Annual Return", "annual_return", "%"),
        ("Annual Volatility", "annual_volatility", "%"),
        ("Sharpe Ratio", "sharpe_ratio", ""),
        ("Sortino Ratio", "sortino_ratio", ""),
        ("Max Drawdown", "max_drawdown", "%"),
        ("Calmar Ratio", "calmar_ratio", ""),
        ("Total Trades", "total_trades", ""),
    ]

    print(f"\n  {'Metric':<25} {'A: 50/50':>12} {'B: 40/30/30':>12} {'Delta':>12}")
    print("  " + "-" * 61)

    for label, key, fmt in metrics:
        va = result_a["metrics"].get(key, 0)
        vb = result_b["metrics"].get(key, 0)
        delta = vb - va

        if fmt == "%":
            sa = f"{va:.2%}"
            sb = f"{vb:.2%}"
            sd = f"{delta:+.2%}"
        else:
            sa = f"{va:.2f}" if isinstance(va, float) else str(va)
            sb = f"{vb:.2f}" if isinstance(vb, float) else str(vb)
            sd = f"{delta:+.2f}" if isinstance(delta, float) else str(delta)

        print(f"  {label:<25} {sa:>12} {sb:>12} {sd:>12}")

    # Also run the same comparison with 60/40 weights
    print("\n\n  --- Additional test: 60/40 momentum+value ---")
    sig_c = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    combined_c = CombinedRanking(momentum_weight=0.6, value_weight=0.4, top_n=TOP_N_STOCKS)
    for rd in rebalance_dates:
        if rd not in prices.index:
            continue
        slc = prices.loc[:rd]
        if len(slc) < 252:
            continue
        try:
            port = combined_c.compute_portfolio(slc, fundamentals)
            if len(port) > 0:
                for _, row in port.iterrows():
                    t = row["ticker"]
                    if t in sig_c.columns:
                        sig_c.loc[rd, t] = 1.0
        except Exception:
            pass
    sig_c = sig_c.replace(0, np.nan).ffill().fillna(0)
    result_c = engine.run(prices, sig_c)
    mc = result_c["metrics"]
    print(f"  60/40: Return={mc['total_return']:.2%}, Sharpe={mc['sharpe_ratio']:.2f}, MaxDD={mc['max_drawdown']:.2%}")

    # Vol targeting comparison
    print("\n  --- With volatility targeting (15% target) ---")
    vt = VolatilityTargeting(target_vol=0.15)
    for name, result, sig_matrix in [("A", result_a, sig_a), ("B", result_b, sig_b)]:
        eq = result.get("equity_curve")
        if eq is not None and len(eq) > 0:
            # Apply vol targeting to equity curve returns
            returns = eq["equity"].pct_change().dropna()
            vol_adj = vt.apply_to_portfolio(returns)
            vol_ret = (1 + vol_adj).prod() - 1
            raw_ret = result["metrics"]["total_return"]
            print(f"  {name} vol-targeted: Return={vol_ret:.2%} (vs raw {raw_ret:.2%})")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Check if the difference is meaningful
    ret_diff = result_b["metrics"]["total_return"] - result_a["metrics"]["total_return"]
    sharpe_diff = result_b["metrics"]["sharpe_ratio"] - result_a["metrics"]["sharpe_ratio"]

    if abs(ret_diff) < 0.02:
        print("  The difference is SMALL (< 2%). Factor weights barely matter.")
        print("  The 18.45% vs 8.87% gap was likely due to different data/caching.")
    elif abs(sharpe_diff) < 0.1:
        print("  Returns differ but risk-adjusted performance is SIMILAR.")
        print("  The difference may be due to concentration or luck.")
    else:
        print(f"  Meaningful difference: {ret_diff:+.2%} return, {sharpe_diff:+.2f} Sharpe.")
        print("  This suggests factor weights DO matter for this universe/period.")

    print("\n  IMPORTANT: Neither result is validated out-of-sample.")
    print("  Both are in-sample backtests on the same 5-year bull market window.")


if __name__ == "__main__":
    run_comparison()
