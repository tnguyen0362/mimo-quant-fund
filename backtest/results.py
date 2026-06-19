import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def print_backtest_results(results: dict, ticker: str = ""):
    """Pretty-print backtest results."""
    if "error" in results:
        print(f"Backtest error: {results['error']}")
        return

    header = f"{'='*50}"
    print(f"\n{header}")
    print(f"  BACKTEST RESULTS {f'- {ticker}' if ticker else ''}")
    print(header)

    print(f"  Initial Capital:    ${results['initial_capital']:>12,.2f}")
    print(f"  Final Equity:       ${results['final_equity']:>12,.2f}")
    print(f"  Total Return:       {results['total_return_pct']:>11.2f}%")
    print(f"  Sharpe Ratio:       {results['sharpe_ratio']:>11.2f}")
    print(f"  Max Drawdown:       {results['max_drawdown_pct']:>11.2f}%")
    print(f"  Total Trades:       {results['total_trades']:>11d}")
    print(f"  Win Rate:           {results['win_rate_pct']:>11.2f}%")
    print(f"  Avg Win:            ${results['avg_win']:>12,.2f}")
    print(f"  Avg Loss:           ${results['avg_loss']:>12,.2f}")
    print(f"  Profit Factor:      {results['profit_factor']:>11.2f}")

    # Pass/fail assessment
    print(f"\n  {'='*50}")
    sharpe_ok = results["sharpe_ratio"] >= 1.0
    dd_ok = results["max_drawdown"] <= 0.20
    print(f"  Sharpe > 1.0:       {'PASS' if sharpe_ok else 'FAIL'} ({results['sharpe_ratio']:.2f})")
    print(f"  Max DD < 20%:       {'PASS' if dd_ok else 'FAIL'} ({results['max_drawdown_pct']:.2f}%)")
    print(f"  OVERALL:            {'PASS' if sharpe_ok and dd_ok else 'NEEDS REVIEW'}")
    print(header)


def save_results_csv(results: dict, filepath: str):
    """Save equity curve and trades to CSV."""
    if "equity_curve" in results:
        eq = results["equity_curve"]
        eq.to_csv(f"{filepath}_equity.csv")
        logger.info(f"Saved equity curve to {filepath}_equity.csv")

    if "trades" in results:
        trades_df = pd.DataFrame(results["trades"])
        trades_df.to_csv(f"{filepath}_trades.csv", index=False)
        logger.info(f"Saved trades to {filepath}_trades.csv")
