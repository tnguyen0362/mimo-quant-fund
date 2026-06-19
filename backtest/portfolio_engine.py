# Multi-asset portfolio backtest engine
# - Handles multiple simultaneous positions
# - Tracks portfolio-level equity curve
# - Models transaction costs across positions
# - Supports walk-forward validation
# - Walk-forward optimization support

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PortfolioTrade:
    ticker: str
    date: pd.Timestamp
    side: str  # "buy" or "sell"
    shares: float
    price: float
    value: float
    reason: str = ""


@dataclass
class PortfolioPosition:
    ticker: str
    shares: float
    entry_date: pd.Timestamp
    entry_price: float
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def pnl(self) -> float:
        return self.shares * (self.current_price - self.entry_price)

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price


class PortfolioBacktestEngine:
    """
    Multi-asset portfolio backtest engine.

    Tracks:
    - Portfolio equity curve (cash + positions)
    - Individual position P&L
    - Transaction costs
    - Trade log
    """

    def __init__(self, initial_capital: float = 100_000.0,
                 commission_rate: float = 0.001,  # 0.1% per trade
                 slippage_rate: float = 0.001):    # 0.1% slippage
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        self.cash = initial_capital
        self.positions: dict[str, PortfolioPosition] = {}
        self.trades: list[PortfolioTrade] = []
        self.equity_curve: list[dict] = []

    def run(self, prices: pd.DataFrame, signals: pd.DataFrame,
            position_sizes: Optional[pd.DataFrame] = None) -> dict:
        """
        Run portfolio backtest.

        Args:
            prices: DataFrame with dates as index, tickers as columns (adj close)
            signals: DataFrame with same structure, values in [-1, 1]
                     Positive = long, Negative = short (or reduce)
            position_sizes: Optional target weights per stock per date
                           If None, equal-weight among positive signals

        Returns:
            dict with equity_curve, trades, metrics
        """
        # Ensure aligned dates
        common_dates = prices.index.intersection(signals.index)
        prices = prices.loc[common_dates]
        signals = signals.loc[common_dates]

        # Initialize
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

        for date in common_dates:
            # Update position prices
            for ticker in list(self.positions.keys()):
                if ticker in prices.columns and not pd.isna(prices.loc[date, ticker]):
                    self.positions[ticker].current_price = prices.loc[date, ticker]

            # Generate target portfolio from signals
            if position_sizes is not None and date in position_sizes.index:
                target_portfolio = self._weights_to_portfolio(
                    position_sizes.loc[date], prices.loc[date]
                )
            else:
                target_portfolio = self._signals_to_portfolio(
                    signals.loc[date], prices.loc[date]
                )

            # Execute trades to match target
            self._rebalance(target_portfolio, date, prices.loc[date])

            # Record equity
            portfolio_value = self.cash + sum(
                pos.market_value for pos in self.positions.values()
            )
            self.equity_curve.append({
                "date": date,
                "equity": portfolio_value,
                "cash": self.cash,
                "positions_value": portfolio_value - self.cash,
                "num_positions": len(self.positions),
            })

        # Compute metrics
        equity_df = pd.DataFrame(self.equity_curve).set_index("date")
        metrics = self._compute_metrics(equity_df)

        return {
            "equity_curve": equity_df,
            "trades": self.trades,
            "metrics": metrics,
            "final_positions": dict(self.positions),
        }

    def run_walk_forward(self, prices: pd.DataFrame, signals: pd.DataFrame,
                         train_window: int = 252, test_window: int = 63,
                         position_sizes: Optional[pd.DataFrame] = None) -> dict:
        """
        Run walk-forward validation.

        Splits data into rolling train/test windows. Backtest only uses signals
        from the test window (train window is available for the signal generator
        to have warmed up).

        Args:
            prices: DataFrame with dates as index, tickers as columns
            signals: DataFrame with same structure
            train_window: Number of trading days for training (warmup)
            test_window: Number of trading days per test fold
            position_sizes: Optional target weights

        Returns:
            dict with combined equity curve, per-fold results, and metrics
        """
        common_dates = prices.index.intersection(signals.index).sort_values()
        n_dates = len(common_dates)

        if n_dates < train_window + test_window:
            # Not enough data for walk-forward; just run normally
            return self.run(prices, signals, position_sizes)

        all_fold_results = []
        all_trades = []
        combined_equity = []

        # Walk-forward windows
        fold_start = 0
        fold_num = 0
        while fold_start + train_window + test_window <= n_dates:
            test_start_idx = fold_start + train_window
            test_end_idx = test_start_idx + test_window
            test_dates = common_dates[test_start_idx:test_end_idx]

            # Slice prices and signals for the test window
            fold_prices = prices.loc[test_dates]
            fold_signals = signals.loc[test_dates]
            fold_sizes = position_sizes.loc[test_dates] if (
                position_sizes is not None and test_dates[0] in position_sizes.index
            ) else None

            # Save and reset engine state for this fold
            saved_cash = self.cash
            saved_positions = dict(self.positions)
            saved_trades = list(self.trades)
            saved_equity = list(self.equity_curve)

            self.cash = saved_cash if fold_num == 0 else self.initial_capital
            if fold_num > 0:
                self.positions = {}
            self.trades = []
            self.equity_curve = []

            fold_result = self.run(fold_prices, fold_signals, fold_sizes)

            # Accumulate
            all_fold_results.append(fold_result)
            all_trades.extend(fold_result["trades"])

            if fold_num == 0:
                combined_equity.append(fold_result["equity_curve"])
            else:
                combined_equity.append(fold_result["equity_curve"])

            fold_num += 1
            fold_start += test_window

        # Combine results
        if combined_equity:
            full_equity_df = pd.concat(combined_equity)
            full_equity_df = full_equity_df[~full_equity_df.index.duplicated(keep='first')]
        else:
            full_equity_df = pd.DataFrame()

        metrics = self._compute_metrics(full_equity_df) if not full_equity_df.empty else {}

        return {
            "equity_curve": full_equity_df,
            "trades": all_trades,
            "metrics": metrics,
            "fold_results": all_fold_results,
            "num_folds": fold_num,
        }

    def _signals_to_portfolio(self, signals: pd.Series, prices: pd.Series,
                              position_sizes: Optional[pd.DataFrame] = None) -> dict[str, float]:
        """
        Convert signals to target portfolio weights.

        Positive signals = long, negative = flat (no shorting for retail).
        Equal-weight among stocks with positive signals.
        """
        # Filter to stocks with positive signals and valid prices
        valid_prices = prices.dropna()
        long_candidates = signals[signals > 0].index.intersection(valid_prices.index)
        long_candidates = signals.loc[long_candidates]

        if len(long_candidates) == 0:
            return {}

        # Equal weight
        weight = 1.0 / len(long_candidates)
        return {ticker: weight for ticker in long_candidates.index}

    def _weights_to_portfolio(self, weights: pd.Series, prices: pd.Series) -> dict[str, float]:
        """Convert target weights to portfolio dict."""
        # Filter positive weights only (no shorting) and valid prices
        valid_prices = prices.dropna()
        positive = weights[weights > 0].index.intersection(valid_prices.index)
        positive = weights.loc[positive]

        # Normalize to sum to 1
        if positive.sum() > 0:
            positive = positive / positive.sum()

        return positive.to_dict()

    def _rebalance(self, target: dict[str, float], date: pd.Timestamp,
                   prices: pd.Series):
        """Execute trades to match target portfolio."""
        portfolio_value = self.cash + sum(
            pos.market_value for pos in self.positions.values()
        )

        # Sell positions not in target
        for ticker in list(self.positions.keys()):
            if ticker not in target:
                self._sell_position(ticker, date, prices.get(ticker, 0))

        # Adjust existing positions
        for ticker, target_weight in target.items():
            current_pos = self.positions.get(ticker)
            target_value = portfolio_value * target_weight
            current_value = current_pos.market_value if current_pos else 0

            price = prices.get(ticker, 0)
            if pd.isna(price) or price <= 0:
                continue

            diff_value = target_value - current_value

            if abs(diff_value) < portfolio_value * 0.01:  # Skip tiny rebalances
                continue

            if diff_value > 0:
                # Buy more
                self._buy_position(ticker, diff_value, date, price)
            elif diff_value < 0 and current_pos:
                # Sell some
                shares_to_sell = min(
                    abs(diff_value) / price,
                    current_pos.shares
                )
                self._sell_position(ticker, date, price, shares_to_sell)

    def _buy_position(self, ticker: str, value: float, date: pd.Timestamp,
                      price: float):
        """Buy a position."""
        # Apply slippage
        exec_price = price * (1 + self.slippage_rate)

        # Calculate shares
        shares = value / exec_price

        # Calculate commission
        commission = value * self.commission_rate

        # Check if we have enough cash
        total_cost = value + commission
        if total_cost > self.cash:
            # Reduce to what we can afford
            available = self.cash - commission
            if available <= 0:
                return
            shares = available / exec_price
            total_cost = shares * exec_price + commission

        if shares <= 0 or total_cost > self.cash:
            return

        # Execute
        self.cash -= total_cost

        if ticker in self.positions:
            # Add to existing position
            pos = self.positions[ticker]
            total_shares = pos.shares + shares
            avg_price = (pos.shares * pos.entry_price + shares * exec_price) / total_shares
            pos.shares = total_shares
            pos.entry_price = avg_price
        else:
            self.positions[ticker] = PortfolioPosition(
                ticker=ticker,
                shares=shares,
                entry_date=date,
                entry_price=exec_price,
                current_price=price,
            )

        self.trades.append(PortfolioTrade(
            ticker=ticker, date=date, side="buy",
            shares=shares, price=exec_price,
            value=shares * exec_price,
        ))

    def _sell_position(self, ticker: str, date: pd.Timestamp,
                       price: float, shares: Optional[float] = None):
        """Sell a position (or partial)."""
        if ticker not in self.positions:
            return

        pos = self.positions[ticker]

        if shares is None:
            shares = pos.shares

        if pd.isna(price) or price <= 0:
            # Can't sell at invalid price; force sell at last known price
            price = pos.current_price
            if pd.isna(price) or price <= 0:
                return

        # Apply slippage
        exec_price = price * (1 - self.slippage_rate)

        # Calculate proceeds
        proceeds = shares * exec_price
        commission = proceeds * self.commission_rate

        # Execute
        self.cash += proceeds - commission
        pos.shares -= shares

        self.trades.append(PortfolioTrade(
            ticker=ticker, date=date, side="sell",
            shares=shares, price=exec_price,
            value=proceeds,
        ))

        # Remove if fully closed
        if pos.shares <= 1e-10:
            del self.positions[ticker]

    def _compute_metrics(self, equity_df: pd.DataFrame) -> dict:
        """Compute portfolio performance metrics."""
        if equity_df.empty or len(equity_df) < 2:
            return {
                "total_return": 0.0,
                "annual_return": 0.0,
                "annual_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0,
                "num_trades": 0,
                "avg_positions": 0.0,
                "final_equity": self.initial_capital,
            }

        returns = equity_df["equity"].pct_change().dropna()

        # Annualization
        trading_days = 252

        # Basic metrics
        total_return = (equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0]) - 1
        n_days = len(returns)
        if n_days > 0:
            annual_return = (1 + total_return) ** (trading_days / n_days) - 1
        else:
            annual_return = 0.0
        annual_vol = returns.std() * np.sqrt(trading_days)

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(trading_days) if len(downside_returns) > 0 else 0
        sortino = annual_return / downside_vol if downside_vol > 0 else 0

        # Maximum drawdown
        rolling_max = equity_df["equity"].cummax()
        drawdowns = (equity_df["equity"] - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        num_trades = len(self.trades)
        num_positions = equity_df["num_positions"].mean()

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "num_trades": num_trades,
            "avg_positions": num_positions,
            "final_equity": equity_df["equity"].iloc[-1],
        }
