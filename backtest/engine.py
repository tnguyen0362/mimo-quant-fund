import pandas as pd
import numpy as np
import logging
from datetime import datetime

from config.settings import config
from engine.signals import SignalGenerator
from engine.position_sizer import PositionSizer
from engine.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Simple event-driven backtester."""

    def __init__(self, initial_capital: float = 100_000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.signal_gen = SignalGenerator()
        self.sizer = PositionSizer()
        self.risk_mgr = RiskManager()
        self.risk_mgr.peak_portfolio_value = initial_capital

    def run(self, price_data: pd.DataFrame, ticker: str) -> dict:
        """Run backtest on historical price data for a single ticker."""
        if price_data.empty or len(price_data) < 50:
            return {"error": "Insufficient data"}

        # Calculate technical indicators
        df = self.signal_gen.technical.calculate_all_indicators(price_data)
        df = self.signal_gen.technical.generate_signals(df)

        # Drop NaN rows from indicator warmup
        df = df.dropna(subset=["rsi", "macd", "sma_50"])

        for date, row in df.iterrows():
            current_price = row["close"]
            technical_score = row.get("technical_score", 0)

            # Generate signal
            combined_score = technical_score * config.TECHNICAL_WEIGHT
            if combined_score > config.SIGNAL_THRESHOLD_BUY:
                signal = "BUY"
            elif combined_score < config.SIGNAL_THRESHOLD_SELL:
                signal = "SELL"
            else:
                signal = "HOLD"

            # Execute signal
            if signal == "BUY" and ticker not in self.positions:
                # Risk check
                can_trade, reasons = self.risk_mgr.can_trade(self.capital)
                if can_trade:
                    sizing = self.sizer.calculate_position_size(
                        self.capital, current_price
                    )
                    if sizing["shares"] > 0:
                        cost = sizing["shares"] * current_price
                        self.capital -= cost
                        self.positions[ticker] = {
                            "shares": sizing["shares"],
                            "entry_price": current_price,
                            "entry_date": date,
                        }
                        self.trades.append({
                            "date": date,
                            "ticker": ticker,
                            "action": "BUY",
                            "price": current_price,
                            "shares": sizing["shares"],
                            "value": cost,
                        })

            elif signal == "SELL" and ticker in self.positions:
                pos = self.positions.pop(ticker)
                proceeds = pos["shares"] * current_price
                pnl = proceeds - (pos["shares"] * pos["entry_price"])
                self.capital += proceeds
                self.trades.append({
                    "date": date,
                    "ticker": ticker,
                    "action": "SELL",
                    "price": current_price,
                    "shares": pos["shares"],
                    "value": proceeds,
                    "pnl": pnl,
                })
                self.risk_mgr.update_portfolio_state(
                    self.capital + self._position_value(df.iloc[-1]["close"]),
                    pnl,
                )

            # Track equity
            equity = self.capital + self._position_value(current_price)
            self.equity_curve.append({"date": date, "equity": equity})

        return self._calculate_results()

    def _position_value(self, current_price: float) -> float:
        total = 0
        for ticker, pos in self.positions.items():
            total += pos["shares"] * current_price
        return total

    def _calculate_results(self) -> dict:
        """Calculate backtest performance metrics."""
        if not self.equity_curve:
            return {"error": "No equity data"}

        eq = pd.DataFrame(self.equity_curve).set_index("date")
        eq["returns"] = eq["equity"].pct_change().fillna(0)

        total_return = (eq["equity"].iloc[-1] / self.initial_capital) - 1
        daily_returns = eq["returns"]
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

        # Max drawdown
        cummax = eq["equity"].cummax()
        drawdown = (cummax - eq["equity"]) / cummax
        max_drawdown = drawdown.max()

        # Win/loss stats
        sell_trades = [t for t in self.trades if t["action"] == "SELL"]
        wins = [t for t in sell_trades if t.get("pnl", 0) > 0]
        losses = [t for t in sell_trades if t.get("pnl", 0) <= 0]

        win_rate = len(wins) / len(sell_trades) if sell_trades else 0
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t["pnl"]) for t in losses]) if losses else 0

        return {
            "initial_capital": self.initial_capital,
            "final_equity": eq["equity"].iloc[-1],
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "total_trades": len(self.trades),
            "win_rate": win_rate,
            "win_rate_pct": win_rate * 100,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": (avg_win * len(wins)) / (avg_loss * len(losses)) if losses and avg_loss > 0 else float("inf"),
            "equity_curve": eq,
            "trades": self.trades,
        }
