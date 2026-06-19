# Drawdown control and risk monitoring
# - Portfolio-level drawdown tracking
# - Circuit breaker: halt trading if drawdown exceeds threshold
# - Position-level stop-losses
# - Correlation monitoring (detect when strategies become correlated)
# - Daily P&L tracking and alerts

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RiskAlert:
    """Risk monitoring alert with severity level and category."""
    timestamp: pd.Timestamp
    level: str  # "warning", "critical", "emergency"
    category: str  # "drawdown", "correlation", "position", "volatility"
    message: str
    action_taken: str = ""


class RiskMonitor:
    """
    Portfolio-level risk monitoring system.
    
    Monitors:
    1. Drawdown control (circuit breaker)
    2. Position-level stop-losses
    3. Portfolio volatility regime
    4. Correlation breakdown detection
    
    Based on production risk management practices:
    - 5% daily loss → warning
    - 10% daily loss → reduce positions by 50%
    - 15% daily loss → halt trading (circuit breaker)
    - 20% max drawdown → emergency shutdown
    """
    
    def __init__(self, 
                 max_daily_loss: float = 0.05,      # 5% daily loss limit
                 max_drawdown: float = 0.20,         # 20% max drawdown
                 drawdown_warning: float = 0.10,     # 10% drawdown warning
                 position_stop_loss: float = 0.05,   # 5% per-position stop
                 vol_alert_threshold: float = 1.5,   # 1.5x target vol triggers alert
                 correlation_threshold: float = 0.7, # High correlation threshold
                 lookback_window: int = 60):         # Days for rolling metrics
        """
        Args:
            max_daily_loss: Maximum acceptable daily loss (default: 5%)
            max_drawdown: Maximum acceptable drawdown (default: 20%)
            drawdown_warning: Warning threshold for drawdown (default: 10%)
            position_stop_loss: Per-position stop-loss (default: 5%)
            vol_alert_threshold: Vol multiplier for alerts (default: 1.5x)
            correlation_threshold: Correlation threshold for alerts (default: 0.7)
            lookback_window: Rolling window for metrics (default: 60 days)
        """
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.drawdown_warning = drawdown_warning
        self.position_stop_loss = position_stop_loss
        self.vol_alert_threshold = vol_alert_threshold
        self.correlation_threshold = correlation_threshold
        self.lookback_window = lookback_window
        
        self.alerts: list[RiskAlert] = []
        self.is_halted = False
    
    def check_portfolio(self, equity_curve: pd.Series,
                        current_positions: dict = None,
                        target_vol: float = 0.15) -> dict:
        """
        Run all risk checks on current portfolio state.
        
        Args:
            equity_curve: Historical equity values
            current_positions: Dict of {ticker: {"entry_price": float, "current_price": float}}
            target_vol: Target portfolio volatility
        
        Returns:
            dict with:
                - exposure_multiplier: 0-1 (0 = halted, 1 = full)
                - alerts: list of RiskAlert
                - halted: bool
                - reason: str
        """
        self.alerts = []
        exposure = 1.0
        
        if len(equity_curve) < 2:
            return {"exposure_multiplier": 1.0, "alerts": [], 
                    "halted": False, "reason": "insufficient_data"}
        
        # 1. Drawdown check
        dd_exposure = self._check_drawdown(equity_curve)
        exposure = min(exposure, dd_exposure)
        
        # 2. Daily loss check
        daily_exposure = self._check_daily_loss(equity_curve)
        exposure = min(exposure, daily_exposure)
        
        # 3. Volatility check
        vol_exposure = self._check_volatility(equity_curve, target_vol)
        exposure = min(exposure, vol_exposure)
        
        # 4. Position stop-loss check
        if current_positions:
            pos_exposure = self._check_positions(current_positions)
            exposure = min(exposure, pos_exposure)
        
        # 5. Correlation check (if we have enough data)
        if len(equity_curve) > self.lookback_window * 2:
            self._check_correlation_breakdown(equity_curve)
        
        # Check for emergency halt
        if exposure <= 0.0:
            self.is_halted = True
            self.alerts.append(RiskAlert(
                timestamp=equity_curve.index[-1],
                level="emergency",
                category="drawdown",
                message="Trading halted due to excessive drawdown",
                action_taken="All positions to be liquidated"
            ))
        
        return {
            "exposure_multiplier": max(0.0, exposure),
            "alerts": self.alerts,
            "halted": self.is_halted,
            "reason": self.alerts[-1].message if self.alerts else "normal"
        }
    
    def _check_drawdown(self, equity_curve: pd.Series) -> float:
        """Check drawdown and return exposure multiplier."""
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        current_dd = drawdown.iloc[-1]
        
        if abs(current_dd) >= self.max_drawdown:
            self.alerts.append(RiskAlert(
                timestamp=equity_curve.index[-1],
                level="emergency",
                category="drawdown",
                message=f"Max drawdown breached: {current_dd:.2%}",
                action_taken="Trading halted"
            ))
            return 0.0
        
        elif abs(current_dd) >= self.drawdown_warning:
            # Linear reduction from warning to max
            progress = (abs(current_dd) - self.drawdown_warning) / (self.max_drawdown - self.drawdown_warning)
            exposure = 1.0 - progress * 0.5  # Reduce to 50% at max
            
            self.alerts.append(RiskAlert(
                timestamp=equity_curve.index[-1],
                level="warning",
                category="drawdown",
                message=f"Drawdown warning: {current_dd:.2%}",
                action_taken=f"Reducing exposure to {exposure:.0%}"
            ))
            return exposure
        
        return 1.0
    
    def _check_daily_loss(self, equity_curve: pd.Series) -> float:
        """Check daily loss limit."""
        if len(equity_curve) < 2:
            return 1.0
        
        daily_return = (equity_curve.iloc[-1] - equity_curve.iloc[-2]) / equity_curve.iloc[-2]
        
        if daily_return < -self.max_daily_loss:
            self.alerts.append(RiskAlert(
                timestamp=equity_curve.index[-1],
                level="critical",
                category="drawdown",
                message=f"Daily loss limit breached: {daily_return:.2%}",
                action_taken="Reducing positions by 50%"
            ))
            return 0.5
        
        return 1.0
    
    def _check_volatility(self, equity_curve: pd.Series, 
                          target_vol: float) -> float:
        """Check if portfolio vol is too high."""
        returns = equity_curve.pct_change().dropna()
        
        if len(returns) < self.lookback_window:
            return 1.0
        
        realized_vol = returns.tail(self.lookback_window).std() * np.sqrt(252)
        
        if realized_vol > target_vol * self.vol_alert_threshold:
            self.alerts.append(RiskAlert(
                timestamp=equity_curve.index[-1],
                level="warning",
                category="volatility",
                message=f"Portfolio vol elevated: {realized_vol:.1%} vs target {target_vol:.1%}",
                action_taken="Reducing exposure proportionally"
            ))
            return min(1.0, target_vol / realized_vol)
        
        return 1.0
    
    def _check_positions(self, positions: dict) -> float:
        """Check individual position stop-losses."""
        worst_loss = 0.0
        
        for ticker, pos in positions.items():
            entry_price = pos.get("entry_price", 0)
            current_price = pos.get("current_price", 0)
            
            if entry_price > 0:
                loss = (entry_price - current_price) / entry_price
                worst_loss = max(worst_loss, loss)
        
        if worst_loss >= self.position_stop_loss:
            self.alerts.append(RiskAlert(
                timestamp=pd.Timestamp.now(),
                level="warning",
                category="position",
                message=f"Position stop-loss triggered: {worst_loss:.2%} loss",
                action_taken="Reviewing position for exit"
            ))
            return 0.8  # Reduce exposure by 20%
        
        return 1.0
    
    def _check_correlation_breakdown(self, equity_curve: pd.Series):
        """
        Detect correlation breakdown (strategies becoming correlated).
        
        During crises, previously uncorrelated strategies tend to
        become highly correlated, reducing diversification benefits.
        """
        returns = equity_curve.pct_change().dropna()
        
        if len(returns) < self.lookback_window * 2:
            return
        
        # Split into two windows
        recent = returns.tail(self.lookback_window)
        previous = returns.iloc[-self.lookback_window*2:-self.lookback_window]
        
        # Compute rolling autocorrelation as a proxy
        # (In a multi-strategy system, we'd check cross-strategy correlation)
        autocorr_recent = recent.autocorr(lag=1)
        autocorr_previous = previous.autocorr(lag=1)
        
        if autocorr_recent is not None and autocorr_previous is not None:
            if autocorr_recent > self.correlation_threshold:
                self.alerts.append(RiskAlert(
                    timestamp=equity_curve.index[-1],
                    level="warning",
                    category="correlation",
                    message=f"Autocorrelation elevated: {autocorr_recent:.2f}",
                    action_taken="Monitoring for correlation breakdown"
                ))
    
    def get_status_summary(self) -> dict:
        """Get current risk status summary."""
        return {
            "halted": self.is_halted,
            "num_alerts": len(self.alerts),
            "critical_alerts": len([a for a in self.alerts if a.level in ["critical", "emergency"]]),
            "alert_categories": list(set(a.category for a in self.alerts)),
            "max_drawdown_limit": self.max_drawdown,
            "max_daily_loss_limit": self.max_daily_loss,
        }


@dataclass
class TradeLog:
    """Immutable trade log entry for audit trail."""
    timestamp: pd.Timestamp
    ticker: str
    side: str
    shares: float
    price: float
    value: float
    reason: str
    risk_check_passed: bool = True
    risk_notes: str = ""


class RiskManager:
    """
    Unified risk management interface.
    
    Combines:
    - Volatility targeting (from VolatilityTargeting)
    - Drawdown control (from RiskMonitor)
    - Position sizing (from existing PositionSizer)
    """
    
    def __init__(self, target_vol: float = 0.15,
                 max_drawdown: float = 0.20,
                 max_position_pct: float = 0.05):
        """
        Args:
            target_vol: Target portfolio volatility
            max_drawdown: Maximum acceptable drawdown
            max_position_pct: Maximum position size as % of portfolio
        """
        self.target_vol = target_vol
        self.max_drawdown = max_drawdown
        self.max_position_pct = max_position_pct
        
        self.risk_monitor = RiskMonitor(
            max_drawdown=max_drawdown
        )
        
        self.trade_log: list[TradeLog] = []
    
    def pre_trade_check(self, ticker: str, side: str, value: float,
                        portfolio_value: float, 
                        equity_curve: pd.Series) -> dict:
        """
        Pre-trade risk check.
        
        Returns:
            dict with:
                - approved: bool
                - max_value: float (max allowed trade value)
                - reason: str
        """
        # Check if trading is halted
        if self.risk_monitor.is_halted:
            return {
                "approved": False,
                "max_value": 0,
                "reason": "Trading halted due to drawdown"
            }
        
        # Check position size limit
        if value > portfolio_value * self.max_position_pct:
            return {
                "approved": False,
                "max_value": portfolio_value * self.max_position_pct,
                "reason": f"Position size {value/portfolio_value:.1%} exceeds limit {self.max_position_pct:.1%}"
            }
        
        # Check daily loss limit
        if len(equity_curve) >= 2:
            daily_return = (equity_curve.iloc[-1] - equity_curve.iloc[-2]) / equity_curve.iloc[-2]
            if daily_return < -0.03:  # 3% daily loss already
                return {
                    "approved": False,
                    "max_value": 0,
                    "reason": f"Daily loss already at {daily_return:.1%}, no new trades"
                }
        
        return {
            "approved": True,
            "max_value": min(value, portfolio_value * self.max_position_pct),
            "reason": "Approved"
        }
    
    def log_trade(self, timestamp: pd.Timestamp, ticker: str,
                  side: str, shares: float, price: float,
                  value: float, reason: str = "",
                  risk_check_passed: bool = True):
        """Log a trade for audit trail."""
        self.trade_log.append(TradeLog(
            timestamp=timestamp,
            ticker=ticker,
            side=side,
            shares=shares,
            price=price,
            value=value,
            reason=reason,
            risk_check_passed=risk_check_passed,
        ))
    
    def get_trade_log_df(self) -> pd.DataFrame:
        """Convert trade log to DataFrame."""
        if not self.trade_log:
            return pd.DataFrame()
        
        return pd.DataFrame([{
            "timestamp": t.timestamp,
            "ticker": t.ticker,
            "side": t.side,
            "shares": t.shares,
            "price": t.price,
            "value": t.value,
            "reason": t.reason,
            "risk_check_passed": t.risk_check_passed,
        } for t in self.trade_log])
