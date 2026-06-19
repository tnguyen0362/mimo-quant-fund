import logging

from config.settings import config

logger = logging.getLogger(__name__)


class RiskManager:
    """Manage trading risk with circuit breakers and limits."""

    def __init__(self):
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.peak_portfolio_value = 0.0
        self.trading_halted = False
        self.halt_reason = ""

    def update_portfolio_state(self, portfolio_value: float, daily_pnl: float):
        self.daily_pnl = daily_pnl
        self.total_pnl += daily_pnl
        self.peak_portfolio_value = max(self.peak_portfolio_value, portfolio_value)

    def check_daily_loss_limit(self, portfolio_value: float) -> bool:
        daily_loss_pct = abs(self.daily_pnl) / portfolio_value if self.daily_pnl < 0 else 0
        if daily_loss_pct > config.MAX_DAILY_LOSS:
            self.trading_halted = True
            self.halt_reason = f"Daily loss limit breached: {daily_loss_pct*100:.1f}% > {config.MAX_DAILY_LOSS*100}%"
            logger.warning(self.halt_reason)
            return False
        return True

    def check_drawdown_limit(self, portfolio_value: float) -> bool:
        if self.peak_portfolio_value == 0:
            return True
        drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
        if drawdown > config.MAX_DRAWDOWN:
            self.trading_halted = True
            self.halt_reason = f"Max drawdown breached: {drawdown*100:.1f}% > {config.MAX_DRAWDOWN*100}%"
            logger.warning(self.halt_reason)
            return False
        return True

    def check_position_risk(self, portfolio_value: float, position_value: float) -> bool:
        position_risk = position_value / portfolio_value
        if position_risk > config.MAX_POSITION_RISK:
            logger.warning(f"Position too large: {position_risk*100:.1f}% > {config.MAX_POSITION_RISK*100}%")
            return False
        return True

    def check_liquidity(self, position_value: float, avg_volume: float) -> bool:
        if avg_volume == 0:
            logger.warning("No volume data available")
            return False
        if position_value > avg_volume * 0.1:
            logger.warning(f"Liquidity risk: position ${position_value:.0f} > 10% of volume ${avg_volume:.0f}")
            return False
        return True

    def can_trade(self, portfolio_value: float, position_value: float = 0, avg_volume: float = 0) -> tuple:
        """Master risk check - returns (can_trade, list_of_reasons)"""
        if self.trading_halted:
            return False, [f"Trading halted: {self.halt_reason}"]

        reasons = []

        if not self.check_daily_loss_limit(portfolio_value):
            reasons.append("Daily loss limit breached")

        if not self.check_drawdown_limit(portfolio_value):
            reasons.append("Max drawdown breached")

        if position_value > 0:
            if not self.check_position_risk(portfolio_value, position_value):
                reasons.append("Position size too large")
            if avg_volume > 0 and not self.check_liquidity(position_value, avg_volume):
                reasons.append("Insufficient liquidity")

        return len(reasons) == 0, reasons

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.trading_halted = False
        self.halt_reason = ""
        logger.info("Daily risk state reset")

    def get_risk_status(self) -> dict:
        return {
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "peak_portfolio_value": self.peak_portfolio_value,
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
        }
