import logging

from config.settings import config

logger = logging.getLogger(__name__)


class PositionSizer:
    """Calculate position sizes using half-Kelly criterion."""

    def __init__(self):
        self.max_position_risk = config.MAX_POSITION_RISK
        self.max_order_value = config.MAX_ORDER_VALUE

    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly criterion fraction."""
        if avg_loss == 0:
            return 0

        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p

        kelly = (b * p - q) / b
        half_kelly = kelly / 2

        return max(0, min(half_kelly, 0.15))

    def calculate_position_size(
        self,
        portfolio_value: float,
        current_price: float,
        win_rate: float = 0.5,
        avg_win: float = 0.05,
        avg_loss: float = 0.03,
    ) -> dict:
        """Calculate position size in shares and dollar amount."""
        kelly_fraction = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        max_dollar = portfolio_value * self.max_position_risk
        position_dollar = min(max_dollar, portfolio_value * kelly_fraction)
        position_dollar = min(position_dollar, self.max_order_value)

        shares = int(position_dollar / current_price) if current_price > 0 else 0
        actual_value = shares * current_price

        result = {
            "kelly_fraction": kelly_fraction,
            "position_dollar": actual_value,
            "shares": shares,
            "current_price": current_price,
            "portfolio_percent": (actual_value / portfolio_value * 100) if portfolio_value > 0 else 0,
        }

        logger.info(f"Position size: {shares} shares @ ${current_price:.2f} = ${actual_value:.2f}")
        return result

    def validate_position(self, portfolio_value: float, position_value: float, current_positions: int) -> tuple:
        """Validate if position meets risk criteria."""
        errors = []

        if position_value / portfolio_value > self.max_position_risk:
            errors.append(f"Position too large: {position_value/portfolio_value*100:.1f}% > {self.max_position_risk*100}%")

        if current_positions >= config.MAX_POSITIONS:
            errors.append(f"Max positions reached: {current_positions} >= {config.MAX_POSITIONS}")

        if position_value > self.max_order_value:
            errors.append(f"Order too large: ${position_value:.2f} > ${self.max_order_value:.2f}")

        return len(errors) == 0, errors
