import pandas as pd
import logging

from config.settings import config
from features.technical import TechnicalFeatures
from features.fundamental import FundamentalFeatures

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generate trading signals by combining technical and fundamental analysis."""

    def __init__(self):
        self.technical = TechnicalFeatures()
        self.fundamental = FundamentalFeatures()

    def generate_combined_signal(self, price_data: pd.DataFrame, fundamental_score: float = 0.5) -> dict:
        """Generate a combined trading signal."""
        if price_data.empty:
            return {"signal": "HOLD", "strength": 0, "reason": "No price data"}

        # Calculate technical indicators
        df = self.technical.calculate_all_indicators(price_data)
        df = self.technical.generate_signals(df)

        # Get latest values
        latest = df.iloc[-1]
        technical_score = latest.get("technical_score", 0)

        # Combined score
        combined_score = (
            technical_score * config.TECHNICAL_WEIGHT
            + (fundamental_score - 0.5) * 2 * config.SENTIMENT_WEIGHT
        )

        # Generate signal
        if combined_score > config.SIGNAL_THRESHOLD_BUY:
            signal = "BUY"
            reason = f"Strong positive signal: {combined_score:.2f}"
        elif combined_score < config.SIGNAL_THRESHOLD_SELL:
            signal = "SELL"
            reason = f"Strong negative signal: {combined_score:.2f}"
        else:
            signal = "HOLD"
            reason = f"Neutral signal: {combined_score:.2f}"

        result = {
            "signal": signal,
            "strength": combined_score,
            "reason": reason,
            "technical_score": technical_score,
            "fundamental_score": fundamental_score,
            "rsi": latest.get("rsi"),
            "macd": latest.get("macd"),
            "momentum_20": latest.get("momentum_20"),
            "price": latest.get("close"),
            "timestamp": pd.Timestamp.now(),
        }

        logger.info(f"Generated signal: {signal} (strength: {combined_score:.2f})")
        return result

    def analyze_ticker(self, ticker: str, price_data: pd.DataFrame, fundamental_data: dict = None) -> dict:
        """Complete analysis for a single ticker."""
        fundamental_score = 0.5
        if fundamental_data:
            ratios = fundamental_data.get("ratios", {})
            fundamental_score = self.fundamental.calculate_fundamental_score(
                fundamental_data.get("metrics", {}), ratios
            )

        signal = self.generate_combined_signal(price_data, fundamental_score)
        signal["ticker"] = ticker
        return signal
