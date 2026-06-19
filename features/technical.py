import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import logging

from config.settings import config

logger = logging.getLogger(__name__)


class TechnicalFeatures:
    """Calculate technical indicators for trading signals."""

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for a price DataFrame."""
        if df.empty or len(df) < 50:
            logger.warning("Insufficient data for technical indicators")
            return df

        df = df.copy()

        # Moving Averages
        df["sma_20"] = SMAIndicator(df["close"], window=20).sma_indicator()
        df["sma_50"] = SMAIndicator(df["close"], window=50).sma_indicator()
        df["sma_200"] = SMAIndicator(df["close"], window=200).sma_indicator()
        df["ema_12"] = EMAIndicator(df["close"], window=12).ema_indicator()
        df["ema_26"] = EMAIndicator(df["close"], window=26).ema_indicator()

        # Momentum
        df["momentum_20"] = df["close"].pct_change(config.MOMENTUM_PERIOD)
        df["momentum_60"] = df["close"].pct_change(60)

        # RSI
        df["rsi"] = RSIIndicator(df["close"], window=14).rsi()

        # MACD
        macd = MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        # Bollinger Bands
        bbands = BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_upper"] = bbands.bollinger_hband()
        df["bb_middle"] = bbands.bollinger_mavg()
        df["bb_lower"] = bbands.bollinger_lband()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # ATR
        df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

        # Volume indicators
        df["volume_sma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # Volatility
        df["volatility_20"] = df["close"].pct_change().rolling(window=20).std() * np.sqrt(252)

        logger.info(f"Calculated technical indicators for {len(df)} rows")
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate basic technical signals."""
        if df.empty:
            return df

        df = df.copy()

        # Momentum signal
        df["momentum_signal"] = 0
        df.loc[df["momentum_20"] > 0.05, "momentum_signal"] = 1
        df.loc[df["momentum_20"] < -0.05, "momentum_signal"] = -1

        # RSI signal
        df["rsi_signal"] = 0
        df.loc[df["rsi"] < config.RSI_OVERSOLD, "rsi_signal"] = 1
        df.loc[df["rsi"] > config.RSI_OVERBOUGHT, "rsi_signal"] = -1

        # MACD signal
        df["macd_signal_indicator"] = 0
        df.loc[df["macd"] > df["macd_signal"], "macd_signal_indicator"] = 1
        df.loc[df["macd"] < df["macd_signal"], "macd_signal_indicator"] = -1

        # Moving average crossover
        df["ma_crossover_signal"] = 0
        df.loc[df["sma_20"] > df["sma_50"], "ma_crossover_signal"] = 1
        df.loc[df["sma_20"] < df["sma_50"], "ma_crossover_signal"] = -1

        # Combined technical score
        df["technical_score"] = (
            df["momentum_signal"] * 0.3
            + df["rsi_signal"] * 0.2
            + df["macd_signal_indicator"] * 0.3
            + df["ma_crossover_signal"] * 0.2
        )

        logger.info("Generated technical signals")
        return df
