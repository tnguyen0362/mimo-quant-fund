import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TradingConfig:
    # Data paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    LOG_DIR: Path = PROJECT_ROOT / "logs"
    DB_PATH: Path = PROJECT_ROOT / "trading.db"

    # API Keys (from environment)
    FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
    MIMO_API_KEY: str = os.getenv("MIMO_API_KEY", "")
    MIMO_API_URL: str = os.getenv("MIMO_API_URL", "https://api.mimo.xiaomi.com/v1")

    # Trading parameters
    MAX_POSITION_RISK: float = 0.02       # 2% portfolio risk per trade
    MAX_DAILY_LOSS: float = 0.05          # 5% daily loss limit
    MAX_DRAWDOWN: float = 0.20            # 20% maximum drawdown
    MAX_POSITIONS: int = 10               # Maximum concurrent positions
    MAX_ORDER_VALUE: float = 10_000       # Maximum $10,000 per order
    MIN_VOLUME_MULTIPLIER: float = 10     # Volume must be 10x position size

    # Signal parameters
    MOMENTUM_PERIOD: int = 20             # 20-day momentum
    RSI_OVERSOLD: float = 30
    RSI_OVERBOUGHT: float = 70
    SIGNAL_THRESHOLD_BUY: float = 0.3
    SIGNAL_THRESHOLD_SELL: float = -0.2

    # Sentiment weights
    TECHNICAL_WEIGHT: float = 0.6
    SENTIMENT_WEIGHT: float = 0.4

    # Backtesting
    BACKTEST_YEARS: int = 5
    MIN_SHARPE_RATIO: float = 1.0
    MAX_BACKTEST_DRAWDOWN: float = 0.20

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


config = TradingConfig()
