# Volatility targeting / scaling overlay
# - Scale portfolio exposure to maintain constant target volatility
# - Uses rolling realized volatility (EWMA or SMA)
# - Caps leverage at 2x
# - Reduces exposure when vol is high, increases when low

import pandas as pd
import numpy as np

class VolatilityTargeting:
    """
    Volatility targeting overlay.
    
    Academic basis: Moreira & Muir (2017)
    "Volatility-Managed Portfolios"
    
    Key insight: Scaling portfolio exposure by (target_vol / realized_vol)
    improves Sharpe ratio by 10-20% by reducing exposure during high-vol
    regimes (when Sharpe ratios decline).
    
    Implementation:
    - Estimate realized volatility (rolling window or EWMA)
    - Scale total portfolio exposure: leverage = target_vol / realized_vol
    - Cap leverage (default: 2x max)
    - Floor leverage (default: 0x min, i.e., cash)
    """
    
    def __init__(self, target_vol: float = 0.15,
                 vol_lookback: int = 60,
                 vol_method: str = "ewma",  # "ewma" or "sma"
                 ewma_halflife: int = 20,
                 max_leverage: float = 2.0,
                 min_leverage: float = 0.0):
        """
        Args:
            target_vol: Target annualized volatility (default: 15%)
            vol_lookback: Lookback window for vol estimation (default: 60 days)
            vol_method: "ewma" or "sma" for volatility estimation
            ewma_halflife: Half-life for EWMA (default: 20 days)
            max_leverage: Maximum leverage (default: 2.0 = 200%)
            min_leverage: Minimum exposure (default: 0.0 = full cash)
        """
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.vol_method = vol_method
        self.ewma_halflife = ewma_halflife
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
    
    def estimate_volatility(self, returns: pd.Series) -> pd.Series:
        """
        Estimate realized volatility.
        
        Args:
            returns: Daily return series
        
        Returns:
            Annualized volatility series
        """
        if self.vol_method == "ewma":
            # Exponentially weighted moving average
            vol = returns.ewm(
                halflife=self.ewma_halflife,
                min_periods=self.vol_lookback
            ).std() * np.sqrt(252)
        else:
            # Simple moving average
            vol = returns.rolling(
                window=self.vol_lookback,
                min_periods=self.vol_lookback
            ).std() * np.sqrt(252)
        
        return vol
    
    def compute_leverage(self, returns: pd.Series) -> pd.Series:
        """
        Compute target leverage based on realized volatility.
        
        Returns:
            Leverage series (0-2 by default)
        """
        realized_vol = self.estimate_volatility(returns)
        
        # Leverage = target / realized
        leverage = self.target_vol / realized_vol
        
        # Apply bounds
        leverage = leverage.clip(lower=self.min_leverage, upper=self.max_leverage)
        
        # Handle NaN (before we have enough data)
        leverage = leverage.fillna(1.0)
        
        return leverage
    
    def apply_to_portfolio(self, portfolio_returns: pd.Series) -> pd.Series:
        """
        Apply volatility targeting to portfolio returns.
        
        Args:
            portfolio_returns: Original portfolio daily returns
        
        Returns:
            Vol-targeted portfolio returns
        """
        leverage = self.compute_leverage(portfolio_returns)
        
        # Shift by 1 day to avoid look-ahead bias
        # Use yesterday's vol estimate for today's position
        leverage = leverage.shift(1).fillna(1.0)
        
        # Apply leverage
        vol_targeted_returns = portfolio_returns * leverage
        
        return vol_targeted_returns
    
    def compute_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volatility-targeted weights for each stock.
        
        Args:
            prices: DataFrame with dates as index, tickers as columns
        
        Returns:
            DataFrame of leverage multipliers per stock
        """
        returns = prices.pct_change()
        
        # Portfolio-level volatility (equal-weight)
        portfolio_returns = returns.mean(axis=1)
        leverage = self.compute_leverage(portfolio_returns)
        leverage = leverage.shift(1).fillna(1.0)
        
        # Apply same leverage to all stocks
        leverage_df = pd.DataFrame(
            leverage.values.reshape(-1, 1).repeat(len(prices.columns), axis=1),
            index=prices.index,
            columns=prices.columns
        )
        
        return leverage_df


class RollingVolThreshold:
    """
    Simple rolling volatility threshold (Oracle's recommended alternative to HMM).
    
    Replaces Hidden Markov Model regime detection with a simpler,
    more robust approach:
    - High vol regime: volatility > 75th percentile -> reduce exposure
    - Normal regime: volatility between 25th-75th -> full exposure
    - Low vol regime: volatility < 25th -> can use slightly more leverage
    
    This captures 80% of the value of HMM regime detection
    with far fewer parameters and estimation errors.
    """
    
    def __init__(self, vol_lookback: int = 60,
                 high_vol_percentile: float = 75,
                 low_vol_percentile: float = 25,
                 high_vol_reduction: float = 0.5,  # Reduce to 50% in high vol
                 low_vol_boost: float = 1.2):       # Boost to 120% in low vol
        """
        Args:
            vol_lookback: Lookback for vol estimation
            high_vol_percentile: Threshold for high vol regime
            low_vol_percentile: Threshold for low vol regime
            high_vol_reduction: Exposure multiplier in high vol (default: 0.5)
            low_vol_boost: Exposure multiplier in low vol (default: 1.2)
        """
        self.vol_lookback = vol_lookback
        self.high_vol_percentile = high_vol_percentile
        self.low_vol_percentile = low_vol_percentile
        self.high_vol_reduction = high_vol_reduction
        self.low_vol_boost = low_vol_boost
    
    def classify_regime(self, returns: pd.Series) -> pd.Series:
        """
        Classify volatility regime.
        
        Returns:
            Series with values: "high", "normal", or "low"
        """
        # Rolling volatility
        vol = returns.rolling(self.vol_lookback).std() * np.sqrt(252)
        
        # Percentile thresholds (rolling)
        high_thresh = vol.rolling(252).quantile(self.high_vol_percentile / 100)
        low_thresh = vol.rolling(252).quantile(self.low_vol_percentile / 100)
        
        # Classify
        regime = pd.Series("normal", index=returns.index)
        regime[vol > high_thresh] = "high"
        regime[vol < low_thresh] = "low"
        
        return regime
    
    def compute_exposure(self, returns: pd.Series) -> pd.Series:
        """
        Compute exposure multiplier based on regime.
        
        Returns:
            Exposure series (0.5-1.2 by default)
        """
        regime = self.classify_regime(returns)
        
        exposure = pd.Series(1.0, index=returns.index)
        exposure[regime == "high"] = self.high_vol_reduction
        exposure[regime == "low"] = self.low_vol_boost
        
        return exposure
