"""
Hybrid Council + Rules Stock Picker
====================================
Combines rule-based portfolio construction with LLM council conviction scoring:

1. Rule-based system (CombinedRanking) ranks ALL stocks by momentum+value → top 15
2. Council analyzes only the top 15 stocks (not 20+20, just 15)
3. For each stock, council returns: BUY (boost weight), HOLD (keep weight), SELL (cut weight)
4. Final weights adjusted by council conviction
5. Equal-weight within adjusted weights

This gives us:
- Rules provide the BASE portfolio (always diversified, always has positions)
- Council provides CONVICTION SCORING (boost winners, cut losers)

Academic basis: Combining quantitative factors with qualitative LLM reasoning
to improve risk-adjusted returns while maintaining diversification.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

from .combined import CombinedRanking
from .council_driven import CouncilStockPicker, StockPick

logger = logging.getLogger(__name__)


class HybridStockPicker:
    """
    Hybrid system: rules for base, council for conviction.
    
    Architecture:
    1. Rule-based system ranks ALL stocks by momentum+value → top 15
    2. Council analyzes only the top 15 (not 20+20, just 15)
    3. For each stock, council returns: action (BUY/HOLD/SELL) + conviction (0-1)
    4. Final weights adjusted by council conviction:
       - BUY with conviction 0.8 → weight × 1.8
       - HOLD → weight × 1.0
       - SELL with conviction 0.7 → weight × 0.3
       - Council unavailable → weight × 1.0 (fallback to rules)
    5. Equal-weight within adjusted weights
    """
    
    def __init__(
        self,
        council_picker: Optional[CouncilStockPicker] = None,
        momentum_weight: float = 0.6,
        value_weight: float = 0.4,
        top_n: int = 15,
        buy_multiplier: float = 1.5,
        sell_multiplier: float = 0.3,
        council_timeout_seconds: float = 30.0,
    ):
        """
        Args:
            council_picker: CouncilStockPicker instance (created if None)
            momentum_weight: Weight for momentum factor in rules (default: 0.6)
            value_weight: Weight for value factor in rules (default: 0.4)
            top_n: Number of stocks for rule-based ranking (default: 15)
            buy_multiplier: Weight boost for BUY conviction (default: 1.5)
            sell_multiplier: Weight cut for SELL conviction (default: 0.3)
            council_timeout_seconds: Timeout for council calls (default: 30)
        """
        # Rule-based component
        self.rules = CombinedRanking(
            momentum_weight=momentum_weight,
            value_weight=value_weight,
            top_n=top_n,
        )
        self.top_n = top_n
        
        # Council component
        self.council_picker = council_picker or CouncilStockPicker(
            prefilter_top_n=top_n,  # Don't pre-filter, rules already did
            final_top_n=top_n,
        )
        
        # Weight adjustment parameters
        self.buy_multiplier = buy_multiplier
        self.sell_multiplier = sell_multiplier
        self.council_timeout_seconds = council_timeout_seconds
    
    def get_signals(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Get trading signals for the hybrid strategy.
        
        1. Get rule-based portfolio (always succeeds)
        2. Get council scores (may fail)
        3. Adjust weights based on council conviction
        4. Return signal matrix (1.0 = hold, 0.0 = flat)
        
        Args:
            prices: Price data for momentum ranking
            fundamentals: Fundamental data for value ranking
            
        Returns:
            DataFrame with dates as index, tickers as columns.
            Values: 1.0 (hold) or 0.0 (flat)
        """
        # Step 1: Get rule-based portfolio (this always works)
        rule_portfolio = self.rules.compute_portfolio(prices, fundamentals)
        
        if len(rule_portfolio) == 0:
            logger.warning("Rule-based portfolio is empty, returning empty signals")
            return pd.DataFrame()
        
        # Get tickers and base weights from rules
        rule_tickers = rule_portfolio["ticker"].tolist()
        rule_weights = rule_portfolio.set_index("ticker")["weight"].to_dict()
        
        logger.info(f"Rule-based portfolio: {len(rule_tickers)} stocks")
        
        # Step 2: Get council scores (may fail)
        council_scores = self._get_council_scores(
            prices, fundamentals, rule_tickers
        )
        
        # Step 3: Adjust weights based on council conviction
        adjusted_weights = self._adjust_weights(rule_weights, council_scores)
        
        # Step 4: Create signal matrix
        # Signal = 1.0 means "hold this stock" (portfolio engine does equal weighting)
        signals = pd.DataFrame(0.0, index=[0], columns=list(adjusted_weights.keys()))
        
        for ticker, weight in adjusted_weights.items():
            if weight > 0:
                signals.loc[0, ticker] = 1.0
        
        logger.info(
            f"Hybrid signals: {int(signals.iloc[0].sum())} stocks with signal=1.0"
        )
        
        return signals
    
    def _get_council_scores(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
        tickers: list[str],
    ) -> dict[str, StockPick]:
        """
        Get council scores for the top N stocks.
        
        Falls back gracefully if council is unavailable (API down, rate limited).
        
        Returns:
            Dict mapping ticker → StockPick (empty dict if council fails)
        """
        try:
            # Get the latest date from prices
            if prices.empty:
                return {}
            
            latest_date = prices.index[-1]
            
            # Get council picks
            council_picks = self.council_picker.pick_stocks(
                prices, fundamentals, latest_date
            )
            
            # Convert to dict for easy lookup
            scores = {}
            for pick in council_picks:
                if pick.ticker in tickers:  # Only score stocks in our rule portfolio
                    scores[pick.ticker] = pick
            
            logger.info(
                f"Council scored {len(scores)} of {len(tickers)} stocks"
            )
            
            return scores
            
        except Exception as e:
            logger.warning(
                f"Council failed (falling back to rules only): {e}"
            )
            return {}
    
    def _adjust_weights(
        self,
        base_weights: dict[str, float],
        council_scores: dict[str, StockPick],
    ) -> dict[str, float]:
        """
        Adjust rule-based weights based on council conviction.
        
        Args:
            base_weights: dict[ticker, float] - equal weight from rules
            council_scores: dict[ticker, StockPick] - council verdicts
            
        Returns:
            dict[ticker, float] - adjusted weights (normalized to sum to 1)
        """
        adjusted = {}
        
        for ticker, weight in base_weights.items():
            if ticker in council_scores:
                pick = council_scores[ticker]
                
                if pick.action == "BUY" and pick.conviction > 0.5:
                    # Boost weight: weight × (1 + conviction × multiplier_delta)
                    # conviction 0.8 → boost up to buy_multiplier
                    boost = 1.0 + pick.conviction * (self.buy_multiplier - 1.0)
                    adjusted[ticker] = weight * boost
                    
                elif pick.action == "SELL" and pick.conviction > 0.5:
                    # Cut weight: weight × (1 - conviction × (1 - sell_multiplier))
                    # conviction 0.7 → cut down to sell_multiplier
                    cut = 1.0 - pick.conviction * (1.0 - self.sell_multiplier)
                    adjusted[ticker] = weight * max(cut, 0.1)  # Floor at 10% of original
                    
                else:
                    # HOLD or low conviction → keep weight
                    adjusted[ticker] = weight
            else:
                # Council didn't score this, keep rules weight
                adjusted[ticker] = weight
        
        # Normalize to sum to 1
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {t: w / total for t, w in adjusted.items()}
        
        return adjusted
    
    def compute_portfolio(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute hybrid portfolio with adjusted weights.
        
        Same as get_signals but returns full portfolio info.
        
        Returns:
            DataFrame with columns:
                - ticker: Stock ticker
                - weight: Adjusted portfolio weight
                - rule_weight: Original rule-based weight
                - council_action: Council action (BUY/HOLD/SELL)
                - council_conviction: Council conviction (0-1)
        """
        # Get rule-based portfolio
        rule_portfolio = self.rules.compute_portfolio(prices, fundamentals)
        
        if len(rule_portfolio) == 0:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "weight",
                    "rule_weight",
                    "council_action",
                    "council_conviction",
                ]
            )
        
        # Get tickers and base weights
        rule_tickers = rule_portfolio["ticker"].tolist()
        rule_weights = rule_portfolio.set_index("ticker")["weight"].to_dict()
        
        # Get council scores
        council_scores = self._get_council_scores(
            prices, fundamentals, rule_tickers
        )
        
        # Adjust weights
        adjusted_weights = self._adjust_weights(rule_weights, council_scores)
        
        # Build result DataFrame
        rows = []
        for ticker in rule_tickers:
            rows.append({
                "ticker": ticker,
                "weight": adjusted_weights.get(ticker, 0),
                "rule_weight": rule_weights.get(ticker, 0),
                "council_action": (
                    council_scores[ticker].action
                    if ticker in council_scores
                    else "N/A"
                ),
                "council_conviction": (
                    council_scores[ticker].conviction
                    if ticker in council_scores
                    else 0.0
                ),
            })
        
        return pd.DataFrame(rows)


def create_hybrid_picker(
    api_key: Optional[str] = None,
    momentum_weight: float = 0.6,
    value_weight: float = 0.4,
    top_n: int = 15,
) -> HybridStockPicker:
    """
    Factory function to create a HybridStockPicker with sensible defaults.
    
    Args:
        api_key: OpenRouter API key for council
        momentum_weight: Weight for momentum (default: 0.6)
        value_weight: Weight for value (default: 0.4)
        top_n: Number of stocks (default: 15)
        
    Returns:
        Configured HybridStockPicker instance
    """
    council_picker = CouncilStockPicker(
        api_key=api_key,
        prefilter_top_n=top_n,
        final_top_n=top_n,
    )
    
    return HybridStockPicker(
        council_picker=council_picker,
        momentum_weight=momentum_weight,
        value_weight=value_weight,
        top_n=top_n,
    )
