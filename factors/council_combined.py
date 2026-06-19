# Council-Driven Combined Ranking
#
# 40% Momentum + 30% Value + 30% LLM Council
# The council replaces single-model LLM sentiment with multi-model voting.

import pandas as pd
import numpy as np
from typing import Optional
from .momentum import MomentumFactor
from .value import ValueFactor
from llm.council import LLMCouncil, print_council_result


class CouncilCombinedRanking:
    """
    Three-factor ranking using LLM Council instead of single model.
    
    40% Cross-Sectional Momentum (12-1 skip)
    30% Value Factor (B/M + E/P)
    30% LLM Council (4 free models voting independently)
    
    The council provides more robust sentiment signals because:
    - Multiple models reduce individual bias
    - Disagreement = lower confidence (valuable signal!)
    - All models are FREE on OpenRouter
    """
    
    def __init__(self,
                 momentum_weight: float = 0.40,
                 value_weight: float = 0.30,
                 council_weight: float = 0.30,
                 top_n: int = 15,
                 api_key: Optional[str] = None):
        self.momentum_weight = momentum_weight
        self.value_weight = value_weight
        self.council_weight = council_weight
        self.top_n = top_n
        
        self.momentum = MomentumFactor(lookback_days=252, skip_days=21)
        self.value = ValueFactor()
        self.council = LLMCouncil(api_key=api_key)
    
    def compute_portfolio(self, prices: pd.DataFrame,
                          fundamentals: pd.DataFrame,
                          news_data: dict[str, list[str]]) -> pd.DataFrame:
        """
        Compute top-N portfolio using three-factor ranking.
        
        Args:
            prices: Historical price matrix (days x tickers)
            fundamentals: Fundamental data (tickers x features)
            news_data: Dict mapping ticker -> list of news headlines
        
        Returns:
            DataFrame with columns: ticker, weight, rank, momentum_score, 
                                    value_score, council_score
        """
        tickers = prices.columns.tolist()
        
        # Factor 1: Momentum (last row only = current cross-section)
        mom_rank = self.momentum.rank_stocks(prices)
        if isinstance(mom_rank, pd.DataFrame):
            mom_rank = mom_rank.iloc[-1]  # Last date's rankings
        mom_score = 1.0 - (mom_rank - 1) / max(len(tickers) - 1, 1)
        
        # Factor 2: Value
        val_rank = self.value.rank_stocks(fundamentals)
        val_score = pd.Series(0.5, index=tickers)
        for t in tickers:
            if t in val_rank.index:
                r = val_rank.loc[t]
                val_score.loc[t] = 1.0 - (r - 1) / max(len(val_rank) - 1, 1)
        
        # Factor 3: LLM Council
        council_score = pd.Series(0.0, index=tickers)
        
        try:
            for ticker in tickers:
                headlines = news_data.get(ticker, [])
                fins_dict = {}
                if fundamentals is not None and ticker in fundamentals.index:
                    row = fundamentals.loc[ticker]
                    fins_dict = {
                        "pe_ratio": row.get("pe_ratio"),
                        "market_cap": row.get("market_cap"),
                        "sector": "Unknown",
                    }
                
                result = self.council.deliberate(ticker, headlines, fins_dict)
                
                # Normalize sentiment (-1..1) to score (0..1)
                council_score.loc[ticker] = (result.sentiment + 1.0) / 2.0
                
                print_council_result(result)
        except Exception as e:
            print(f"  Council error: {e} — using neutral scores")
            council_score = pd.Series(0.5, index=tickers)
        
        # Combined score
        combined = (
            self.momentum_weight * mom_score +
            self.value_weight * val_score +
            self.council_weight * council_score
        )
        
        # Ensure it's a Series (not DataFrame from index mismatch)
        if isinstance(combined, pd.DataFrame):
            combined = combined.squeeze()
        
        combined = combined.sort_values(ascending=False)
        
        top_tickers = combined.head(self.top_n)
        weight = 1.0 / len(top_tickers) if len(top_tickers) > 0 else 0.0
        
        result = pd.DataFrame({
            "ticker": top_tickers.index,
            "weight": weight,
            "rank": range(1, len(top_tickers) + 1),
            "momentum_score": [mom_score.get(t, 0.5) for t in top_tickers.index],
            "value_score": [val_score.get(t, 0.5) for t in top_tickers.index],
            "council_score": [council_score.get(t, 0.5) for t in top_tickers.index],
            "combined_score": top_tickers.values,
        })
        
        return result
