# Council-Driven Alpha Factor
#
# Uses the LLM Council (4 free models voting independently)
# instead of a single model. More robust, less biased.

import pandas as pd
import numpy as np
from typing import Optional
from llm.council import LLMCouncil, CouncilResult


class CouncilFactor:
    """
    LLM Council-driven alpha factor.
    
    Instead of one model making the call, 4 free models vote independently:
    - Llama 3.3 70B
    - Qwen3 Next 80B
    - Gemma 4 31B
    - GPT-OSS 120B
    
    Their votes are confidence-weighted and aggregated.
    Disagreement = lower confidence (valuable signal!).
    """
    
    def __init__(self, api_key: Optional[str] = None,
                 confidence_threshold: float = 0.2,
                 cache_dir: str = "data/cache/council"):
        """
        Args:
            api_key: OpenRouter API key (or OPENROUTER_API_KEY env var)
            confidence_threshold: Minimum confidence to use council signal
            cache_dir: Cache directory for council responses
        """
        self.council = LLMCouncil(api_key=api_key, cache_dir=cache_dir)
        self.confidence_threshold = confidence_threshold
    
    def compute_signal(self, news_data: dict[str, list[str]],
                       financials_data: Optional[dict] = None,
                       tickers: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Compute council sentiment signals for all stocks.
        
        Args:
            news_data: Dict mapping ticker -> list of news headlines
            financials_data: Dict mapping ticker -> financials dict
            tickers: List of tickers to analyze
        
        Returns:
            DataFrame with: ticker, sentiment, confidence, agreement, signal
        """
        if tickers is None:
            tickers = list(news_data.keys())
        
        # Batch deliberation
        df = self.council.deliberate_batch(
            tickers=tickers,
            news_data=news_data,
            financials_data=financials_data,
        )
        
        # Compute confidence-weighted signal
        df["signal"] = df["sentiment"] * df["confidence"]
        
        # Zero out low-confidence signals
        df.loc[df["confidence"] < self.confidence_threshold, "signal"] = 0.0
        
        return df
    
    def rank_stocks(self, news_data: dict[str, list[str]],
                    financials_data: Optional[dict] = None) -> pd.Series:
        """Rank stocks by council sentiment (1 = most bullish)."""
        signal_df = self.compute_signal(news_data, financials_data)
        signal_df = signal_df.set_index("ticker")
        ranks = signal_df["signal"].rank(ascending=False)
        return ranks
