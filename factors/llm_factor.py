# LLM-Driven Alpha Factor
#
# This is the CORE differentiator — MiMo 2.5 analyzes news and earnings
# to generate sentiment signals that augment the rule-based factors.
#
# Signal: sentiment_score × confidence = alpha signal
# The LLM's confidence weights its opinion — low-confidence = small signal

import pandas as pd
import numpy as np
from typing import Optional
from llm.sentiment import MiMoSentiment


class LLMFactor:
    """
    LLM-driven alpha factor using MiMo 2.5.

    Architecture:
    - Takes news headlines + financial data for each stock
    - Sends to MiMo 2.5 for sentiment analysis
    - Returns sentiment scores as a factor signal
    - Confidence-weighted: low LLM confidence = smaller signal

    This factor is genuinely orthogonal to momentum and value:
    - Momentum = price-based (past returns)
    - Value = fundamental-based (financials)
    - Sentiment = information-based (news, events, context)

    Expected correlation with momentum: LOW (news can contradict price trends)
    Expected correlation with value: LOW (news can contradict cheap/expensive)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        confidence_threshold: float = 0.3,
        cache_dir: str = "data/cache/llm",
    ):
        """
        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            confidence_threshold: Minimum confidence to use LLM signal
            cache_dir: Directory to cache LLM responses
        """
        self.llm = MiMoSentiment(api_key=api_key, cache_dir=cache_dir)
        self.confidence_threshold = confidence_threshold

    def compute_signal(
        self,
        news_data: dict[str, list[str]],
        financials_data: Optional[dict] = None,
        tickers: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute LLM sentiment signals for all stocks.

        Args:
            news_data: Dict mapping ticker -> list of news headlines
            financials_data: Dict mapping ticker -> financials dict
            tickers: List of tickers to analyze (default: all in news_data)

        Returns:
            DataFrame with columns: ticker, sentiment, confidence, signal
            where signal = sentiment × confidence (confidence-weighted)
        """
        if tickers is None:
            tickers = list(news_data.keys())

        # Batch analyze all stocks
        df = self.llm.analyze_batch(
            tickers=tickers,
            news_data=news_data,
            financials_data=financials_data,
        )

        # Compute confidence-weighted signal
        df["signal"] = df["sentiment"] * df["confidence"]

        # Zero out low-confidence signals
        df.loc[df["confidence"] < self.confidence_threshold, "signal"] = 0.0

        return df

    def rank_stocks(
        self,
        news_data: dict[str, list[str]],
        financials_data: Optional[dict] = None,
    ) -> pd.Series:
        """
        Rank stocks by LLM sentiment (1 = most bullish).

        Returns:
            Series with ticker as index, rank as value
        """
        signal_df = self.compute_signal(news_data, financials_data)

        signal_df = signal_df.set_index("ticker")

        # Rank by signal (higher = more bullish = lower rank number)
        ranks = signal_df["signal"].rank(ascending=False)

        return ranks

    def get_top_n(
        self,
        news_data: dict[str, list[str]],
        financials_data: Optional[dict] = None,
        n: int = 20,
    ) -> pd.Index:
        """
        Get top N stocks by LLM sentiment.

        Returns:
            Index of tickers with highest bullish sentiment
        """
        signal_df = self.compute_signal(news_data, financials_data)

        # Sort by signal descending
        sorted_df = signal_df.sort_values("signal", ascending=False)

        # Take top N (only positive sentiment)
        top_n = sorted_df.head(n)
        top_n = top_n[top_n["signal"] > 0]

        return top_n["ticker"]
