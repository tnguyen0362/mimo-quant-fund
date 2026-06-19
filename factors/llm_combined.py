# Three-Factor Combined Ranking: Momentum + Value + LLM Sentiment
#
# The LLM adds genuine alpha because:
# - It processes information (news, events) that momentum and value can't see
# - It can identify inflection points before they show up in price or fundamentals
# - Its reasoning is explainable (unlike black-box ML models)
#
# Weights: 40% momentum + 30% value + 30% LLM sentiment
# (LLM gets meaningful weight because it's the differentiator)

import pandas as pd
import numpy as np
from typing import Optional
from .momentum import MomentumFactor
from .value import ValueFactor
from .llm_factor import LLMFactor


class LLMCombinedRanking:
    """
    Three-factor combined ranking: Momentum + Value + LLM Sentiment.

    The LLM factor is the key innovation — it provides information-based
    alpha that's orthogonal to the rule-based factors.
    """

    def __init__(
        self,
        momentum_weight: float = 0.40,
        value_weight: float = 0.30,
        llm_weight: float = 0.30,
        top_n: int = 15,
        api_key: Optional[str] = None,
    ):
        """
        Args:
            momentum_weight: Weight for momentum factor
            value_weight: Weight for value factor
            llm_weight: Weight for LLM sentiment factor
            top_n: Number of stocks to hold
            api_key: OpenRouter API key for MiMo 2.5
        """
        self.momentum_weight = momentum_weight
        self.value_weight = value_weight
        self.llm_weight = llm_weight
        self.top_n = top_n

        self.momentum = MomentumFactor()
        self.value = ValueFactor()
        self.llm_factor = LLMFactor(api_key=api_key)

    def compute_portfolio(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
        news_data: dict[str, list[str]],
        financials_data: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Compute three-factor combined portfolio.

        Args:
            prices: Price data for momentum
            fundamentals: Fundamental data for value
            news_data: News headlines for LLM sentiment
            financials_data: Financials for LLM context

        Returns:
            DataFrame with ticker, weight, and factor scores
        """
        # Get factor ranks
        mom_ranks = self.momentum.rank_stocks(prices)
        val_ranks = self.value.rank_stocks(fundamentals)

        try:
            llm_ranks = self.llm_factor.rank_stocks(news_data, financials_data)
        except Exception:
            # LLM failed — fall back to 2-factor ranking
            llm_ranks = pd.Series(dtype=float)

        # Handle momentum ranks: DataFrame (dates x tickers) -> extract latest row as Series
        if len(mom_ranks) > 0:
            mom_valid = mom_ranks.dropna(how="all")
            mom_latest = (
                mom_valid.iloc[-1] if len(mom_valid) > 0 else pd.Series(dtype=float)
            )
        else:
            mom_latest = pd.Series(dtype=float)

        # Handle value ranks: DataFrame with "value_rank" column -> extract as Series
        if len(val_ranks) > 0:
            val_latest = val_ranks.iloc[:, 0]
        else:
            val_latest = pd.Series(dtype=float)

        # LLM ranks already come as a Series from rank_stocks()
        llm_latest = llm_ranks if len(llm_ranks) > 0 else pd.Series(dtype=float)

        # Find common tickers across all available factors
        common_tickers = mom_latest.dropna().index.intersection(
            val_latest.dropna().index
        )

        if len(common_tickers) == 0:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "weight",
                    "momentum_rank",
                    "value_rank",
                    "llm_rank",
                    "combined_score",
                ]
            )

        # Check if LLM has data for any common tickers
        llm_common = llm_latest.index.intersection(common_tickers)
        llm_has_data = len(llm_common) > 0 and llm_common.shape[0] > 0

        if llm_has_data:
            llm_weight_adj = self.llm_weight
            remaining_weight = self.momentum_weight + self.value_weight
        else:
            llm_weight_adj = 0.0
            remaining_weight = self.momentum_weight + self.value_weight

        # Build combined DataFrame
        combined = pd.DataFrame(index=common_tickers)
        combined["momentum_rank"] = mom_latest.loc[common_tickers]
        combined["value_rank"] = val_latest.loc[common_tickers]

        if llm_has_data:
            combined["llm_rank"] = llm_latest.loc[common_tickers]
        else:
            # Neutral rank for all tickers when LLM is unavailable
            combined["llm_rank"] = len(common_tickers) / 2.0

        # Normalize ranks to scores (lower rank = better = higher score)
        max_mom = combined["momentum_rank"].max()
        max_val = combined["value_rank"].max()
        max_llm = combined["llm_rank"].max()

        if pd.notna(max_mom) and max_mom > 0:
            combined["momentum_score"] = 1.0 - (combined["momentum_rank"] / max_mom)
        else:
            combined["momentum_score"] = 0.5

        if pd.notna(max_val) and max_val > 0:
            combined["value_score"] = 1.0 - (combined["value_rank"] / max_val)
        else:
            combined["value_score"] = 0.5

        if pd.notna(max_llm) and max_llm > 0:
            combined["llm_score"] = 1.0 - (combined["llm_rank"] / max_llm)
        else:
            combined["llm_score"] = 0.5

        # Weighted combination
        if remaining_weight > 0:
            normalized_mom = self.momentum_weight / remaining_weight
            normalized_val = self.value_weight / remaining_weight
        else:
            normalized_mom = 0.5
            normalized_val = 0.5

        combined["combined_score"] = (
            normalized_mom * combined["momentum_score"]
            + normalized_val * combined["value_score"]
            + llm_weight_adj * combined["llm_score"]
        )

        # Rank by combined score
        combined["combined_rank"] = combined["combined_score"].rank(ascending=False)

        # Select top N
        top_stocks = combined[combined["combined_rank"] <= self.top_n].copy()

        if len(top_stocks) == 0:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "weight",
                    "momentum_rank",
                    "value_rank",
                    "llm_rank",
                    "combined_score",
                ]
            )

        # Equal weight
        top_stocks["weight"] = 1.0 / len(top_stocks)

        result = top_stocks[
            [
                "momentum_rank",
                "value_rank",
                "llm_rank",
                "combined_score",
                "weight",
            ]
        ].copy()
        result.index.name = "ticker"
        result = result.reset_index()

        return result

    def get_signals(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
        news_data: dict[str, list[str]],
        financials_data: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Get trading signals for the three-factor strategy.

        Returns:
            DataFrame with dates as index, tickers as columns.
            Values: weight (positive = long, 0 = flat)
        """
        portfolio = self.compute_portfolio(
            prices, fundamentals, news_data, financials_data
        )

        if len(portfolio) == 0:
            return pd.DataFrame()

        # Create signal DataFrame
        tickers = portfolio["ticker"].tolist()
        weights = portfolio.set_index("ticker")["weight"]

        # For now, return static weights (will be rebalanced monthly)
        signals = pd.DataFrame(0.0, index=[0], columns=tickers)
        for ticker in tickers:
            signals.loc[0, ticker] = weights[ticker]

        return signals
