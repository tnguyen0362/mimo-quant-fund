# Combined momentum + value ranking
# - Equal-weight combination of factor ranks
# - Select top N stocks
# - Equal-weight allocation

import pandas as pd
import numpy as np
from .momentum import MomentumFactor
from .value import ValueFactor


class CombinedRanking:
    """
    Combined momentum + value factor ranking.

    Academic basis: Asness, Moskowitz, Pedersen (2013)
    "Value and Momentum Everywhere"

    Key insight: Value and momentum are negatively correlated,
    providing natural diversification.

    Implementation:
    - Combine momentum rank and value rank
    - Weight: 50% momentum + 50% value
    - Select top N stocks by combined rank
    - Equal-weight allocation
    """

    def __init__(
        self,
        momentum_weight: float = 0.5,
        value_weight: float = 0.5,
        top_n: int = 15,
    ):
        """
        Args:
            momentum_weight: Weight for momentum factor (default: 0.5)
            value_weight: Weight for value factor (default: 0.5)
            top_n: Number of stocks to hold (default: 15)
        """
        self.momentum_weight = momentum_weight
        self.value_weight = value_weight
        self.top_n = top_n

        self.momentum = MomentumFactor()
        self.value = ValueFactor()

    def compute_portfolio(
        self, prices: pd.DataFrame, fundamentals: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute combined portfolio.

        Args:
            prices: Price data for momentum ranking
            fundamentals: Fundamental data for value ranking

        Returns:
            DataFrame with columns:
                - ticker: Stock ticker
                - weight: Portfolio weight (equal-weight)
                - momentum_rank: Momentum rank
                - value_rank: Value rank
                - combined_rank: Combined rank
        """
        # Get factor ranks
        mom_ranks = self.momentum.rank_stocks(prices)
        val_ranks = self.value.rank_stocks(fundamentals)

        # Align on common tickers
        if mom_ranks.empty or val_ranks.empty:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "weight",
                    "momentum_rank",
                    "value_rank",
                    "combined_rank",
                ]
            )

        # Use the last valid cross-section (rows with non-NaN data)
        if len(mom_ranks) > 0:
            mom_valid = mom_ranks.dropna(how="all")
            mom_latest = mom_valid.iloc[-1] if len(mom_valid) > 0 else pd.Series(dtype=float)
        else:
            mom_latest = pd.Series(dtype=float)

        # Value ranks have tickers as index, extract the rank column as a Series
        if len(val_ranks) > 0:
            val_latest = val_ranks.iloc[:, 0]  # Series with ticker index
        else:
            val_latest = pd.Series(dtype=float)

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
                    "combined_rank",
                ]
            )

        # Combine ranks
        combined = pd.DataFrame(index=common_tickers)
        combined["momentum_rank"] = mom_latest.loc[common_tickers]
        combined["value_rank"] = val_latest.loc[common_tickers]

        # Normalize ranks to 0-1 (lower rank = better, so invert)
        max_mom = combined["momentum_rank"].max()
        max_val = combined["value_rank"].max()

        if pd.notna(max_mom) and max_mom > 0:
            combined["momentum_score"] = 1 - (combined["momentum_rank"] / max_mom)
        else:
            combined["momentum_score"] = 0.5

        if pd.notna(max_val) and max_val > 0:
            combined["value_score"] = 1 - (combined["value_rank"] / max_val)
        else:
            combined["value_score"] = 0.5

        # Combined score
        combined["combined_score"] = (
            self.momentum_weight * combined["momentum_score"]
            + self.value_weight * combined["value_score"]
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
                    "combined_rank",
                ]
            )

        # Equal weight
        top_stocks["weight"] = 1.0 / len(top_stocks)

        # Format output
        result = top_stocks[
            ["momentum_rank", "value_rank", "combined_rank", "combined_score", "weight"]
        ].copy()
        result.index.name = "ticker"
        result = result.reset_index()

        return result

    def get_signals(
        self, prices: pd.DataFrame, fundamentals: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get trading signals for the combined strategy.

        Returns:
            DataFrame with dates as index, tickers as columns.
            Values: weight (positive = long, 0 = flat)
        """
        portfolio = self.compute_portfolio(prices, fundamentals)

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
