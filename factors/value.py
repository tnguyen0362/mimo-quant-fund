# Value factor (Fama-French 1993)
# - Book-to-market ratio (HML)
# - Earnings yield as alternative
# - Fama-French 2x3 sorting methodology (simplified for retail)

import pandas as pd
import numpy as np


class ValueFactor:
    """
    Value factor based on fundamental data.

    Academic basis: Fama & French (1993)
    "The Cross-Section of Expected Stock Returns"

    Implementation:
    - Primary: Book-to-market ratio (Price-to-Book inverse)
    - Secondary: Earnings yield (E/P ratio)
    - Combined: Average of normalized value scores
    """

    def __init__(
        self,
        use_book_to_market: bool = True,
        use_earnings_yield: bool = True,
    ):
        """
        Args:
            use_book_to_market: Use P/B ratio (inverse)
            use_earnings_yield: Use P/E ratio (inverse)
        """
        self.use_book_to_market = use_book_to_market
        self.use_earnings_yield = use_earnings_yield

    def compute_signal(self, fundamentals: pd.DataFrame) -> pd.DataFrame:
        """
        Compute value signal from fundamental data.

        Args:
            fundamentals: DataFrame with columns:
                - pb_ratio: Price-to-Book
                - pe_ratio: Price-to-Earnings
                (from FundamentalData.get_fundamentals())

        Returns:
            DataFrame with value scores (higher = more value)
        """
        scores = pd.DataFrame(index=fundamentals.index)

        # Book-to-market (inverse of P/B)
        if self.use_book_to_market and "pb_ratio" in fundamentals.columns:
            pb = fundamentals["pb_ratio"].replace(0, np.nan)
            scores["book_to_market"] = 1.0 / pb

        # Earnings yield (inverse of P/E)
        if self.use_earnings_yield and "pe_ratio" in fundamentals.columns:
            pe = fundamentals["pe_ratio"].replace(0, np.nan)
            # Filter out negative P/E (unprofitable companies)
            pe = pe.where(pe > 0, np.nan)
            scores["earnings_yield"] = 1.0 / pe

        # Normalize each component to 0-1
        for col in scores.columns:
            min_val = scores[col].min()
            max_val = scores[col].max()
            if pd.notna(max_val) and pd.notna(min_val) and max_val > min_val:
                scores[col] = (scores[col] - min_val) / (max_val - min_val)
            else:
                scores[col] = 0.5

        # Average of available components
        if scores.empty:
            return pd.DataFrame({"value_score": pd.Series(dtype=float)})
        scores["value_score"] = scores.mean(axis=1)

        return scores[["value_score"]]

    def rank_stocks(self, fundamentals: pd.DataFrame) -> pd.DataFrame:
        """
        Rank stocks by value (1 = highest value).

        Returns:
            DataFrame of ranks
        """
        signal = self.compute_signal(fundamentals)

        # Rank (higher value score = lower rank number = better)
        ranks = signal["value_score"].rank(ascending=False)

        return pd.DataFrame({"value_rank": ranks})

    def get_top_n(self, fundamentals: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """
        Get top N value stocks.

        Returns:
            Boolean Series (True = in top N)
        """
        ranks = self.rank_stocks(fundamentals)

        top_n = ranks["value_rank"] <= n

        return top_n
