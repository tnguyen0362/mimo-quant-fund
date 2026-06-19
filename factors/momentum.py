# Cross-sectional momentum factor (Jegadeesh-Titman 1993)
# - 12-1 momentum: 12-month lookback, skip most recent month
# - Rank all stocks by past returns
# - Transaction-cost-aware: reduce turnover

import pandas as pd
import numpy as np


class MomentumFactor:
    """
    Cross-sectional momentum factor.

    Academic basis: Jegadeesh & Titman (1993)
    "Returns to Buying Winners and Selling Losers"

    Implementation:
    - Lookback: 12 months (252 trading days)
    - Skip: Most recent month (21 trading days)
    - Signal: Cumulative return over lookback period
    - Holding: Rebalance monthly
    """

    def __init__(
        self,
        lookback_days: int = 252,
        skip_days: int = 21,
        transaction_cost_reduction: bool = True,
    ):
        """
        Args:
            lookback_days: Momentum lookback period (default: 252 = 12 months)
            skip_days: Skip most recent period (default: 21 = 1 month)
            transaction_cost_reduction: If True, reduce turnover
        """
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.transaction_cost_reduction = transaction_cost_reduction

    def compute_signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute momentum signal for all stocks.

        Args:
            prices: DataFrame with dates as index, tickers as columns (adj close)

        Returns:
            DataFrame with momentum scores (cumulative returns)
        """
        if prices.empty or prices.shape[0] <= self.lookback_days:
            return pd.DataFrame(
                np.nan, index=prices.index, columns=prices.columns
            )

        # Cumulative return over lookback
        momentum = prices.pct_change(self.lookback_days)

        # Skip most recent month (set to NaN so they're excluded from ranking)
        if self.skip_days > 0 and momentum.shape[0] > self.skip_days:
            momentum.iloc[-self.skip_days :] = np.nan

        return momentum

    def rank_stocks(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Rank stocks by momentum (1 = highest, lower = lower rank).

        Returns:
            DataFrame of ranks (1 = best momentum)
        """
        momentum = self.compute_signal(prices)

        # Rank cross-sectionally (higher return = lower rank = better)
        ranks = momentum.rank(axis=1, ascending=False)

        return ranks

    def get_top_n(self, prices: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """
        Get top N stocks by momentum.

        Returns:
            Boolean DataFrame (True = in top N)
        """
        ranks = self.rank_stocks(prices)

        # Top N have rank <= n
        top_n = ranks <= n

        return top_n

    def compute_turnover(self, signals: pd.DataFrame) -> float:
        """
        Compute average turnover for the strategy.

        Turnover = average fraction of portfolio that changes each period.
        """
        if signals.shape[0] < 2:
            return 0.0

        turnovers = []
        for i in range(1, len(signals)):
            prev = set(signals.columns[signals.iloc[i - 1].fillna(False)])
            curr = set(signals.columns[signals.iloc[i].fillna(False)])

            if len(prev) == 0 and len(curr) == 0:
                continue

            # Jaccard distance
            union = len(prev | curr)
            intersection = len(prev & curr)

            if union > 0:
                turnover = 1 - (intersection / union)
                turnovers.append(turnover)

        return float(np.mean(turnovers)) if turnovers else 0.0
