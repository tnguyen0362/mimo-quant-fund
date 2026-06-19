import logging

logger = logging.getLogger(__name__)


class FundamentalFeatures:
    """Calculate fundamental features from financial data."""

    def calculate_fundamental_score(self, metrics: dict, ratios: dict) -> float:
        """Calculate a composite fundamental score (0-1)."""
        score = 0.0
        weights = {
            "profit_margin": 0.25,
            "roe": 0.25,
            "debt_to_equity": 0.20,
            "revenue_growth": 0.15,
            "cash_to_debt": 0.15,
        }

        # Profit Margin (higher is better)
        if "profit_margin" in ratios:
            pm = ratios["profit_margin"]
            if pm > 0.20:
                score += weights["profit_margin"]
            elif pm > 0.10:
                score += weights["profit_margin"] * 0.7
            elif pm > 0.05:
                score += weights["profit_margin"] * 0.4

        # ROE (higher is better)
        if "roe" in ratios:
            roe = ratios["roe"]
            if roe > 0.20:
                score += weights["roe"]
            elif roe > 0.10:
                score += weights["roe"] * 0.7
            elif roe > 0.05:
                score += weights["roe"] * 0.4

        # Debt-to-Equity (lower is better)
        if "debt_to_equity" in ratios:
            dte = ratios["debt_to_equity"]
            if dte < 0.5:
                score += weights["debt_to_equity"]
            elif dte < 1.0:
                score += weights["debt_to_equity"] * 0.7
            elif dte < 2.0:
                score += weights["debt_to_equity"] * 0.4

        # Revenue Growth (use profit margin as proxy)
        if ratios.get("profit_margin", 0) > 0.15:
            score += weights["revenue_growth"]

        # Cash-to-Debt (higher is better)
        if "cash_to_debt" in ratios:
            ctd = ratios["cash_to_debt"]
            if ctd > 0.5:
                score += weights["cash_to_debt"]
            elif ctd > 0.2:
                score += weights["cash_to_debt"] * 0.7
            elif ctd > 0.1:
                score += weights["cash_to_debt"] * 0.4

        logger.info(f"Calculated fundamental score: {score:.2f}")
        return min(score, 1.0)

    def classify_stock(self, score: float) -> str:
        if score >= 0.8:
            return "STRONG_BUY"
        elif score >= 0.6:
            return "BUY"
        elif score >= 0.4:
            return "HOLD"
        elif score >= 0.2:
            return "SELL"
        else:
            return "STRONG_SELL"
