# Fundamental data for value factor scoring
# Uses yfinance for basic fundamentals (P/E, P/B, P/S, EV/EBITDA)
# Cache results to avoid repeated API calls

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from config.settings import config

logger = logging.getLogger(__name__)

# SEC EDGAR helpers (kept for reference / advanced use-cases)
HEADERS = {
    "User-Agent": "TradingSystem/1.0 (contact@example.com)",
    "Accept-Encoding": "gzip, deflate",
}

TICKER_TO_CIK = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "NVDA": "0001045810",
    "META": "0001326801",
    "TSLA": "0001318605",
    "BRK-B": "0001067983",
    "JPM": "0000019617",
    "V": "0001403172",
    "JNJ": "0000200406",
    "UNH": "0002010971",
}


class FundamentalData:
    """Fetch fundamental data for value-factor scoring.

    Primary interface uses ``yfinance`` to pull point-in-time ratios
    (P/E, P/B, P/S, EV/EBITDA, dividend yield, market cap).  A secondary
    SEC EDGAR interface is retained for deeper accounting data.
    """

    def __init__(self, cache_dir: str | Path | None = None):
        if cache_dir is None:
            self.cache_dir = config.DATA_DIR / "cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # yfinance-based value-factor data
    # ------------------------------------------------------------------

    def get_fundamentals(self, tickers: list[str]) -> pd.DataFrame:
        """Fetch fundamental data for value factor scoring.

        Returns a DataFrame indexed by ticker with columns:

        * ``pe_ratio`` – Price-to-Earnings (trailing twelve months)
        * ``pb_ratio`` – Price-to-Book
        * ``ps_ratio`` – Price-to-Sales (trailing twelve months)
        * ``ev_ebitda`` – Enterprise Value to EBITDA
        * ``dividend_yield`` – Dividend yield (decimal)
        * ``market_cap`` – Market capitalization (USD)

        Results are cached for 30 days to avoid excessive API calls.
        """
        cache_file = self.cache_dir / "fundamentals.parquet"

        # Check cache (refresh monthly)
        if cache_file.exists():
            try:
                cache_age = datetime.now() - datetime.fromtimestamp(
                    cache_file.stat().st_mtime
                )
                if cache_age < timedelta(days=30):
                    cached = pd.read_parquet(cache_file)
                    missing = set(tickers) - set(cached.index)
                    if not missing:
                        return cached.loc[tickers].copy()
            except (OSError, Exception) as exc:
                logger.warning("Cache read failed, re-fetching: %s", exc)

        fundamentals: dict[str, dict] = {}

        for i, ticker in enumerate(tickers):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                fundamentals[ticker] = {
                    "pe_ratio": info.get("trailingPE"),
                    "pb_ratio": info.get("priceToBook"),
                    "ps_ratio": info.get("priceToSalesTrailing12Months"),
                    "ev_ebitda": info.get("enterpriseToEbitda"),
                    "dividend_yield": info.get("dividendYield"),
                    "market_cap": info.get("marketCap"),
                }

                # Rate-limit: pause every 10 tickers
                if (i + 1) % 10 == 0:
                    time.sleep(1)

            except Exception as exc:
                logger.error("Error fetching fundamentals for %s: %s", ticker, exc)
                fundamentals[ticker] = {
                    k: None
                    for k in [
                        "pe_ratio",
                        "pb_ratio",
                        "ps_ratio",
                        "ev_ebitda",
                        "dividend_yield",
                        "market_cap",
                    ]
                }

        df = pd.DataFrame(fundamentals).T
        df.index.name = "ticker"

        # Cache to disk
        try:
            df.to_parquet(cache_file)
        except Exception as exc:
            logger.warning("Failed to cache fundamentals: %s", exc)

        logger.info("Fetched fundamentals for %d tickers", len(df))
        return df

    # ------------------------------------------------------------------
    # SEC EDGAR helpers (backward-compatible)
    # ------------------------------------------------------------------

    def get_company_facts(self, ticker: str) -> dict:
        """Get company facts from SEC EDGAR."""
        import requests

        cik = TICKER_TO_CIK.get(ticker, "")
        if not cik:
            logger.warning("No CIK mapping for %s", ticker)
            return {}

        try:
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data.get("facts", {}).get("us-gaap", {})
        except Exception as exc:
            logger.error("Error fetching EDGAR facts for %s: %s", ticker, exc)
            return {}

    def get_key_metrics(self, ticker: str) -> dict:
        """Extract key financial metrics from SEC EDGAR filings."""
        facts = self.get_company_facts(ticker)
        if not facts:
            return {}

        metrics: dict = {}
        metric_mapping = {
            "revenue": [
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet",
            ],
            "net_income": ["NetIncomeLoss", "ProfitLoss"],
            "total_assets": ["Assets"],
            "total_liabilities": ["Liabilities"],
            "stockholders_equity": ["StockholdersEquity"],
            "cash": ["CashAndCashEquivalentsAtCarryingValue"],
            "operating_income": ["OperatingIncomeLoss"],
            "gross_profit": ["GrossProfit"],
        }

        for metric_name, possible_keys in metric_mapping.items():
            for key in possible_keys:
                if key in facts:
                    values = facts[key].get("units", {}).get("USD", [])
                    annual_values = [v for v in values if v.get("fp") == "FY"]
                    if annual_values:
                        most_recent = max(annual_values, key=lambda x: x.get("end", ""))
                        metrics[metric_name] = most_recent.get("val")
                        break

        return metrics

    def calculate_ratios(self, metrics: dict) -> dict:
        """Calculate financial ratios from raw accounting metrics."""
        ratios: dict = {}

        if "net_income" in metrics and "revenue" in metrics:
            if metrics["revenue"] and metrics["revenue"] > 0:
                ratios["profit_margin"] = metrics["net_income"] / metrics["revenue"]

        if "total_liabilities" in metrics and "stockholders_equity" in metrics:
            if metrics["stockholders_equity"] and metrics["stockholders_equity"] > 0:
                ratios["debt_to_equity"] = (
                    metrics["total_liabilities"] / metrics["stockholders_equity"]
                )

        if "net_income" in metrics and "stockholders_equity" in metrics:
            if metrics["stockholders_equity"] and metrics["stockholders_equity"] > 0:
                ratios["roe"] = metrics["net_income"] / metrics["stockholders_equity"]

        if "net_income" in metrics and "total_assets" in metrics:
            if metrics["total_assets"] and metrics["total_assets"] > 0:
                ratios["roa"] = metrics["net_income"] / metrics["total_assets"]

        if "cash" in metrics and "total_liabilities" in metrics:
            if metrics["total_liabilities"] and metrics["total_liabilities"] > 0:
                ratios["cash_to_debt"] = metrics["cash"] / metrics["total_liabilities"]

        return ratios
