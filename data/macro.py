import requests
import pandas as pd
from datetime import datetime, timedelta
import logging

from config.settings import config

logger = logging.getLogger(__name__)


class MacroData:
    """Fetch macroeconomic data from FRED API."""

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    SERIES = {
        "GDP": "GDP",
        "UNEMPLOYMENT": "UNRATE",
        "CPI": "CPIAUCSL",
        "FED_FUNDS_RATE": "FEDFUNDS",
        "TREASURY_10Y": "DGS10",
        "TREASURY_2Y": "DGS2",
        "YIELD_CURVE": "T10Y2Y",
        "VIX": "VIXCLS",
        "INDUSTRIAL_PRODUCTION": "INDPRO",
        "RETAIL_SALES": "RSAFS",
    }

    def __init__(self):
        self.api_key = config.FRED_API_KEY
        if not self.api_key:
            logger.warning("FRED_API_KEY not set. Macro data will not be available.")

    def get_series(self, series_id: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch a single FRED series."""
        if not self.api_key:
            logger.error("FRED_API_KEY not configured")
            return pd.DataFrame()

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date or (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d"),
            "observation_end": end_date or datetime.now().strftime("%Y-%m-%d"),
        }

        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            observations = data.get("observations", [])
            if not observations:
                logger.warning(f"No observations for {series_id}")
                return pd.DataFrame()

            df = pd.DataFrame(observations)
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.set_index("date")
            df = df[["value"]].rename(columns={"value": series_id.lower()})

            logger.info(f"Fetched {len(df)} observations for {series_id}")
            return df

        except Exception as e:
            logger.error(f"Error fetching {series_id}: {e}")
            return pd.DataFrame()

    def get_all_series(self) -> pd.DataFrame:
        """Fetch all configured macro series."""
        dfs = []
        for name, series_id in self.SERIES.items():
            df = self.get_series(series_id)
            if not df.empty:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, axis=1)
        result = result.ffill()

        logger.info(f"Merged {len(dfs)} macro series")
        return result

    def get_yield_curve_spread(self) -> pd.DataFrame:
        return self.get_series("T10Y2Y")

    def get_vix(self) -> pd.DataFrame:
        return self.get_series("VIXCLS")
