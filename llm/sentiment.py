# MiMo 2.5 LLM Sentiment Analysis Engine
# 
# The LLM is the BRAIN. It:
# 1. Reads news headlines + earnings data for each stock
# 2. Scores sentiment (-1.0 to +1.0)
# 3. Provides a confidence level (0-1)
# 4. Explains its reasoning in plain English
#
# Uses OpenRouter API to access MiMo 2.5 (~$0.15/M input, $1.20/M output)
# Falls back to a simple keyword scorer if API is unavailable.

import os
import json
import time
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import pandas as pd
import numpy as np

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


@dataclass
class SentimentResult:
    """LLM sentiment analysis result for a single stock."""
    ticker: str
    sentiment: float        # -1.0 (very bearish) to +1.0 (very bullish)
    confidence: float       # 0.0 to 1.0
    reasoning: str          # LLM's explanation
    source: str             # "llm" or "fallback"
    raw_response: str = ""  # Full LLM response for debugging


class MiMoSentiment:
    """
    MiMo 2.5 sentiment analysis engine.
    
    Architecture:
    - Primary: MiMo 2.5 via OpenRouter API
    - Fallback: Simple keyword-based sentiment (if API unavailable)
    
    The LLM receives:
    - Stock ticker and company name
    - Recent news headlines (last 7 days)
    - Key financial metrics (P/E, market cap, sector)
    - Earnings data if available
    
    The LLM returns:
    - Sentiment score: -1.0 to +1.0
    - Confidence: 0-1
    - Brief reasoning (1-2 sentences)
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "openrouter/xiaomi/mimo-v2.5",
                 use_fallback: bool = True,
                 cache_dir: str = "data/cache/llm"):
        """
        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            model: Model identifier (default: MiMo 2.5 via OpenRouter)
            use_fallback: Use keyword fallback if API unavailable
            cache_dir: Directory to cache LLM responses
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model
        self.use_fallback = use_fallback
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Simple sentiment keywords for fallback
        self.positive_words = {
            "beat", "beats", "surpass", "upgrade", "buy", "bullish",
            "strong", "growth", "profit", "revenue", "record", "gain",
            "outperform", "positive", "optimistic", "rally", "surge"
        }
        self.negative_words = {
            "miss", "misses", "downgrade", "sell", "bearish", "weak",
            "loss", "decline", "drop", "fall", "crash", "warning",
            "underperform", "negative", "pessimistic", "recession", "fear"
        }
    
    def analyze_stock(self, ticker: str, 
                      news_headlines: list[str],
                      financials: Optional[dict] = None,
                      earnings_data: Optional[dict] = None) -> SentimentResult:
        """
        Analyze sentiment for a single stock using MiMo 2.5.
        
        Args:
            ticker: Stock ticker symbol
            news_headlines: List of recent news headlines
            financials: Optional dict with P/E, market_cap, sector, etc.
            earnings_data: Optional dict with recent earnings info
        
        Returns:
            SentimentResult with score, confidence, and reasoning
        """
        # Check cache first
        cache_key = f"{ticker}_{hash(str(news_headlines[:5]))}"
        cached = self._load_cache(cache_key)
        if cached:
            return cached
        
        # Try LLM analysis
        if self.api_key and HAS_HTTPX:
            try:
                result = self._llm_analyze(ticker, news_headlines, 
                                           financials, earnings_data)
                self._save_cache(cache_key, result)
                return result
            except Exception as e:
                print(f"LLM analysis failed for {ticker}: {e}")
                if not self.use_fallback:
                    raise
        
        # Fallback to keyword analysis
        result = self._keyword_analyze(ticker, news_headlines)
        self._save_cache(cache_key, result)
        return result
    
    def analyze_batch(self, tickers: list[str],
                      news_data: dict[str, list[str]],
                      financials_data: Optional[dict] = None) -> pd.DataFrame:
        """
        Batch analyze sentiment for multiple stocks.
        
        Args:
            tickers: List of stock tickers
            news_data: Dict mapping ticker -> list of news headlines
            financials_data: Optional dict mapping ticker -> financials dict
        
        Returns:
            DataFrame with columns: ticker, sentiment, confidence, reasoning
        """
        results = []
        
        for ticker in tickers:
            news = news_data.get(ticker, [])
            fins = financials_data.get(ticker) if financials_data else None
            
            result = self.analyze_stock(ticker, news, fins)
            results.append({
                "ticker": ticker,
                "sentiment": result.sentiment,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "source": result.source,
            })
            
            # Rate limiting: 1 request per second for free tier
            time.sleep(0.1)
        
        return pd.DataFrame(results)
    
    def _llm_analyze(self, ticker: str, news_headlines: list[str],
                     financials: Optional[dict] = None,
                     earnings_data: Optional[dict] = None) -> SentimentResult:
        """Call MiMo 2.5 via OpenRouter for sentiment analysis."""
        
        # Build the prompt
        news_text = "\n".join(f"- {h}" for h in news_headlines[:10]) if news_headlines else "No recent news available."
        
        financials_text = ""
        if financials:
            financials_text = f"""
Financial Data:
- P/E Ratio: {financials.get('pe_ratio', 'N/A')}
- Market Cap: {financials.get('market_cap', 'N/A')}
- Sector: {financials.get('sector', 'N/A')}
"""
        
        earnings_text = ""
        if earnings_data:
            earnings_text = f"""
Recent Earnings:
- EPS Estimate: {earnings_data.get('eps_estimate', 'N/A')}
- EPS Actual: {earnings_data.get('eps_actual', 'N/A')}
- Revenue Estimate: {earnings_data.get('revenue_estimate', 'N/A')}
- Revenue Actual: {earnings_data.get('revenue_actual', 'N/A')}
"""
        
        prompt = f"""You are a quantitative investment analyst. Analyze the sentiment for {ticker} stock.

Recent News:
{news_text}
{financials_text}
{earnings_text}

Respond in EXACTLY this JSON format (no other text):
{{
    "sentiment": <float from -1.0 to 1.0>,
    "confidence": <float from 0.0 to 1.0>,
    "reasoning": "<1-2 sentence explanation>"
}}

Where:
- sentiment: -1.0 = extremely bearish, 0.0 = neutral, 1.0 = extremely bullish
- confidence: How confident you are in this assessment (0.0 to 1.0)
- reasoning: Brief explanation of your assessment

Only output the JSON, nothing else."""

        # Call OpenRouter API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://trading-system.local",
            "X-Title": "Quant Trading System",
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,  # Low temperature for consistent analysis
            "max_tokens": 200,
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
        
        # Parse response
        try:
            # Handle potential markdown code blocks
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            parsed = json.loads(content)
            
            return SentimentResult(
                ticker=ticker,
                sentiment=float(parsed.get("sentiment", 0.0)),
                confidence=float(parsed.get("confidence", 0.5)),
                reasoning=parsed.get("reasoning", "No reasoning provided"),
                source="llm",
                raw_response=content,
            )
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse LLM response for {ticker}: {e}")
            print(f"Raw response: {content}")
            return self._keyword_analyze(ticker, news_headlines)
    
    def _keyword_analyze(self, ticker: str, 
                         news_headlines: list[str]) -> SentimentResult:
        """Fallback keyword-based sentiment analysis."""
        if not news_headlines:
            return SentimentResult(
                ticker=ticker,
                sentiment=0.0,
                confidence=0.1,
                reasoning="No news available for analysis",
                source="fallback",
            )
        
        # Count positive and negative words
        pos_count = 0
        neg_count = 0
        total_words = 0
        
        for headline in news_headlines:
            words = headline.lower().split()
            total_words += len(words)
            for word in words:
                if word in self.positive_words:
                    pos_count += 1
                elif word in self.negative_words:
                    neg_count += 1
        
        # Compute sentiment
        if total_words > 0:
            sentiment = (pos_count - neg_count) / total_words
            sentiment = max(-1.0, min(1.0, sentiment * 10))  # Scale up
        else:
            sentiment = 0.0
        
        # Confidence based on signal strength and data quantity
        signal_strength = abs(sentiment)
        data_quantity = min(len(news_headlines) / 5, 1.0)  # More news = more confidence
        confidence = signal_strength * 0.7 + data_quantity * 0.3
        
        # Build reasoning
        if sentiment > 0.2:
            reasoning = f"Positive sentiment from {pos_count} bullish keywords across {len(news_headlines)} headlines"
        elif sentiment < -0.2:
            reasoning = f"Negative sentiment from {neg_count} bearish keywords across {len(news_headlines)} headlines"
        else:
            reasoning = f"Neutral sentiment from balanced keywords across {len(news_headlines)} headlines"
        
        return SentimentResult(
            ticker=ticker,
            sentiment=sentiment,
            confidence=confidence,
            reasoning=reasoning,
            source="fallback",
        )
    
    def _load_cache(self, key: str) -> Optional[SentimentResult]:
        """Load cached sentiment result."""
        cache_file = self.cache_dir / f"{hash(key)}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                return SentimentResult(**data)
            except Exception:
                pass
        return None
    
    def _save_cache(self, key: str, result: SentimentResult):
        """Save sentiment result to cache."""
        cache_file = self.cache_dir / f"{hash(key)}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump({
                    "ticker": result.ticker,
                    "sentiment": result.sentiment,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "source": result.source,
                    "raw_response": result.raw_response,
                }, f, indent=2)
        except Exception:
            pass
