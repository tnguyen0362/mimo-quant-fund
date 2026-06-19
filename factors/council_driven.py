"""
Council-Driven Stock Picker
===========================
Instead of blending council sentiment with momentum/value as a third factor,
the council becomes the PRIMARY stock picker:

1. Pre-filter: momentum + value narrow to top 20 candidates
2. Council analyzes each candidate with REAL financial data
3. Council votes BUY / HOLD / SELL with conviction
4. Only BUY stocks with majority council agreement
5. Rank final picks by momentum + value

This makes the LLM the decision maker, not a passenger.
"""

import os
import json
import time
import logging
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import pandas as pd
import numpy as np

from llm.council import LLMCouncil, CouncilResult, CouncilVote
from factors.momentum import MomentumFactor
from factors.value import ValueFactor

logger = logging.getLogger(__name__)


@dataclass
class StockPick:
    """A single stock pick from the council."""
    ticker: str
    action: str           # "BUY", "HOLD", "SELL"
    conviction: float     # 0.0 to 1.0 (how strong the conviction)
    sentiment: float      # -1.0 to +1.0
    confidence: float     # 0.0 to 1.0
    reasoning: str
    num_votes: int
    momentum_rank: float  # Normalized momentum rank (0-1)
    value_rank: float     # Normalized value rank (0-1)
    composite_score: float  # Final ranking score


class CouncilStockPicker:
    """
    Council-driven stock picker.
    
    Architecture:
    1. Pre-filter: momentum + value to narrow universe
    2. Council analyzes candidates with real financial data
    3. Council votes BUY/HOLD/SELL
    4. Final ranking by composite score (council conviction + momentum + value)
    """
    
    def __init__(self,
                 council: Optional[LLMCouncil] = None,
                 api_key: Optional[str] = None,
                 prefilter_top_n: int = 20,
                 final_top_n: int = 15,
                 momentum_weight: float = 0.4,
                 value_weight: float = 0.3,
                 council_weight: float = 0.3,
                 min_council_agreement: float = 0.5,
                 cache_dir: str = "data/cache/picker"):
        """
        Args:
            council: LLMCouncil instance (created if None)
            api_key: OpenRouter API key
            prefilter_top_n: How many candidates to send to council
            final_top_n: How many stocks to actually buy
            momentum_weight: Weight for momentum in final ranking
            value_weight: Weight for value in final ranking
            council_weight: Weight for council conviction in final ranking
            min_council_agreement: Minimum agreement to count as BUY
        """
        self.council = council or LLMCouncil(
            api_key=api_key,
            max_models_per_stock=2,  # Use only the 2 most reliable models
        )
        self.prefilter_top_n = prefilter_top_n
        self.final_top_n = final_top_n
        self.momentum_weight = momentum_weight
        self.value_weight = value_weight
        self.council_weight = council_weight
        self.min_council_agreement = min_council_agreement
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def pick_stocks(self,
                    prices: pd.DataFrame,
                    fundamentals: pd.DataFrame,
                    rebalance_date: pd.Timestamp) -> list[StockPick]:
        """
        Main entry point: pick stocks for a rebalance date.
        
        Args:
            prices: Full price history up to rebalance_date
            fundamentals: Fundamental data for all stocks
            rebalance_date: The date we're making decisions for
            
        Returns:
            List of StockPick objects, sorted by composite_score descending
        """
        # Step 1: Pre-filter with momentum + value
        candidates = self._prefilter(prices, fundamentals, rebalance_date)
        
        if len(candidates) == 0:
            return []
        
        # Step 2: Build financial context for each candidate
        financial_context = self._build_financial_context(
            prices, fundamentals, candidates, rebalance_date
        )
        
        # Step 3: Council deliberates on each candidate
        council_picks = self._council_deliberate(
            candidates, financial_context
        )
        
        # Step 4: Filter to BUY recommendations with sufficient agreement
        buy_picks = [
            p for p in council_picks
            if p.action == "BUY" and p.confidence >= self.min_council_agreement
        ]
        
        if len(buy_picks) == 0:
            # Fallback: if council rejects everything, take top momentum stocks
            buy_picks = [
                p for p in council_picks
                if p.action != "SELL"
            ][:self.final_top_n]
        
        # Step 5: Sort by composite score and take top N
        buy_picks.sort(key=lambda x: x.composite_score, reverse=True)
        
        return buy_picks[:self.final_top_n]
    
    def _prefilter(self, prices: pd.DataFrame, 
                   fundamentals: pd.DataFrame,
                   date: pd.Timestamp) -> list[str]:
        """
        Pre-filter universe to top N candidates using momentum + value.
        This saves API calls by only sending promising stocks to the council.
        """
        prices_to_date = prices.loc[:date]
        
        if len(prices_to_date) < 252:
            return prices.columns.tolist()[:self.prefilter_top_n]
        
        # Momentum
        mom = MomentumFactor(lookback_days=252, skip_days=21)
        mom_signals = mom.compute_signal(prices_to_date)
        last_mom = mom_signals.iloc[-1] if len(mom_signals) > 0 else pd.Series(dtype=float)
        
        # Value
        val = ValueFactor()
        val_signals = val.compute_signal(fundamentals)
        val_scores = val_signals.set_index("ticker")["value_score"] if "ticker" in val_signals.columns else pd.Series(dtype=float)
        
        # Combine ranks
        combined = pd.DataFrame(index=prices.columns)
        combined["mom_rank"] = last_mom.reindex(prices.columns).rank(pct=True)
        combined["val_rank"] = val_scores.reindex(prices.columns).rank(pct=True)
        combined["combined"] = 0.5 * combined["mom_rank"].fillna(0.5) + 0.5 * combined["val_rank"].fillna(0.5)
        
        # Top candidates
        top = combined.nlargest(self.prefilter_top_n, "combined")
        
        return top.index.tolist()
    
    def _build_financial_context(self, prices: pd.DataFrame,
                                 fundamentals: pd.DataFrame,
                                 tickers: list[str],
                                 date: pd.Timestamp) -> dict[str, dict]:
        """
        Build rich financial context for the council to analyze.
        Includes price history, technicals, and fundamentals.
        """
        prices_to_date = prices.loc[:date]
        context = {}
        
        for ticker in tickers:
            if ticker not in prices_to_date.columns:
                continue
            
            stock_prices = prices_to_date[ticker].dropna()
            
            if len(stock_prices) < 20:
                context[ticker] = {"ticker": ticker, "insufficient_data": True}
                continue
            
            # Recent price action
            last_price = stock_prices.iloc[-1]
            ret_1d = (stock_prices.iloc[-1] / stock_prices.iloc[-2] - 1) if len(stock_prices) >= 2 else 0
            ret_5d = (stock_prices.iloc[-1] / stock_prices.iloc[-5] - 1) if len(stock_prices) >= 5 else 0
            ret_1m = (stock_prices.iloc[-1] / stock_prices.iloc[-21] - 1) if len(stock_prices) >= 21 else 0
            ret_3m = (stock_prices.iloc[-1] / stock_prices.iloc[-63] - 1) if len(stock_prices) >= 63 else 0
            ret_6m = (stock_prices.iloc[-1] / stock_prices.iloc[-126] - 1) if len(stock_prices) >= 126 else 0
            ret_12m = (stock_prices.iloc[-1] / stock_prices.iloc[-252] - 1) if len(stock_prices) >= 252 else 0
            
            # Volatility
            daily_vol = stock_prices.pct_change().iloc[-60:].std() * np.sqrt(252) if len(stock_prices) >= 60 else None
            
            # 52-week high/low
            high_52w = stock_prices.iloc[-252:].max() if len(stock_prices) >= 252 else stock_prices.max()
            low_52w = stock_prices.iloc[-252:].min() if len(stock_prices) >= 252 else stock_prices.min()
            pct_from_high = (last_price / high_52w - 1) if high_52w > 0 else 0
            
            # Fundamentals
            fins = {}
            if fundamentals is not None and ticker in fundamentals.index:
                row = fundamentals.loc[ticker]
                fins = {
                    "pe_ratio": row.get("pe_ratio", None),
                    "pb_ratio": row.get("pb_ratio", None),
                    "dividend_yield": row.get("dividend_yield", None),
                    "market_cap": row.get("market_cap", None),
                }
            
            context[ticker] = {
                "ticker": ticker,
                "last_price": round(last_price, 2),
                "returns": {
                    "1d": round(ret_1d * 100, 2),
                    "5d": round(ret_5d * 100, 2),
                    "1m": round(ret_1m * 100, 2),
                    "3m": round(ret_3m * 100, 2),
                    "6m": round(ret_6m * 100, 2),
                    "12m": round(ret_12m * 100, 2),
                },
                "volatility_annual": round(daily_vol * 100, 1) if daily_vol else None,
                "pct_from_52w_high": round(pct_from_high * 100, 1),
                "fundamentals": fins,
            }
        
        return context
    
    def _council_deliberate(self, candidates: list[str],
                            financial_context: dict) -> list[StockPick]:
        """
        Council analyzes ALL candidates in ONE API call per model.
        Uses the batch prompt from council.deliberate_all_at_once().
        """
        # Filter out insufficient data
        valid_tickers = [t for t in candidates 
                         if not financial_context.get(t, {}).get("insufficient_data")]
        
        if not valid_tickers:
            return []
        
        # Build news_data and financials_data for batch call
        # The batch method expects: news_data = {ticker: [headlines...]}
        # and financials_data = {ticker: {pe_ratio, market_cap, ...}}
        # But our financial_context has richer data (returns, volatility, etc.)
        # So we'll use a different approach: build the batch prompt ourselves
        # using the picker's custom format, then send it via council's model querying
        
        # Build the batch prompt with ALL stocks
        batch_prompt = self._build_batch_picker_prompt(valid_tickers, financial_context)
        
        # Send to council models (ONE call per model)
        # Use council's internal model querying with our custom prompt
        votes_per_model = []
        for model_key, model_config in list(self.council.models.items())[:self.council.max_models_per_stock]:
            vote = self.council._do_query_batch(model_key, model_config, batch_prompt)
            if vote:
                votes_per_model.append(vote)
        
        # Parse batch response — each model returns a JSON array
        # We need to extract per-ticker results from each model's response
        ticker_results = self._parse_batch_results(valid_tickers, votes_per_model, financial_context)
        
        # Convert to StockPick objects
        picks = []
        for ticker in valid_tickers:
            if ticker in ticker_results:
                result = ticker_results[ticker]
                ctx = financial_context.get(ticker, {})
                pick = self._result_to_pick(ticker, result, ctx)
                picks.append(pick)
        
        return picks
    
    def _build_picker_prompt(self, ctx: dict) -> str:
        """
        Build a chain-of-thought prompt that forces the model to ANALYZE
        the data first, then DECIDE. This prevents the common failure mode
        of returning sentiment=0.0.
        """
        ticker = ctx.get("ticker", "UNKNOWN")
        price = ctx.get("last_price", "N/A")
        returns = ctx.get("returns", {})
        vol = ctx.get("volatility_annual", "N/A")
        pct_high = ctx.get("pct_from_52w_high", "N/A")
        fins = ctx.get("fundamentals", {})
        
        fin_text = ""
        if fins.get("pe_ratio"):
            fin_text += f"- P/E Ratio: {fins['pe_ratio']:.1f}\n"
        if fins.get("pb_ratio"):
            fin_text += f"- P/B Ratio: {fins['pb_ratio']:.1f}\n"
        if fins.get("dividend_yield"):
            fin_text += f"- Dividend Yield: {fins['dividend_yield']:.2%}\n"
        if fins.get("market_cap"):
            mc = fins["market_cap"]
            if mc > 1e12:
                fin_text += f"- Market Cap: ${mc/1e12:.1f}T\n"
            elif mc > 1e9:
                fin_text += f"- Market Cap: ${mc/1e9:.1f}B\n"
        
        return f"""You are a quantitative analyst analyzing {ticker}.

DATA:
- Current Price: ${price}
- 1-day return: {returns.get('1d', 'N/A')}%
- 5-day return: {returns.get('5d', 'N/A')}%
- 1-month return: {returns.get('1m', 'N/A')}%
- 3-month return: {returns.get('3m', 'N/A')}%
- 6-month return: {returns.get('6m', 'N/A')}%
- 12-month return: {returns.get('12m', 'N/A')}%
- Annualized Volatility: {vol}%
- Distance from 52-week High: {pct_high}%

{f"FUNDAMENTALS:\n{fin_text}" if fin_text else "No fundamental data available."}

STEP 1 - ANALYZE: Look at the returns and volatility. What story do the numbers tell?
- Is the stock trending up or down?
- Is volatility normal or extreme?
- How far from 52-week high?

STEP 2 - DECIDE: Based on your analysis, pick ONE action:
- BUY: Stock has strong momentum, reasonable valuation, manageable risk
- HOLD: Stock is neutral or you're unsure
- SELL: Stock has weak momentum, expensive valuation, or excessive risk

You MUST return a non-zero sentiment. Strong opinions should have high absolute sentiment values.

        JSON only:
        {{"action": "BUY"|"HOLD"|"SELL", "conviction": 0.1-1.0, "sentiment": -0.9 to 0.9 (NEVER 0.0), "confidence": 0.1-1.0, "reasoning": "1 sentence"}}"""
    
    def _build_batch_picker_prompt(self, tickers: list[str], 
                                    financial_context: dict) -> str:
        """Build a batch prompt for ALL stocks at once."""
        stock_blocks = []
        for ticker in tickers:
            ctx = financial_context.get(ticker, {})
            if ctx.get("insufficient_data"):
                continue
            
            price = ctx.get("last_price", "N/A")
            returns = ctx.get("returns", {})
            vol = ctx.get("volatility_annual", "N/A")
            pct_high = ctx.get("pct_from_52w_high", "N/A")
            fins = ctx.get("fundamentals", {})
            
            fin_lines = []
            if fins.get("pe_ratio"): fin_lines.append(f"P/E: {fins['pe_ratio']:.1f}")
            if fins.get("pb_ratio"): fin_lines.append(f"P/B: {fins['pb_ratio']:.1f}")
            if fins.get("dividend_yield"): fin_lines.append(f"Div Yield: {fins['dividend_yield']:.2%}")
            if fins.get("market_cap"):
                mc = fins["market_cap"]
                if mc > 1e12: fin_lines.append(f"Mkt Cap: ${mc/1e12:.1f}T")
                elif mc > 1e9: fin_lines.append(f"Mkt Cap: ${mc/1e9:.1f}B")
            
            fin_text = ", ".join(fin_lines) if fin_lines else "No fundamental data"
            
            block = f"""{ticker}:
Price: ${price}, 1d: {returns.get('1d','N/A')}%, 5d: {returns.get('5d','N/A')}%, 1m: {returns.get('1m','N/A')}%, 3m: {returns.get('3m','N/A')}%, 6m: {returns.get('6m','N/A')}%, 12m: {returns.get('12m','N/A')}%
Vol: {vol}%, From 52w High: {pct_high}%, {fin_text}"""
            stock_blocks.append(block)
        
        stocks_text = "\n---\n".join(stock_blocks)
        
        return f"""You are a quantitative analyst analyzing {len(tickers)} stocks simultaneously.

For EACH stock below, determine: action (BUY/HOLD/SELL), conviction (0.1-1.0), sentiment (-0.9 to 0.9, NEVER 0.0), confidence (0.1-1.0), and a 1-sentence reasoning.

STOCKS:
{stocks_text}

Return a JSON array with EXACTLY one object per stock, in the SAME ORDER:
[
  {{"ticker": "TICKER1", "action": "BUY"|"HOLD"|"SELL", "conviction": 0.1-1.0, "sentiment": -0.9 to 0.9, "confidence": 0.1-1.0, "reasoning": "1 sentence"}},
  ...
]

RULES:
- BUY: strong momentum + reasonable valuation + manageable risk
- HOLD: neutral or uncertain
- SELL: weak momentum or expensive or excessive risk
- Sentiment MUST be non-zero. Strong opinions = high absolute values.
- Return ONLY the JSON array, no other text."""

    def _parse_batch_results(self, tickers: list[str], 
                             votes_per_model: list,
                             financial_context: dict) -> dict:
        """Parse batch results from multiple models into per-ticker CouncilResults."""
        # Collect all model responses into per-ticker data
        ticker_model_data = {t: [] for t in tickers}
        
        for vote in votes_per_model:
            if not vote or vote.error:
                continue
            
            # vote.raw_content contains the raw API response text for batch
            content = vote.raw_content if hasattr(vote, 'raw_content') and vote.raw_content else vote.reasoning
            
            try:
                import json
                parsed = self.council._parse_response(content)
                if isinstance(parsed, dict) and parsed.get("reasoning") == "Parse error":
                    # _parse_response returned fallback, skip
                    continue
                if isinstance(parsed, list):
                    for item in parsed:
                        t = item.get("ticker", "").upper()
                        if t in ticker_model_data:
                            ticker_model_data[t].append({
                                "model_name": vote.model_name,
                                "sentiment": float(item.get("sentiment", 0)),
                                "confidence": float(item.get("confidence", 0)),
                                "reasoning": item.get("reasoning", ""),
                            })
            except Exception as e:
                logger.warning(f"Failed to parse batch response from {vote.model_name}: {e}")
        
        # Aggregate per ticker
        results = {}
        for ticker in tickers:
            model_data = ticker_model_data.get(ticker, [])
            if not model_data:
                # No valid data for this ticker
                results[ticker] = CouncilResult(
                    ticker=ticker, sentiment=0.0, confidence=0.0,
                    agreement=0.0, reasoning="No council data", votes=[], num_votes=0
                )
                continue
            
            # Confidence-weighted average
            total_weight = 0.0
            weighted_sentiment = 0.0
            confidences = []
            
            for md in model_data:
                model_weight = self.council.models.get(md["model_name"], {}).get("weight", 1.0)
                weight = md["confidence"] * model_weight
                weighted_sentiment += md["sentiment"] * weight
                total_weight += weight
                confidences.append(md["confidence"])
            
            avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Agreement
            sentiments = [md["sentiment"] for md in model_data]
            if len(sentiments) > 1:
                agreement = max(0.0, 1.0 - np.std(sentiments))
            else:
                agreement = 1.0
            
            final_confidence = avg_confidence * (0.5 + 0.5 * agreement)
            
            reasoning = " | ".join([f"[{md['model_name']}] {md['reasoning']}" for md in model_data[:3]])
            
            results[ticker] = CouncilResult(
                ticker=ticker,
                sentiment=avg_sentiment,
                confidence=final_confidence,
                agreement=agreement,
                reasoning=reasoning,
                votes=[],
                num_votes=len(model_data),
            )
        
        return results
    
    def _query_council(self, ticker: str, prompt: str) -> CouncilResult:
        """Query council models with our custom prompt."""
        # Use the council's model list but our custom prompt
        votes = self.council._collect_votes(prompt, ticker=ticker)
        return self.council._aggregate_votes(ticker, votes)
    
    def _result_to_pick(self, ticker: str, result: CouncilResult,
                        ctx: dict) -> StockPick:
        """Convert council result to a StockPick with composite score."""
        # Determine action from sentiment
        sentiment = result.sentiment
        confidence = result.confidence
        agreement = result.agreement
        
        if sentiment > 0.2 and agreement >= self.min_council_agreement:
            action = "BUY"
            conviction = sentiment * confidence
        elif sentiment < -0.2 and agreement >= self.min_council_agreement:
            action = "SELL"
            conviction = abs(sentiment) * confidence
        else:
            action = "HOLD"
            conviction = (1 - abs(sentiment)) * confidence
        
        # Compute momentum and value ranks from context
        returns = ctx.get("returns", {})
        mom_12m = returns.get("12m", 0) / 100  # Convert from percentage
        mom_1m = returns.get("1m", 0) / 100
        
        # Simple momentum rank (higher is better)
        mom_rank = min(1.0, max(0.0, (mom_12m + 0.3) / 0.6))  # Normalize -30% to +30% → 0 to 1
        
        # Simple value rank (lower P/E is better)
        pe = ctx.get("fundamentals", {}).get("pe_ratio")
        if pe and pe > 0:
            val_rank = min(1.0, max(0.0, (35 - pe) / 30))  # P/E 5-35 → 0 to 1
        else:
            val_rank = 0.5
        
        # Composite score
        composite = (
            self.council_weight * conviction +
            self.momentum_weight * mom_rank +
            self.value_weight * val_rank
        )
        
        return StockPick(
            ticker=ticker,
            action=action,
            conviction=conviction,
            sentiment=sentiment,
            confidence=confidence,
            reasoning=result.reasoning,
            num_votes=result.num_votes,
            momentum_rank=mom_rank,
            value_rank=val_rank,
            composite_score=composite,
        )
