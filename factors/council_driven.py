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
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import pandas as pd
import numpy as np

from llm.council import LLMCouncil, CouncilResult
from factors.momentum import MomentumFactor
from factors.value import ValueFactor


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
        Council analyzes each candidate with real financial data.
        Each model votes BUY / HOLD / SELL.
        """
        picks = []
        
        for ticker in candidates:
            ctx = financial_context.get(ticker, {})
            
            if ctx.get("insufficient_data"):
                continue
            
            # Build prompt with REAL data
            prompt = self._build_picker_prompt(ctx)
            
            # Get council votes
            # We override the deliberation to use our custom prompt
            result = self._query_council(ticker, prompt)
            
            # Convert to StockPick
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
