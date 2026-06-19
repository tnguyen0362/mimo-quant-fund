# LLM Council — Multi-Model Sentiment Ensemble
#
# Instead of one LLM making the call, multiple free models vote independently.
# The council approach:
# 1. Each model analyzes the same news headlines
# 2. Each returns sentiment + confidence
# 3. Votes are aggregated (confidence-weighted average)
# 4. Disagreement = lower confidence (feature, not bug)
#
# This reduces individual model bias and improves signal quality.
# All models are FREE on OpenRouter.

import os
import json
import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


# Council configuration: REASONING models on OpenRouter (verified June 2026)
# Strategy: Reasoning models think through sentiment step-by-step before deciding
# This produces more reliable, explainable sentiment scores than standard models

COUNCIL_MODELS = {
    # --- TIER 1: Mandatory reasoning (always show chain-of-thought) ---
    "gpt-oss-120b": {
        "model_id": "openai/gpt-oss-120b:free",
        "name": "GPT-OSS 120B",
        "weight": 1.2,  # Large reasoning model, higher weight
        "reasoning": True,
    },
    "gpt-oss-20b": {
        "model_id": "openai/gpt-oss-20b:free",
        "name": "GPT-OSS 20B",
        "weight": 1.0,
        "reasoning": True,
    },
    # --- TIER 2: Optional reasoning (toggle chain-of-thought) ---
    "nemotron-ultra-550b": {
        "model_id": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "name": "Nemotron 3 Ultra 550B",
        "weight": 1.2,  # Largest model, highest weight
        "reasoning": True,
    },
    "nemotron-super-120b": {
        "model_id": "nvidia/nemotron-3-super-120b-a12b:free",
        "name": "Nemotron 3 Super 120B",
        "weight": 1.0,
        "reasoning": True,
    },
    "nemotron-nano-30b": {
        "model_id": "nvidia/nemotron-3-nano-30b-a3b:free",
        "name": "Nemotron 3 Nano 30B",
        "weight": 0.9,
        "reasoning": True,
    },
    "nemotron-nano-9b": {
        "model_id": "nvidia/nemotron-nano-9b-v2:free",
        "name": "Nemotron Nano 9B",
        "weight": 0.7,
        "reasoning": True,
    },
    # --- TIER 3: Strong general models (diversity) ---
    "hermes-405b": {
        "model_id": "nousresearch/hermes-3-llama-3.1-405b:free",
        "name": "Hermes 3 Llama 405B",
        "weight": 1.1,
        "reasoning": False,
    },
    "llama-70b": {
        "model_id": "meta-llama/llama-3.3-70b-instruct:free",
        "name": "Llama 3.3 70B",
        "weight": 1.0,
        "reasoning": False,
    },
    "gemma-31b": {
        "model_id": "google/gemma-4-31b-it:free",
        "name": "Gemma 4 31B",
        "weight": 0.9,
        "reasoning": False,
    },
    # --- AUTO-SELECT: Let OpenRouter pick the best free model ---
    "auto-router": {
        "model_id": "openrouter/free",
        "name": "Free Models Router",
        "weight": 1.0,
        "reasoning": False,
    },
}


# Simplified retry prompt when models return sentiment=0.0
_RETRY_PROMPT_TEMPLATE = (
    "Is {ticker} stock likely to go UP or DOWN from here? "
    'Return JSON: {{"sentiment": +0.5 or -0.5, "confidence": 0.5, "reasoning": "brief"}}'
)


@dataclass
class CouncilVote:
    """A single model's vote on sentiment."""
    model_name: str
    model_id: str
    sentiment: float      # -1.0 to +1.0
    confidence: float     # 0.0 to 1.0
    reasoning: str
    latency_ms: float = 0.0
    error: str = ""


@dataclass
class CouncilResult:
    """Aggregated council decision."""
    ticker: str
    sentiment: float          # Confidence-weighted average
    confidence: float         # Average confidence × agreement bonus
    agreement: float          # How much models agree (0-1)
    reasoning: str            # Combined reasoning from all models
    votes: list               # List of CouncilVote objects
    num_votes: int
    source: str = "council"


class LLMCouncil:
    """
    Multi-model sentiment council using free OpenRouter models.
    
    Architecture:
    - Spawns parallel requests to N free models
    - Each model independently scores sentiment
    - Aggregates via confidence-weighted average
    - Agreement metric: if models disagree, confidence drops
    
    Benefits over single model:
    - Reduces individual model bias
    - Disagreement signals uncertainty (valuable!)
    - No single point of failure
    - All models are FREE
    """
    
    def __init__(self,
                 models: Optional[dict] = None,
                 api_key: Optional[str] = None,
                 max_workers: int = 4,
                 timeout: float = 30.0,
                 cache_dir: str = "data/cache/council",
                 max_models_per_stock: int = 2):
        """
        Args:
            models: Dict of model configs (default: COUNCIL_MODELS)
            api_key: OpenRouter API key (or OPENROUTER_API_KEY env var)
            max_workers: Max parallel API calls
            timeout: Request timeout in seconds
            cache_dir: Cache directory for responses
            max_models_per_stock: Max models to query per stock (rate limit control)
        """
        self.models = models or COUNCIL_MODELS
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.max_workers = max_workers
        self.timeout = timeout
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_models_per_stock = max_models_per_stock
        
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def deliberate(self, ticker: str, 
                   news_headlines: list[str],
                   financials: Optional[dict] = None) -> CouncilResult:
        """
        Council deliberation on a single stock.
        
        All models analyze the same information independently.
        Their votes are aggregated into a final decision.
        
        Args:
            ticker: Stock ticker
            news_headlines: Recent news headlines
            financials: Optional financial data for context
        
        Returns:
            CouncilResult with aggregated sentiment and individual votes
        """
        # Check cache
        cache_key = f"council_{ticker}_{hash(str(news_headlines[:5]))}"
        cached = self._load_cache(cache_key)
        if cached:
            return cached
        
        # Build prompt (same for all models)
        prompt = self._build_prompt(ticker, news_headlines, financials)
        
        # Collect votes from all models (parallel)
        votes = self._collect_votes(prompt, ticker=ticker)
        
        # Aggregate votes
        result = self._aggregate_votes(ticker, votes)
        
        # Cache result
        self._save_cache(cache_key, result)
        
        return result
    
    def deliberate_batch(self, tickers: list[str],
                         news_data: dict[str, list[str]],
                         financials_data: Optional[dict] = None) -> pd.DataFrame:
        """
        Batch deliberation for multiple stocks.
        
        Returns:
            DataFrame with columns: ticker, sentiment, confidence, agreement, 
                                    reasoning, num_votes
        """
        results = []
        
        for ticker in tickers:
            news = news_data.get(ticker, [])
            fins = financials_data.get(ticker) if financials_data else None
            
            result = self.deliberate(ticker, news, fins)
            
            results.append({
                "ticker": ticker,
                "sentiment": result.sentiment,
                "confidence": result.confidence,
                "agreement": result.agreement,
                "reasoning": result.reasoning,
                "num_votes": result.num_votes,
            })
            
            # Brief pause between stocks to respect rate limits
            time.sleep(0.5)
        
        return pd.DataFrame(results)
    
    def _build_prompt(self, ticker: str, 
                      news_headlines: list[str],
                      financials: Optional[dict] = None) -> str:
        """Build the sentiment analysis prompt."""
        news_text = "\n".join(f"- {h}" for h in news_headlines[:10]) if news_headlines else "No recent news available."
        
        financials_text = ""
        if financials:
            financials_text = f"""
Financial Data:
- P/E Ratio: {financials.get('pe_ratio', 'N/A')}
- Market Cap: {financials.get('market_cap', 'N/A')}
- Sector: {financials.get('sector', 'N/A')}
"""
        
        return f"""You are a quantitative investment analyst. Think step-by-step about the sentiment for {ticker} stock.

Recent News:
{news_text}
{financials_text}

Analyze this information:
1. What are the key themes in these headlines?
2. What do they suggest about the company's near-term outlook?
3. Are there any risks or concerns mentioned?
4. Overall, is this bullish, bearish, or neutral?

Respond in EXACTLY this JSON format (no other text):
{{
    "sentiment": <float from -1.0 to 1.0>,
    "confidence": <float from 0.0 to 1.0>,
    "reasoning": "<1-2 sentence explanation of your analysis>"
}}

Where:
- sentiment: -1.0 = extremely bearish, 0.0 = neutral, 1.0 = extremely bullish
- confidence: How confident you are in this assessment (0.0 to 1.0)
- reasoning: Brief explanation of your analysis

Only output the JSON, nothing else."""
    
    def _collect_votes(self, prompt: str, ticker: str = "") -> list[CouncilVote]:
        """Collect votes from council models in parallel (rate-limit aware)."""
        votes = []
        
        # Select models: use max_models_per_stock to stay within rate limits
        # With 50 stocks × 2 models = 100 requests (well within 200/day limit)
        model_items = list(self.models.items())[:self.max_models_per_stock]
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(model_items))) as executor:
            futures = {}
            for model_key, model_config in model_items:
                future = executor.submit(
                    self._query_model, model_key, model_config, prompt, ticker
                )
                futures[future] = model_key
            
            for future in as_completed(futures):
                model_key = futures[future]
                try:
                    vote = future.result()
                    if vote:
                        votes.append(vote)
                except Exception as e:
                    votes.append(CouncilVote(
                        model_name=model_key,
                        model_id=self.models[model_key]["model_id"],
                        sentiment=0.0,
                        confidence=0.0,
                        reasoning=f"Error: {str(e)}",
                        error=str(e),
                    ))
        
        return votes
    
    def _query_model(self, model_key: str, 
                     model_config: dict, 
                     prompt: str,
                     ticker: str = "") -> Optional[CouncilVote]:
        """Query a single model for its sentiment vote. Retries once if sentiment=0.0."""
        if not self.api_key or not HAS_HTTPX:
            return None
        
        vote = self._do_query(model_key, model_config, prompt)
        
        # Retry with simplified prompt if model returned sentiment=0.0
        if vote and vote.sentiment == 0.0 and not vote.error and ticker:
            retry_prompt = _RETRY_PROMPT_TEMPLATE.format(ticker=ticker)
            retry_vote = self._do_query(model_key, model_config, retry_prompt)
            if retry_vote and retry_vote.sentiment != 0.0 and not retry_vote.error:
                retry_vote.reasoning = f"[retry] {retry_vote.reasoning}"
                return retry_vote
        
        return vote
    
    def _do_query(self, model_key: str, 
                  model_config: dict, 
                  prompt: str) -> Optional[CouncilVote]:
        """Execute a single API query to a model."""
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://trading-system.local",
            "X-Title": "MiMo Quant Fund Council",
        }
        
        payload = {
            "model": model_config["model_id"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 200,
        }
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Parse response
                parsed = self._parse_response(content)
                
                return CouncilVote(
                    model_name=model_key,
                    model_id=model_config["model_id"],
                    sentiment=parsed.get("sentiment", 0.0),
                    confidence=parsed.get("confidence", 0.0),
                    reasoning=parsed.get("reasoning", "No reasoning"),
                    latency_ms=latency_ms,
                )
                
        except Exception as e:
            return CouncilVote(
                model_name=model_key,
                model_id=model_config["model_id"],
                sentiment=0.0,
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
                error=str(e),
            )
    
    def _parse_response(self, content: str) -> dict:
        """Parse LLM JSON response, handling markdown code blocks."""
        try:
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return {"sentiment": 0.0, "confidence": 0.0, "reasoning": "Parse error"}
    
    def _aggregate_votes(self, ticker: str, 
                         votes: list[CouncilVote]) -> CouncilResult:
        """Aggregate council votes into a final decision."""
        # Filter out failed votes
        valid_votes = [v for v in votes if v.error == "" and v.confidence > 0]
        
        if not valid_votes:
            return CouncilResult(
                ticker=ticker,
                sentiment=0.0,
                confidence=0.0,
                agreement=0.0,
                reasoning="All council models failed",
                votes=votes,
                num_votes=0,
            )
        
        # Confidence-weighted sentiment
        total_weight = 0.0
        weighted_sentiment = 0.0
        
        for vote in valid_votes:
            model_weight = self.models.get(vote.model_name, {}).get("weight", 1.0)
            weight = vote.confidence * model_weight
            weighted_sentiment += vote.sentiment * weight
            total_weight += weight
        
        avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0
        
        # Agreement metric: 1 = perfect agreement, 0 = total disagreement
        sentiments = [v.sentiment for v in valid_votes]
        if len(sentiments) > 1:
            std_sentiment = np.std(sentiments)
            # Map std to 0-1 agreement (std=0 → agreement=1, std=2 → agreement=0)
            agreement = max(0.0, 1.0 - std_sentiment)
        else:
            agreement = 1.0  # Single vote = full agreement with itself
        
        # Average confidence
        avg_confidence = np.mean([v.confidence for v in valid_votes])
        
        # Final confidence: average confidence × agreement bonus
        # If models disagree, we're less confident
        final_confidence = avg_confidence * (0.5 + 0.5 * agreement)
        
        # Combined reasoning
        reasoning_parts = []
        for v in valid_votes:
            reasoning_parts.append(f"[{v.model_name}] {v.reasoning}")
        combined_reasoning = " | ".join(reasoning_parts[:3])  # Top 3 for brevity
        
        return CouncilResult(
            ticker=ticker,
            sentiment=avg_sentiment,
            confidence=final_confidence,
            agreement=agreement,
            reasoning=combined_reasoning,
            votes=valid_votes,
            num_votes=len(valid_votes),
        )
    
    def _load_cache(self, key: str) -> Optional[CouncilResult]:
        """Load cached council result."""
        cache_file = self.cache_dir / f"{hash(key)}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                # Reconstruct votes
                votes = [CouncilVote(**v) for v in data.get("votes", [])]
                return CouncilResult(
                    ticker=data["ticker"],
                    sentiment=data["sentiment"],
                    confidence=data["confidence"],
                    agreement=data["agreement"],
                    reasoning=data["reasoning"],
                    votes=votes,
                    num_votes=data["num_votes"],
                )
            except Exception:
                pass
        return None
    
    def _save_cache(self, key: str, result: CouncilResult):
        """Save council result to cache."""
        cache_file = self.cache_dir / f"{hash(key)}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump({
                    "ticker": result.ticker,
                    "sentiment": result.sentiment,
                    "confidence": result.confidence,
                    "agreement": result.agreement,
                    "reasoning": result.reasoning,
                    "num_votes": result.num_votes,
                    "votes": [
                        {
                            "model_name": v.model_name,
                            "model_id": v.model_id,
                            "sentiment": v.sentiment,
                            "confidence": v.confidence,
                            "reasoning": v.reasoning,
                            "latency_ms": v.latency_ms,
                            "error": v.error,
                        }
                        for v in result.votes
                    ],
                }, f, indent=2)
        except Exception:
            pass


def print_council_result(result: CouncilResult):
    """Pretty-print a council deliberation result."""
    print(f"\n{'='*60}")
    print(f"COUNCIL DELIBERATION: {result.ticker}")
    print(f"{'='*60}")
    print(f"  Final Sentiment:  {result.sentiment:+.3f}")
    print(f"  Confidence:       {result.confidence:.1%}")
    print(f"  Agreement:        {result.agreement:.1%}")
    print(f"  Votes Cast:       {result.num_votes}")
    print(f"\n  Individual Votes:")
    for vote in result.votes:
        status = "OK" if vote.error == "" else "ERR"
        print(f"    [{status}] {vote.model_name:15s} | sentiment={vote.sentiment:+.3f} | conf={vote.confidence:.2f} | {vote.latency_ms:.0f}ms")
    print(f"\n  Reasoning: {result.reasoning[:100]}...")
    print(f"{'='*60}")
