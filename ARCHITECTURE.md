# Architecture

## Design Philosophy

**MiMo 2.5 is the brain. Code is the plumbing.**

The LLM makes the investment decisions. Python handles data fetching, risk management, and trade execution. This is not a rule-based system with an LLM bolted on — the LLM is the primary alpha source.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                │
│                                                             │
│  UniverseManager ──→ MarketData ──→ FundamentalData         │
│  (S&P 500 list)     (yfinance)     (P/E, P/B, etc.)        │
│                                                             │
│  NewsFetcher ──→ headlines per stock                        │
│  (yfinance news)                                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    SIGNAL LAYER                              │
│                                                             │
│  MomentumFactor ──┐                                         │
│  (12-1 ranking)   │                                         │
│                   ├──→ LLMCombinedRanking ──→ Top 15 stocks │
│  ValueFactor ─────┤    (40/30/30)           with weights    │
│  (B/M + E/P)      │                                         │
│                   │                                         │
│  LLMFactor ───────┘                                         │
│  (MiMo 2.5 sentiment)                                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    RISK LAYER                                │
│                                                             │
│  RiskMonitor ──→ Pre-trade checks                           │
│  - Drawdown control (10% warning, 20% halt)                 │
│  - Daily loss limit (5%)                                    │
│  - Position stop-loss (5%)                                  │
│  - Volatility regime (1.5x target)                          │
│  - Correlation breakdown                                    │
│                                                             │
│  VolatilityTargeting ──→ Scale exposure to target vol       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    EXECUTION LAYER                           │
│                                                             │
│  PortfolioBacktestEngine ──→ Backtesting                    │
│  (commission + slippage modeling)                            │
│                                                             │
│  Robinhood MCP ──→ Live trading (future)                    │
│  (via agent.robinhood.com/mcp/trading)                      │
└─────────────────────────────────────────────────────────────┘
```

## Factor Orthogonality

The three factors are chosen because they measure fundamentally different things:

| Factor | Data Source | What It Measures | Correlation to Others |
|--------|-----------|-----------------|----------------------|
| Momentum | Price history | "Is this stock going up?" | Low (price-based) |
| Value | Financial statements | "Is this stock cheap?" | Negative to momentum |
| LLM Sentiment | News headlines | "What's happening with this company?" | Low (information-based) |

Momentum and value are negatively correlated (~-0.3), which is exactly what we want — they naturally diversify each other. The LLM sentiment factor adds a third orthogonal signal based on information that neither price nor fundamentals capture.

## LLM Integration

### How MiMo 2.5 Works in the Pipeline

```
News Headlines for AAPL:
  - "Apple reports record Q4 revenue"
  - "iPhone sales beat expectations"
  - "Analysts upgrade AAPL to buy"

          │
          ▼

MiMo 2.5 Prompt:
  "Analyze sentiment for AAPL stock.
   Recent news: [headlines above]
   Respond in JSON: {sentiment, confidence, reasoning}"

          │
          ▼

MiMo 2.5 Response:
  {
    "sentiment": 0.72,      // Bullish
    "confidence": 0.85,     // High confidence
    "reasoning": "Strong earnings beat with positive analyst revisions"
  }

          │
          ▼

Alpha Signal:
  signal = sentiment × confidence = 0.72 × 0.85 = 0.612
```

### Cost

MiMo 2.5 via OpenRouter: ~$0.15/M input tokens, ~$1.20/M output tokens.

Full 5-year backtest (50 stocks × 32 rebalances): ~$0.24 total.

### Fallback

If the API is unavailable or no key is set, the system falls back to keyword-based sentiment (crude but functional). The `source` field in results indicates whether LLM or fallback was used.

## Risk Management

### Drawdown Control

```
Drawdown < 10%     → Full exposure (100%)
10% < DD < 20%     → Linear reduction to 50%
DD > 20%           → HALT (0% exposure, liquidate)
```

### Daily Loss Limit

```
Daily loss < 3%    → No action
Daily loss > 5%    → Reduce positions by 50%
```

### Position Sizing

- Maximum 5% of portfolio per position
- Equal-weight among selected stocks
- Half-Kelly criterion available for advanced sizing

## Backtest Engine

The `PortfolioBacktestEngine` models:
- Commission: 0.1% per trade (configurable)
- Slippage: 0.1% per trade (configurable)
- Monthly rebalancing with forward-fill between rebalances
- Walk-forward validation support
- Portfolio-level equity tracking (not per-position)

## Data Pipeline

- **Prices**: yfinance (free, daily OHLCV, cached to parquet)
- **Fundamentals**: yfinance (P/E, P/B, EV/EBITDA, cached 30 days)
- **News**: yfinance news (free, limited)
- **Universe**: Wikipedia S&P 500 list (cached weekly)
- **Rate limiting**: 1 second between yfinance batches

## Future: Robinhood MCP Integration

```
Your Python Code (MCP Client)
        ↓ HTTP/MCP protocol
Robinhood MCP Server (agent.robinhood.com/mcp/trading)
        ↓
Robinhood Trading API
        ↓
Your Agentic Account
```

Requires a dedicated "Agentic" Robinhood account. Only market and limit orders. No paper trading — all trades are real money.
