# MiMo Quant Fund

LLM-powered quantitative hedge fund. AI council is the brain, Python is the plumbing.

## What It Does

Three-factor stock selection system:
1. **Momentum** (40%) — Jegadeesh-Titman 12-1 cross-sectional momentum
2. **Value** (30%) — Fama-French book-to-market + earnings yield
3. **LLM Council** (30%) — 4 free models vote independently on sentiment

The council approach: multiple AI models analyze the same news, each votes independently, votes are confidence-weighted and aggregated. This reduces individual model bias and improves signal quality.

## The Council

4 free models on OpenRouter, each analyzing the same data:

| Model | Params | Context | Role |
|-------|--------|---------|------|
| Llama 3.3 70B | 70B | 131K | Meta's flagship open model |
| Qwen3 Next 80B | 80B | 262K | Alibaba's frontier model |
| Gemma 4 31B | 31B | 262K | Google's efficient model |
| GPT-OSS 120B | 120B | 131K | OpenAI's open model |

**Cost: $0** — all models are free on OpenRouter.

## Backtest Results

| Metric | 2-Factor (Mom+Val) | 3-Factor (+Council) |
|--------|-------------------|---------------------|
| Annual Return | 8.87% | **11.75%** |
| Sharpe Ratio | 0.78 | **1.03** |
| Sortino Ratio | 0.96 | **1.27** |
| Max Drawdown | -14.48% | -15.34% |
| Calmar Ratio | 0.61 | **0.77** |

5-year backtest (2021-2026), 50 S&P 500 stocks, $100K initial capital.

## Quick Start

```bash
# Clone
git clone https://github.com/tnguyen0362/mimo-quant-fund.git
cd mimo-quant-fund

# Setup
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate    # Mac/Linux
pip install -r requirements.txt

# Run (2-factor, no API key needed)
python run_quant_backtest.py

# Run (3-factor with LLM Council)
set OPENROUTER_API_KEY=sk-or-your-key-here    # Windows
# export OPENROUTER_API_KEY=sk-or-your-key    # Mac/Linux
python run_llm_backtest.py
```

## Project Structure

```
trading-system/
├── config/
│   └── settings.py              # TradingConfig (all parameters)
├── data/
│   ├── universe.py              # S&P 500 universe management
│   ├── market.py                # Price data (yfinance, batch fetch)
│   └── fundamentals.py          # P/E, P/B, EV/EBITDA (yfinance)
├── factors/
│   ├── momentum.py              # 12-1 cross-sectional momentum
│   ├── value.py                 # Fama-French value factor
│   ├── combined.py              # 2-factor ranking (mom + value)
│   ├── llm_factor.py            # Single-model LLM sentiment
│   ├── llm_combined.py          # 3-factor ranking (+ single LLM)
│   ├── council_factor.py        # Multi-model council sentiment
│   └── council_combined.py      # 3-factor ranking (+ council)
├── llm/
│   ├── sentiment.py             # Single-model LLM sentiment
│   └── council.py               # 4-model LLM council
├── engine/
│   ├── signals.py               # Signal generation
│   ├── position_sizer.py        # Half-Kelly position sizing
│   ├── risk_manager.py          # Drawdown circuit breakers
│   ├── risk_monitor.py          # 5-check risk monitoring
│   └── volatility_targeting.py  # Vol scaling overlay
├── backtest/
│   ├── engine.py                # Single-ticker backtest
│   └── portfolio_engine.py      # Multi-asset portfolio backtest
├── features/
│   ├── technical.py             # SMA, RSI, MACD, Bollinger
│   └── fundamental.py           # Fundamental scoring
├── run_quant_backtest.py        # 2-factor backtest entry point
├── run_llm_backtest.py          # 3-factor (council) backtest entry point
└── requirements.txt
```

## How It Works

```
Monthly Rebalance Pipeline:
─────────────────────────────────────────────────────
1. Fetch price data for 50 S&P 500 stocks (yfinance)
2. Fetch fundamentals (P/E, P/B from yfinance)
3. Fetch news headlines (yfinance news)
4. Council deliberation:
   - Llama 3.3 70B → sentiment vote
   - Qwen3 Next 80B → sentiment vote
   - Gemma 4 31B → sentiment vote
   - GPT-OSS 120B → sentiment vote
   - Aggregate: confidence-weighted average
5. Rank by: 0.4 × momentum + 0.3 × value + 0.3 × council
6. Select top 15, equal-weight
7. Risk check: drawdown, position size, volatility
8. Execute trades via Robinhood MCP (future)
```

## Council Architecture

```
News Headlines for AAPL
        │
        ├──── Llama 3.3 70B ──── sentiment: +0.7, confidence: 0.8
        │
        ├──── Qwen3 Next 80B ── sentiment: +0.6, confidence: 0.7
        │
        ├──── Gemma 4 31B ───── sentiment: +0.8, confidence: 0.9
        │
        └──── GPT-OSS 120B ──── sentiment: +0.5, confidence: 0.6
                                    │
                                    ▼
                        Council Aggregator
                        (confidence-weighted)
                                    │
                                    ▼
                        Final: sentiment=+0.65, confidence=0.75
                        Agreement: 85% (models agree)
```

## Configuration

All parameters in `config/settings.py`, overridable via environment variables:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INITIAL_CAPITAL` | 100000 | Starting capital |
| `TOP_N_STOCKS` | 15 | Number of stocks to hold |
| `TARGET_VOL` | 0.15 | Target portfolio volatility |
| `MOMENTUM_WEIGHT` | 0.40 | Weight for momentum factor |
| `VALUE_WEIGHT` | 0.30 | Weight for value factor |
| `LLM_WEIGHT` | 0.30 | Weight for LLM council |
| `COMMISSION_RATE` | 0.001 | Commission per trade (0.1%) |
| `SLIPPAGE_RATE` | 0.001 | Slippage per trade (0.1%) |

## Cost

Running the full LLM backtest: **$0.00** — all council models are free on OpenRouter.

## Roadmap

- [x] Rule-based quant engine (momentum + value)
- [x] Single-model LLM sentiment (MiMo 2.5)
- [x] **Multi-model LLM council (4 free models)**
- [x] Three-factor combined ranking
- [x] Portfolio backtest engine
- [x] Risk monitoring + volatility targeting
- [ ] Real news API integration (NewsAPI, Finnhub)
- [ ] Robinhood MCP execution layer
- [ ] Paper trading via Alpaca
- [ ] Live trading with $1,000

## License

MIT
