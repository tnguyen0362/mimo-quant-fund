# MiMo Quant Fund

LLM-powered quantitative hedge fund. MiMo 2.5 is the brain, Python is the plumbing.

## What It Does

Three-factor stock selection system:
1. **Momentum** (40%) — Jegadeesh-Titman 12-1 cross-sectional momentum
2. **Value** (30%) — Fama-French book-to-market + earnings yield
3. **LLM Sentiment** (30%) — MiMo 2.5 analyzes news headlines, scores sentiment

The LLM reads news and earnings data, outputs a sentiment score (-1 to +1) with confidence. Combined with rule-based momentum and value factors, it selects the top 15 stocks, equal-weighted, rebalanced monthly.

## Backtest Results

| Metric | 2-Factor (Mom+Val) | 3-Factor (+MiMo) |
|--------|-------------------|-----------------|
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

# Run (3-factor with MiMo 2.5 sentiment)
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
│   ├── llm_factor.py            # MiMo 2.5 sentiment factor
│   └── llm_combined.py          # 3-factor ranking (+ LLM)
├── llm/
│   └── sentiment.py             # MiMo 2.5 via OpenRouter API
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
├── run_llm_backtest.py          # 3-factor (LLM) backtest entry point
└── requirements.txt
```

## How It Works

```
Daily Pipeline:
─────────────────────────────────────────────────────
1. Fetch price data for 50 S&P 500 stocks (yfinance)
2. Fetch fundamentals (P/E, P/B from yfinance)
3. Fetch news headlines (yfinance news)
4. MiMo 2.5 scores sentiment for each stock
5. Rank by: 0.4 × momentum + 0.3 × value + 0.3 × sentiment
6. Select top 15, equal-weight
7. Risk check: drawdown, position size, volatility
8. Execute trades via Robinhood MCP (future)
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
| `LLM_WEIGHT` | 0.30 | Weight for LLM sentiment |
| `COMMISSION_RATE` | 0.001 | Commission per trade (0.1%) |
| `SLIPPAGE_RATE` | 0.001 | Slippage per trade (0.1%) |

## Cost

Running the full LLM backtest costs ~$0.24 on OpenRouter (MiMo 2.5 at $0.15/M tokens).

## Roadmap

- [x] Rule-based quant engine (momentum + value)
- [x] MiMo 2.5 sentiment analysis
- [x] Three-factor combined ranking
- [x] Portfolio backtest engine
- [x] Risk monitoring + volatility targeting
- [ ] Real news API integration (NewsAPI, Finnhub)
- [ ] Robinhood MCP execution layer
- [ ] Paper trading via Alpaca
- [ ] Live trading with $1,000

## License

MIT
