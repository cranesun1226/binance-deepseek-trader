# Binance DeepSeek Trader

Binance USDT-M futures portfolio runner powered by a trend-quality active screener, DeepSeek direction checks, native stop-loss synchronization, and Telegram notifications.

## Strategy

- Four active slots, each targeting 25% margin allocation.
- Active 1, 2, and 3 screen TradFi USDT-M perpetual symbols.
- Active 4 screens the full USDT-M perpetual universe, including TradFi and non-TradFi crypto symbols.
- This keeps three dedicated TradFi slots while allowing the fourth slot to select the strongest overall USDT setup.
- All active slots use 2x leverage by default and a fixed 4% stop loss from entry.
- Open active positions use the 1% price trigger for DeepSeek trader direction checks and the configured active re-screen interval, default 24 hours, for universe ranking and symbol review.

## Active Screener

The screener fetches `ai_prompt_candle_count` closes on `ai_prompt_timeframe`, default 168 one-hour closes. Each candidate is fit on log prices and ranked by `trend_score`, a weighted percentile score using:

- trend strength
- R squared
- path efficiency
- directional consistency
- daily consistency
- adverse excursion score
- trend magnitude

Weekly consistency is still recorded as diagnostic metadata, but it is not part of
the default score because the default 168 one-hour window creates a single
weekly segment, which mostly duplicates the endpoint trend filter.

The fitted slope defines the screening direction:

- `LONG` when the slope is positive
- `SHORT` when the slope is negative

Flat paths or endpoint returns that do not align with the fitted trend are skipped.

Before ranking, the screener applies the directional entry filter against the live lookup price:

- `LONG`: `min(low over latest 24 1h klines) >= live_price * 0.96`
- `SHORT`: `max(high over latest 24 1h klines) <= live_price * 1.04`

The trend direction is screening metadata. After a candidate is selected, the DeepSeek trader flow still evaluates the final `LONG` or `SHORT` execution direction for entries and active reviews.

## Active Review

When a slot has an open position, it is held until one of these happens:

- the fixed exchange stop loss is hit
- the position is manually or externally removed
- price moves by the configured trigger percentage from the last DeepSeek trader anchor
- the 24h active review is due

On a price-triggered review, DeepSeek trader rechecks the current symbol direction and the existing position is kept, resized, or reversed as needed. During the 24h review, the screener re-ranks the slot's own universe while allowing the current symbol. If the current symbol remains top-ranked, DeepSeek trader reviews direction again. If a different candidate is top-ranked, DeepSeek trader first evaluates both the current symbol and the candidate, then DeepSeek rebalancer chooses between those two trader-backed directions. The current position is closed only when the new candidate is selected.

## Configuration

Edit [setting.yaml](/Users/haksunlee/investment/binance-deepseek-trader/setting.yaml) for runtime settings:

```yaml
cycle_interval_seconds: 60
trigger_pct_usdt: 1.0

ai_prompt_timeframe: 1h
ai_prompt_candle_count: 168

deepseek_model: deepseek-v4-flash
deepseek_reasoning_effort: max
deepseek_max_tokens: 8192
deepseek_timeout_seconds: 300.0

active_leverage: 2
capital_usage_ratio: 0.99
rebalance_threshold_pct: 0.02
stop_loss_pct: 0.04
active_rescreen_interval_hours: 24

screener_quote: USDT
screener_timeout: 30.0
screener_retries: 3
screener_request_sleep: 0.10
```

## Environment

Create a local `.env` with:

```bash
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
DEEPSEEK_API_KEY=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

Optional:

```bash
BINANCE_FUTURES_BASE_URL=https://fapi.binance.com
```

## Run

```bash
pip install -r requirements.txt
python main.py
```

Useful checks:

```bash
python -m py_compile main.py src/ai/deepseek_trader.py src/ai/deepseek_rebalancer.py src/strategy/portfolio_strategy.py src/strategy/active_screener.py src/strategy/scheduler.py src/binance/trade_position.py src/infra/env_loader.py src/infra/telegram.py
python -m unittest discover
```

## Project Layout

```text
src/
  ai/
    deepseek_trader.py       # single-symbol direction decision
    deepseek_rebalancer.py   # current-vs-candidate active review selector
  binance/
    market_data.py
    trade_position.py
  infra/
    logger.py
    price_chart.py
    telegram.py
  strategy/
    active_screener.py       # trend-quality active market screening
    portfolio_strategy.py    # four-active-slot runtime
    scheduler.py
```

`db/` stores DeepSeek decisions, active screening outputs, and material position-event artifacts. Routine mechanical cycles are not persisted unless they include a decision, screening event, unmanaged close, or material position change.
