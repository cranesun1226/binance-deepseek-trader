# Binance ZAI Trader

Binance USDT-M futures portfolio runner powered by a trend-quality active screener, ZAI GLM-5.2 direction checks, native stop-loss synchronization, and Telegram notifications.

## Strategy

- Four active slots, each targeting 25% margin allocation.
- Active 1, 2, and 3 screen TradFi USDT-M perpetual symbols.
- Active 4 screens the full USDT-M perpetual universe, including TradFi and non-TradFi crypto symbols.
- This keeps three dedicated TradFi slots while allowing the fourth slot to select the strongest overall USDT setup.
- All active slots use 1x leverage by default and a fixed 5% stop loss from entry.
- ZAI is called only when an active slot has no open position or when the configured active re-screen interval is due, default 18 hours.

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

- `LONG`: `min(low over latest 24 1h klines) >= live_price * 0.95`
- `SHORT`: `max(high over latest 24 1h klines) <= live_price * 1.05`

The trend direction is screening metadata. After a fresh candidate is selected for an empty active slot, the ZAI trader flow evaluates the final `LONG` or `SHORT` execution direction for entry. In active rebalancing, the rebalancer uses the current position direction and the screened candidate direction directly.

## Active Review

When a slot has an open position, it is held until one of these happens:

- the fixed exchange stop loss is hit
- the position is manually or externally removed
- the 18h active review is due

Between 18h reviews, the runtime only checks exchange state and synchronizes the fixed 5% stop loss. During the 18h review, the screener re-ranks the slot's own universe while allowing the current symbol. If the current symbol remains top-ranked, ZAI trader reviews direction again. If a different candidate is top-ranked, ZAI rebalancer directly compares the current position direction against the screened candidate direction and chooses one symbol. The current position is closed only when the new candidate is selected.

## Configuration

Edit [setting.yaml](setting.yaml) for runtime settings:

```yaml
cycle_interval_seconds: 60

ai_prompt_timeframe: 1h
ai_prompt_candle_count: 168

zai_model: glm-5.2
zai_reasoning_effort: max
zai_max_tokens: 8192
zai_timeout_seconds: 300.0

active_leverage: 1
capital_usage_ratio: 0.99
rebalance_threshold_pct: 0.02
stop_loss_pct: 0.05
active_rescreen_interval_hours: 18

screener_quote: USDT
screener_timeout: 30.0
screener_retries: 3
screener_request_sleep: 0.10
```

ZAI direction and active-rebalance reasons are constrained to 300 English characters.

## Environment

Create a local `.env` with:

```bash
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
ZAI_API_ID=...
ZAI_API_KEY=...
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
python -m py_compile main.py src/ai/zai_trader.py src/ai/zai_rebalancer.py src/strategy/portfolio_strategy.py src/strategy/active_screener.py src/strategy/scheduler.py src/binance/trade_position.py src/infra/env_loader.py src/infra/telegram.py
python -m unittest discover
```

## Project Layout

```text
src/
  ai/
    zai_trader.py            # single-symbol direction decision
    zai_rebalancer.py        # current-vs-candidate active review selector
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

`db/` stores ZAI decisions, active screening outputs, and material position-event artifacts. Routine mechanical cycles are not persisted unless they include a decision, screening event, unmanaged close, or material position change.
