# Binance DeepSeek Trader

DeepSeek-powered eight-slot Binance USDT-M futures trading bot.

This project runs an eight-slot portfolio loop across four fixed 2x-leverage passive markets and four dynamically screened 2x-leverage active markets: two crypto USDT-M perpetuals and two TradFi USDT-M perpetuals. Each slot is evaluated one symbol at a time by DeepSeek using `deepseek-v4-flash` with `max` reasoning effort, then managed with market entries/exits, rebalancing, stop-loss synchronization, and Telegram reporting.

> This software is for research and automation experiments. It is not financial advice. Futures trading can lose money quickly, and live mode sends real Binance Futures orders.

## LLM 자동 매매

### 주요 기능

- DeepSeek 기반 LLM 판단
  - 기본 모델: `deepseek-v4-flash`
  - reasoning effort: `max` (`xhigh` legacy values are normalized to `max`; `low`/`medium` map to `high`)
  - DeepSeek JSON Output 응답을 로컬에서 검증해 `LONG` 또는 `SHORT`만 허용
- 8개 슬롯 포트폴리오
  - Passive: `CLUSDT`, `XAUUSDT`, `QQQUSDT`, `BTCUSDT`
  - Active 1, 3: 168개 종가 range volatility가 가장 큰 crypto USDT-M perpetual
  - Active 2, 4: 168개 종가 range volatility가 가장 큰 TradFi USDT-M perpetual
- 자산 배분
  - Passive 각 12.5%
  - Active 각 12.5%
  - 기본 총 투입률 99%, passive 2x / active 2x 레버리지
- 리스크 관리
  - 각 포지션 진입가 기준 4% stop loss 동기화
  - 마지막 LLM 판단 기준가에서 ±1% 이동 시 재판단
  - 목표 비중에서 벗어나면 리밸런싱
  - 자동화 계좌 전용 전제: 관리되지 않는 수동 포지션은 자동 청산 대상
- 알림
  - Telegram 메시지
  - structured `reason` 판단 근거 요약 전송
  - 1시간봉 가격 차트 이미지 전송
- 검증
  - 실제 주문 없이 Binance 공개 데이터, DeepSeek, Telegram까지 통과하는 live-data dry-run 스크립트 제공

### 실제 구동해도 되나?

실제 구동 전에는 전체 8슬롯 사이클, DeepSeek 판단, Telegram 전송, active screener, 주문 수량 계산까지 live-data dry-run으로 다시 확인하세요. 다만 실제 구동 전에는 아래 체크리스트를 반드시 확인하세요.

- Binance 계좌를 자동화 전용으로 분리했는지 확인
- 기존 수동 포지션이 있어도 자동 청산되어도 괜찮은 계좌인지 확인
- Binance Futures API key에 필요한 futures trading 권한이 있는지 확인
- `.env`가 절대 Git에 올라가지 않는지 확인
- `python scripts/dry_run_live_cycle.py`가 다시 성공하는지 확인
- 첫 live run은 아주 작은 자금으로 `python main.py --once`부터 실행
- Telegram에서 각 슬롯 판단, 주문 결과, stop sync 메시지를 확인한 뒤 scheduler 상시 실행

### LLM 프롬프트

LLM에는 한 번에 한 종목의 `symbol`, 현재 `reference_price`, 그리고 1시간봉 종가 168개만 전달됩니다. 최신 종가는 판단 시점의 실시간 기준가로 보정됩니다. DeepSeek API에는 `response_format: {"type":"json_object"}`를 요청하고, 런타임이 `{"decision":"LONG","reason":"..."}` 또는 `{"decision":"SHORT","reason":"..."}`만 통과시킵니다. `reason`은 영어 200단어 이하의 판단 근거 요약이며, Telegram에는 raw reasoning 대신 이 값이 표시됩니다.

### Active 스크리닝

Active 1과 Active 3은 crypto USDT-M perpetual을, Active 2와 Active 4는 TradFi USDT-M perpetual을 대상으로 합니다. passive 심볼과 이미 관리 중인 심볼은 제외하며, 앞선 active 슬롯이 선택한 심볼도 뒤 슬롯에서 제외됩니다.

각 후보는 `ai_prompt_timeframe`의 `ai_prompt_candle_count`개 종가를 가져와 `abs(max(close)-min(close))/((max(close)+min(close))/2)`로 계산한 `close_range_volatility`가 큰 순서로 선정합니다. 기본값은 168개의 1시간봉 종가입니다. 랭킹 전에 Binance가 최신으로 반환한 2개의 1시간봉 high/low range를 검사하며, 둘 중 하나라도 4%를 넘으면 후보에서 제외합니다. 이 최신 2개에는 반드시 현재 진행 중인 1시간봉과 그 직전 1시간봉이 포함됩니다. 완전히 닫힌 봉 2개만 보는 방식이 아닙니다.

### 설치

```bash
git clone <repository-url>
cd binance-deepseek-trader

python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
```

`.env`를 채웁니다.

```bash
BINANCE_API_KEY="..."
BINANCE_API_SECRET="..."
DEEPSEEK_API_KEY="..."
TELEGRAM_BOT_TOKEN="..."
TELEGRAM_CHAT_ID="..."
```

### 설정

런타임 설정은 [setting.yaml](./setting.yaml)에 있습니다.

```yaml
cycle_interval_seconds: 60
trigger_pct_usdt: 1.0
ai_prompt_timeframe: 1h
ai_prompt_candle_count: 168
deepseek_model: deepseek-v4-flash
deepseek_reasoning_effort: max
deepseek_max_tokens: 8192
deepseek_timeout_seconds: 300.0
passive_leverage: 2
active_leverage: 2
capital_usage_ratio: 0.99
rebalance_threshold_pct: 0.03
stop_loss_pct: 0.04
passive_symbols:
  - CLUSDT
  - XAUUSDT
  - QQQUSDT
  - BTCUSDT
# Active candidates are ranked by close_range_volatility over
# ai_prompt_candle_count closes on ai_prompt_timeframe. The 4% filter uses the
# latest returned 2 klines, intentionally including the current forming 1h candle.
screener_quote: USDT
screener_timeout: 30.0
screener_retries: 3
screener_request_sleep: 0.10
```

### Dry Run

실제 Binance 주문 없이 다음 항목을 실제로 검증합니다.

- Binance public market data
- Active symbol screening
- DeepSeek LLM 판단
- Telegram 메시지와 차트 전송
- 주문 수량 계산
- stop-loss sync 경로

```bash
python scripts/dry_run_live_cycle.py
```

Telegram 없이 검증하려면:

```bash
python scripts/dry_run_live_cycle.py --no-telegram
```

### 실제 실행

한 번만 실행:

```bash
python main.py --once
```

상시 실행:

```bash
python main.py
```

Ubuntu systemd로 운영할 때는 호스트별 service 파일과 배포 스크립트를 로컬 전용으로 관리하세요. Public repo에는 서버 IP, SSH alias, private key 경로, 실제 원격 경로를 커밋하지 않습니다.

메모리가 작은 인스턴스에서는 swap을 먼저 붙이는 것을 권장합니다.

```bash
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

호스트 로컬 service 파일을 만든 뒤 시작합니다.

```bash
sudo systemctl daemon-reload
sudo systemctl enable binance-deepseek-trader
sudo systemctl start binance-deepseek-trader
sudo systemctl status binance-deepseek-trader
```

로그 확인:

```bash
journalctl -u binance-deepseek-trader -f
```

### 테스트

```bash
python -m unittest discover
python -m py_compile main.py src/ai/deepseek_trader.py src/strategy/portfolio_strategy.py src/strategy/active_screener.py src/strategy/scheduler.py src/binance/trade_position.py src/infra/env_loader.py src/infra/telegram.py scripts/dry_run_live_cycle.py
```

### 프로젝트 구조

```text
.
├── main.py                         # CLI entrypoint
├── setting.yaml                    # Runtime portfolio and model config
├── scripts/
│   └── dry_run_live_cycle.py       # Live-data dry-run without Binance orders
├── src/
│   ├── ai/
│   │   └── deepseek_trader.py      # DeepSeek structured LONG/SHORT decisions
│   ├── binance/
│   │   ├── common.py
│   │   ├── market_data.py
│   │   └── trade_position.py       # Binance Futures position/order helpers
│   ├── infra/
│   │   ├── env_loader.py
│   │   ├── logger.py
│   │   ├── price_chart.py
│   │   └── telegram.py
│   └── strategy/
│       ├── active_screener.py      # crypto and TradFi active market screening
│       ├── portfolio_strategy.py   # Eight-slot portfolio state machine
│       ├── runtime_config.py
│       └── scheduler.py
└── tests/
```

### 공개 저장소 주의사항

- `.env`, `log/`, `db/`, `scheduler_state.json`은 `.gitignore`에 포함되어 있습니다.
- 호스트별 운영 문서, service 파일, SSH 설정, 배포 스크립트는 public repo에 커밋하지 않습니다.
- API key, Telegram token, 실제 계좌 정보, live cycle output은 절대 커밋하지 마세요.
- `db/`에는 LLM 판단이 발생한 사이클과 중요한 포지션 이벤트의 입출력/차트 산출물이 저장됩니다. 일반 1분 점검 사이클은 저장하지 않으며, 최대 20개 cycle 디렉터리만 유지합니다.
- `log/ai_trader.log`는 10MB 단위로 회전하며 최대 5개 백업 파일을 유지합니다.

## LLM Auto Trading

### Features

- DeepSeek-based LLM decisions
  - Default model: `deepseek-v4-flash`
  - Reasoning effort: `max` (`xhigh` legacy values are normalized to `max`; `low`/`medium` map to `high`)
  - DeepSeek JSON Output is locally validated to accept only `LONG` or `SHORT`
- Eight-slot portfolio
  - Passive: `CLUSDT`, `XAUUSDT`, `QQQUSDT`, `BTCUSDT`
- Active 1 and 3: crypto USDT-M perpetuals with the largest 168-close range volatility
- Active 2 and 4: TradFi USDT-M perpetuals with the largest 168-close range volatility
- Allocation
  - 12.5% per passive slot
  - 12.5% per active slot
  - 99% default capital usage, 2x passive leverage and 2x active leverage
- Risk management
  - Native stop loss synchronized at 4% from entry price
  - Re-evaluates a slot after a ±1% move from the last LLM decision anchor
  - Rebalances toward target slot weights
  - Designed for a dedicated automation account: unmanaged manual positions may be closed automatically
- Notifications
  - Telegram messages
  - Structured `reason` rationale summaries
  - 1h price chart images
- Validation
  - Includes a live-data dry-run script that calls Binance public data, DeepSeek, and Telegram without submitting Binance orders

### Can I Run It Live?

Before running live, re-run the live-data dry run for the full eight-slot cycle, including DeepSeek decisions, Telegram delivery, active screening, and order quantity planning. Then complete this checklist:

- Use a dedicated Binance Futures automation account
- Confirm unmanaged/manual positions may be closed automatically
- Confirm your Binance API key has the required Futures permissions
- Confirm `.env` is never committed
- Re-run `python scripts/dry_run_live_cycle.py`
- Start live trading with a very small balance and `python main.py --once`
- Review Telegram messages for each slot before running the scheduler continuously

### LLM Prompt

The LLM receives only one symbol at a time: `symbol`, live `reference_price`, and 168 recent 1h close prices. The newest close is aligned to the live reference price at decision time. The API request uses `response_format: {"type":"json_object"}`, and the runtime accepts only `{"decision":"LONG","reason":"..."}` or `{"decision":"SHORT","reason":"..."}`. The `reason` value must be an English rationale summary of 200 words or fewer, and Telegram displays it instead of raw reasoning output.

### Active Screening

Active 1 and Active 3 screen crypto USDT-M perpetuals. Active 2 and Active 4 screen TradFi USDT-M perpetuals. Passive symbols, already-managed symbols, and symbols selected by earlier active slots in the same cycle are excluded.

For each candidate, the screener fetches `ai_prompt_candle_count` closes on `ai_prompt_timeframe` and ranks by `close_range_volatility = abs(max(close)-min(close))/((max(close)+min(close))/2)`, highest first. The default is 168 recent 1h closes. Before ranking, the screener checks the high/low range of the latest two 1h klines returned by Binance and excludes a candidate if either one exceeds 4%. These latest two klines intentionally include the currently forming 1h candle and the immediately preceding candle; the filter does not use only fully closed candles.

### Installation

```bash
git clone <repository-url>
cd binance-deepseek-trader

python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
```

Fill `.env`.

```bash
BINANCE_API_KEY="..."
BINANCE_API_SECRET="..."
DEEPSEEK_API_KEY="..."
TELEGRAM_BOT_TOKEN="..."
TELEGRAM_CHAT_ID="..."
```

### Configuration

Runtime settings live in [setting.yaml](./setting.yaml).

```yaml
cycle_interval_seconds: 60
trigger_pct_usdt: 1.0
ai_prompt_timeframe: 1h
ai_prompt_candle_count: 168
deepseek_model: deepseek-v4-flash
deepseek_reasoning_effort: max
deepseek_max_tokens: 8192
deepseek_timeout_seconds: 300.0
passive_leverage: 2
active_leverage: 2
capital_usage_ratio: 0.99
rebalance_threshold_pct: 0.03
stop_loss_pct: 0.04
passive_symbols:
  - CLUSDT
  - XAUUSDT
  - QQQUSDT
  - BTCUSDT
# Active candidates are ranked by close_range_volatility over
# ai_prompt_candle_count closes on ai_prompt_timeframe. The 4% filter uses the
# latest returned 2 klines, intentionally including the current forming 1h candle.
screener_quote: USDT
screener_timeout: 30.0
screener_retries: 3
screener_request_sleep: 0.10
```

### Dry Run

This validates the live external integrations without Binance order submission:

- Binance public market data
- Active symbol screening
- DeepSeek LLM decisions
- Telegram messages and chart images
- Order quantity planning
- Stop-loss sync path

```bash
python scripts/dry_run_live_cycle.py
```

Without Telegram:

```bash
python scripts/dry_run_live_cycle.py --no-telegram
```

### Live Run

Run one cycle:

```bash
python main.py --once
```

Run the scheduler:

```bash
python main.py
```

For Ubuntu systemd operation, keep host-specific service files and deployment scripts local-only. Do not commit server IPs, SSH aliases, private key paths, or real remote paths to the public repository.

On small-memory instances, add swap before running the bot.

```bash
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Create a host-local service file, then start it.

```bash
sudo systemctl daemon-reload
sudo systemctl enable binance-deepseek-trader
sudo systemctl start binance-deepseek-trader
sudo systemctl status binance-deepseek-trader
```

Follow logs:

```bash
journalctl -u binance-deepseek-trader -f
```

### Tests

```bash
python -m unittest discover
python -m py_compile main.py src/ai/deepseek_trader.py src/strategy/portfolio_strategy.py src/strategy/active_screener.py src/strategy/scheduler.py src/binance/trade_position.py src/infra/env_loader.py src/infra/telegram.py scripts/dry_run_live_cycle.py
```

### Repository Layout

```text
.
├── main.py                         # CLI entrypoint
├── setting.yaml                    # Runtime portfolio and model config
├── scripts/
│   └── dry_run_live_cycle.py       # Live-data dry-run without Binance orders
├── src/
│   ├── ai/
│   │   └── deepseek_trader.py      # DeepSeek structured LONG/SHORT decisions
│   ├── binance/
│   │   ├── common.py
│   │   ├── market_data.py
│   │   └── trade_position.py       # Binance Futures position/order helpers
│   ├── infra/
│   │   ├── env_loader.py
│   │   ├── logger.py
│   │   ├── price_chart.py
│   │   └── telegram.py
│   └── strategy/
│       ├── active_screener.py      # crypto and TradFi active market screening
│       ├── portfolio_strategy.py   # Eight-slot portfolio state machine
│       ├── runtime_config.py
│       └── scheduler.py
└── tests/
```

### Open Source Safety

- `.env`, `log/`, `db/`, and `scheduler_state.json` are ignored by Git.
- Host-specific operation docs, service files, SSH configuration, and deployment scripts are kept out of the public repository.
- Never commit API keys, Telegram tokens, account data, or live cycle outputs.
- `db/` stores LLM-triggered cycle artifacts and important position-event records only. Routine one-minute mechanical checks are not written to `db/`, and only the latest 20 cycle directories are retained.
- `log/ai_trader.log` rotates at 10MB and keeps up to 5 backup files.

## License

MIT. See [LICENSE](./LICENSE).
