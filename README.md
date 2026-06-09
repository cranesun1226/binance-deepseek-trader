# Binance DeepSeek Trader

DeepSeek-powered six-slot Binance USDT-M futures trading bot.

This project runs a 1x-leverage portfolio loop across four fixed passive markets and two dynamically screened active markets. Each slot is evaluated one symbol at a time by DeepSeek using `deepseek-v4-flash` with `max` reasoning effort, then managed with market entries/exits, rebalancing, stop-loss synchronization, and Telegram reporting.

> This software is for research and automation experiments. It is not financial advice. Futures trading can lose money quickly, and live mode sends real Binance Futures orders.

## LLM 자동 매매

### 주요 기능

- DeepSeek 기반 LLM 판단
  - 기본 모델: `deepseek-v4-flash`
  - reasoning effort: `max` (`xhigh` legacy values are normalized to `max`; `low`/`medium` map to `high`)
  - DeepSeek JSON Output 응답을 로컬에서 검증해 `LONG` 또는 `SHORT`만 허용
- 6개 슬롯 포트폴리오
  - Passive: `CLUSDT`, `XAUUSDT`, `QQQUSDT`, `BTCUSDT`
  - Active 1: 24시간 변동률 절대값이 4%에 가까운 crypto USDT-M perpetual 후보 10개 중 거래대금 최대 심볼
  - Active 2: TradFi USDT-M perpetual 중 24시간 변동률 절대값이 3~5%인 후보를 4% 근접순으로 최대 10개 추린 뒤 거래대금 최대 심볼
- 자산 배분
  - Passive 각 12.5%
  - Active 각 25%
  - 기본 총 투입률 99%, 레버리지 1x
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

기술적으로는 live-data dry-run에서 전체 6슬롯 사이클, DeepSeek 판단, Telegram 전송, active screener, 주문 수량 계산까지 확인했습니다. 다만 실제 구동 전에는 아래 체크리스트를 반드시 확인하세요.

- Binance 계좌를 자동화 전용으로 분리했는지 확인
- 기존 수동 포지션이 있어도 자동 청산되어도 괜찮은 계좌인지 확인
- Binance Futures API key에 필요한 futures trading 권한이 있는지 확인
- `.env`가 절대 Git에 올라가지 않는지 확인
- `python scripts/dry_run_live_cycle.py`가 다시 성공하는지 확인
- 첫 live run은 아주 작은 자금으로 `python main.py --once`부터 실행
- Telegram에서 각 슬롯 판단, 주문 결과, stop sync 메시지를 확인한 뒤 scheduler 상시 실행

### LLM 프롬프트

LLM에는 한 번에 한 종목의 `symbol`, 현재 `reference_price`, 그리고 1시간봉 종가 672개만 전달됩니다. 최신 종가는 판단 시점의 실시간 기준가로 보정됩니다. DeepSeek API에는 `response_format: {"type":"json_object"}`를 요청하고, 런타임이 `{"decision":"LONG","reason":"..."}` 또는 `{"decision":"SHORT","reason":"..."}`만 통과시킵니다. `reason`은 영어 200단어 이하의 판단 근거 요약이며, Telegram에는 raw reasoning 대신 이 값이 표시됩니다.

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
ai_prompt_candle_count: 672
deepseek_model: deepseek-v4-flash
deepseek_reasoning_effort: max
deepseek_max_tokens: 8192
deepseek_timeout_seconds: 300.0
fixed_leverage: 1
capital_usage_ratio: 0.99
rebalance_threshold_pct: 0.03
stop_loss_pct: 0.04
passive_symbols:
  - CLUSDT
  - XAUUSDT
  - QQQUSDT
  - BTCUSDT
active_targets:
  - 4.0
  - 4.0
# Active candidates must also have at least ai_prompt_candle_count candles on ai_prompt_timeframe.
active_candidate_pool_size: 10
active2_tradfi_min_abs_change_pct: 3.0
active2_tradfi_max_abs_change_pct: 5.0
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
│       ├── portfolio_strategy.py   # Six-slot portfolio state machine
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
- Six-slot portfolio
  - Passive: `CLUSDT`, `XAUUSDT`, `QQQUSDT`, `BTCUSDT`
- Active 1: top-volume crypto USDT-M perpetual among the 10 candidates closest to a 4% absolute 24h move
- Active 2: top-volume TradFi USDT-M perpetual after filtering to 3-5% absolute 24h move, ranking by closeness to 4%, and capping the pool at 10
- Allocation
  - 12.5% per passive slot
  - 25% per active slot
  - 99% default capital usage, 1x leverage
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

The project has passed a full live-data dry run for all six slots, including DeepSeek decisions, Telegram delivery, active screening, and order quantity planning. Before running live, complete this checklist:

- Use a dedicated Binance Futures automation account
- Confirm unmanaged/manual positions may be closed automatically
- Confirm your Binance API key has the required Futures permissions
- Confirm `.env` is never committed
- Re-run `python scripts/dry_run_live_cycle.py`
- Start live trading with a very small balance and `python main.py --once`
- Review Telegram messages for each slot before running the scheduler continuously

### LLM Prompt

The LLM receives only one symbol at a time: `symbol`, live `reference_price`, and 672 recent 1h close prices. The newest close is aligned to the live reference price at decision time. The API request uses `response_format: {"type":"json_object"}`, and the runtime accepts only `{"decision":"LONG","reason":"..."}` or `{"decision":"SHORT","reason":"..."}`. The `reason` value must be an English rationale summary of 200 words or fewer, and Telegram displays it instead of raw reasoning output.

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
ai_prompt_candle_count: 672
deepseek_model: deepseek-v4-flash
deepseek_reasoning_effort: max
deepseek_max_tokens: 8192
deepseek_timeout_seconds: 300.0
fixed_leverage: 1
capital_usage_ratio: 0.99
rebalance_threshold_pct: 0.03
stop_loss_pct: 0.04
passive_symbols:
  - CLUSDT
  - XAUUSDT
  - QQQUSDT
  - BTCUSDT
active_targets:
  - 4.0
  - 4.0
# Active candidates must also have at least ai_prompt_candle_count candles on ai_prompt_timeframe.
active_candidate_pool_size: 10
active2_tradfi_min_abs_change_pct: 3.0
active2_tradfi_max_abs_change_pct: 5.0
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
│       ├── portfolio_strategy.py   # Six-slot portfolio state machine
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
