"""Runtime configuration defaults and loader helpers."""

import os
from copy import deepcopy
from typing import Any, Dict

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test environments
    class _YamlFallback:
        @staticmethod
        def safe_load(*_args, **_kwargs):
            return {}

    yaml = _YamlFallback()


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "setting.yaml")

# Runtime defaults live here so optional keys can be omitted from setting.yaml.
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_REASONING_EFFORT = "max"
DEFAULT_DEEPSEEK_MAX_TOKENS = 8192
DEFAULT_DEEPSEEK_TIMEOUT_SECONDS = 300.0
DEFAULT_AI_PROMPT_TIMEFRAME = "1h"
DEFAULT_AI_PROMPT_CANDLE_COUNT = 168
DEFAULT_TRIGGER_PCT_USDT = 1.0
DEFAULT_FIXED_LEVERAGE = 1
DEFAULT_CAPITAL_USAGE_RATIO = 0.99
DEFAULT_REBALANCE_THRESHOLD_PCT = 0.03
DEFAULT_PASSIVE_SYMBOLS = ["CLUSDT", "XAUUSDT", "QQQUSDT", "BTCUSDT"]
DEFAULT_ACTIVE_FILTER_MIN_7D_AVG_DAILY_QUOTE_VOLUME_USDT = 50_000_000.0
DEFAULT_ACTIVE_FILTER_MIN_7D_P10_HOURLY_QUOTE_VOLUME_USDT = 100_000.0
DEFAULT_ACTIVE_FILTER_MIN_OPEN_INTEREST_NOTIONAL_USDT = 50_000_000.0
DEFAULT_ACTIVE_FILTER_MIN_ORDER_BOOK_DEPTH_10BPS_USDT = 100_000.0

DEFAULT_CONFIG: Dict[str, Any] = {
    "project_name": "binance-deepseek-trader",
    "cycle_interval_seconds": 60,
    "trigger_pct_usdt": DEFAULT_TRIGGER_PCT_USDT,
    "fixed_leverage": DEFAULT_FIXED_LEVERAGE,
    "stop_loss_pct": 0.04,
    "ai_prompt_timeframe": DEFAULT_AI_PROMPT_TIMEFRAME,
    "ai_prompt_candle_count": DEFAULT_AI_PROMPT_CANDLE_COUNT,
    "capital_usage_ratio": DEFAULT_CAPITAL_USAGE_RATIO,
    "rebalance_threshold_pct": DEFAULT_REBALANCE_THRESHOLD_PCT,
    "deepseek_model": DEFAULT_DEEPSEEK_MODEL,
    "deepseek_reasoning_effort": DEFAULT_DEEPSEEK_REASONING_EFFORT,
    "deepseek_max_tokens": DEFAULT_DEEPSEEK_MAX_TOKENS,
    "deepseek_timeout_seconds": DEFAULT_DEEPSEEK_TIMEOUT_SECONDS,
    "passive_symbols": list(DEFAULT_PASSIVE_SYMBOLS),
    "screener_quote": "USDT",
    "screener_timeout": 30.0,
    "screener_retries": 3,
    "screener_request_sleep": 0.10,
    "active_filter_min_7d_avg_daily_quote_volume_usdt": DEFAULT_ACTIVE_FILTER_MIN_7D_AVG_DAILY_QUOTE_VOLUME_USDT,
    "active_filter_min_7d_p10_hourly_quote_volume_usdt": DEFAULT_ACTIVE_FILTER_MIN_7D_P10_HOURLY_QUOTE_VOLUME_USDT,
    "active_filter_min_open_interest_notional_usdt": DEFAULT_ACTIVE_FILTER_MIN_OPEN_INTEREST_NOTIONAL_USDT,
    "active_filter_min_order_book_depth_10bps_usdt": DEFAULT_ACTIVE_FILTER_MIN_ORDER_BOOK_DEPTH_10BPS_USDT,
}


def get_default_config() -> Dict[str, Any]:
    """Return a deep-copied default configuration payload."""
    return deepcopy(DEFAULT_CONFIG)


def get_default_config_value(key: str, default: Any = None) -> Any:
    """Return one default config value without exposing shared mutable state."""
    return deepcopy(DEFAULT_CONFIG.get(key, default))


def load_runtime_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Load runtime config from disk and merge it on top of the defaults."""
    config = get_default_config()
    try:
        with open(config_path, "r", encoding="utf-8") as file_obj:
            loaded = yaml.safe_load(file_obj) or {}
    except FileNotFoundError:
        return config
    except Exception as exc:
        # Fail closed on malformed config so the bot never trades with accidental defaults.
        raise ValueError(f"failed to load runtime config from {config_path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ValueError(f"runtime config must be a mapping in {config_path}")

    config.update(loaded)
    return config
