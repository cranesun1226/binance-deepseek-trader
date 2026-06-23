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
DEFAULT_ZAI_MODEL = "glm-5.2"
DEFAULT_ZAI_REASONING_EFFORT = "max"
DEFAULT_ZAI_MAX_TOKENS = 8192
DEFAULT_ZAI_TIMEOUT_SECONDS = 300.0
DEFAULT_AI_PROMPT_TIMEFRAME = "1h"
DEFAULT_AI_PROMPT_CANDLE_COUNT = 168
DEFAULT_ACTIVE_LEVERAGE = 1
DEFAULT_CAPITAL_USAGE_RATIO = 0.99
DEFAULT_REBALANCE_THRESHOLD_PCT = 0.02
DEFAULT_ACTIVE_RESCREEN_INTERVAL_HOURS = 24.0

DEFAULT_CONFIG: Dict[str, Any] = {
    "project_name": "binance-zai-trader",
    "cycle_interval_seconds": 60,
    "active_leverage": DEFAULT_ACTIVE_LEVERAGE,
    "stop_loss_pct": 0.05,
    "ai_prompt_timeframe": DEFAULT_AI_PROMPT_TIMEFRAME,
    "ai_prompt_candle_count": DEFAULT_AI_PROMPT_CANDLE_COUNT,
    "capital_usage_ratio": DEFAULT_CAPITAL_USAGE_RATIO,
    "rebalance_threshold_pct": DEFAULT_REBALANCE_THRESHOLD_PCT,
    "active_rescreen_interval_hours": DEFAULT_ACTIVE_RESCREEN_INTERVAL_HOURS,
    "zai_model": DEFAULT_ZAI_MODEL,
    "zai_reasoning_effort": DEFAULT_ZAI_REASONING_EFFORT,
    "zai_max_tokens": DEFAULT_ZAI_MAX_TOKENS,
    "zai_timeout_seconds": DEFAULT_ZAI_TIMEOUT_SECONDS,
    "screener_quote": "USDT",
    "screener_timeout": 30.0,
    "screener_retries": 3,
    "screener_request_sleep": 0.10,
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
