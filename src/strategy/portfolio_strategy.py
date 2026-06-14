"""Eight-slot Binance futures portfolio runtime powered by DeepSeek."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Sequence

from src.ai.deepseek_trader import evaluate_trade_direction
from src.binance.market_data import fetch_klines, parse_klines
from src.binance.trade_position import (
    adjust_qty_for_symbol,
    calculate_position_metrics,
    cancel_all_symbol_orders,
    close_position,
    decimal_to_str,
    evaluate_entry_order_notional,
    get_account_overview,
    get_position_snapshot,
    get_positions,
    get_reference_price,
    place_market_entry_order,
    safe_decimal,
    set_leverage,
    sync_existing_position_stop_loss,
    wait_for_close_propagation,
)
from src.infra.env_loader import get_binance_credentials
from src.infra.logger import format_log_details, get_logger
from src.strategy.active_screener import NoActiveCandidateError, screen_active_symbol, screen_active_tradfi_symbol
from src.strategy.runtime_config import (
    DEFAULT_AI_PROMPT_CANDLE_COUNT,
    DEFAULT_AI_PROMPT_TIMEFRAME,
    DEFAULT_ACTIVE_LEVERAGE,
    DEFAULT_ACTIVE_RESCREEN_INTERVAL_HOURS,
    DEFAULT_CAPITAL_USAGE_RATIO,
    DEFAULT_FIXED_LEVERAGE,
    DEFAULT_PASSIVE_LEVERAGE,
    DEFAULT_DEEPSEEK_MAX_TOKENS,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_DEEPSEEK_REASONING_EFFORT,
    DEFAULT_DEEPSEEK_TIMEOUT_SECONDS,
    DEFAULT_PASSIVE_SYMBOLS,
    DEFAULT_REBALANCE_THRESHOLD_PCT,
    DEFAULT_TRIGGER_PCT_USDT,
    load_runtime_config,
)

logger = get_logger("portfolio_strategy")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "setting.yaml")
DB_DIR = os.path.join(ROOT_DIR, "db")
MAX_DB_CYCLE_DIRS = 20
STATE_VERSION = "1.0.0"
TRIGGER_PRICE_DIGITS = 8
MANAGED_DECISIONS = {"LONG", "SHORT"}
MATERIAL_POSITION_RECORD_ACTIONS = {
    "active_rank_review_failed_position_kept",
    "active_rank_switch_entry_failed",
    "active_rank_review_position_kept",
    "closed_position",
    "entry_order_failed",
    "increased_position",
    "opened_new_position",
    "rebalance_reduce_failed",
    "reduced_position",
    "reverse_close_failed",
    "reverse_reopen_failed",
    "reversed_position",
    "invalid_symbol_for_leverage",
    "set_leverage_failed",
    "switch_close_failed",
    "switched_active_position_by_rank",
}
NotificationCallback = Optional[Callable[[str, Dict[str, Any]], None]]
CycleDirFactory = Callable[[], str]


@dataclass(frozen=True)
class PortfolioSlot:
    slot_id: str
    label: str
    kind: str
    target_margin_ratio: float
    symbol: Optional[str] = None
    active_screening_mode: str = "crypto"


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_symbol(value: Any) -> Optional[str]:
    normalized = str(value or "").strip().upper()
    return normalized or None


def _normalize_positive_int(value: Any, default: int) -> int:
    parsed = _safe_int(value, default)
    return parsed if parsed > 0 else default


def _normalize_positive_float(value: Any, default: float) -> float:
    parsed = _safe_float(value, default)
    if parsed is None or parsed <= 0.0:
        return float(default)
    return float(parsed)


def _normalize_ratio(value: Any, default: float) -> float:
    parsed_value = value
    if isinstance(value, str):
        parsed_value = value.strip()
        if parsed_value.endswith("%"):
            parsed_value = parsed_value[:-1]
    parsed = _safe_float(parsed_value, default)
    if parsed is None or parsed <= 0.0:
        parsed = default
    if 1.0 < parsed <= 100.0:
        parsed /= 100.0
    return float(parsed)


def _normalize_trigger_percent(value: Any, default: float) -> float:
    parsed_value = value
    if isinstance(value, str):
        parsed_value = value.strip()
        if parsed_value.endswith("%"):
            parsed_value = parsed_value[:-1]
    parsed = _safe_float(parsed_value, default)
    if parsed is None or parsed <= 0.0 or parsed >= 100.0:
        parsed = default
    return float(parsed)


def _normalize_reasoning_effort(value: Any) -> str:
    normalized = str(value or DEFAULT_DEEPSEEK_REASONING_EFFORT).strip().lower()
    if normalized in {"max", "xhigh"}:
        return "max"
    if normalized in {"low", "medium", "high"}:
        return "high"
    if normalized in {"", "none", "minimal"}:
        return DEFAULT_DEEPSEEK_REASONING_EFFORT
    logger.warning(
        "Unsupported deepseek_reasoning_effort=%s; using %s",
        value,
        DEFAULT_DEEPSEEK_REASONING_EFFORT,
    )
    return DEFAULT_DEEPSEEK_REASONING_EFFORT


def _normalize_passive_symbols(value: Any) -> list[str]:
    raw_symbols = value if isinstance(value, (list, tuple)) else DEFAULT_PASSIVE_SYMBOLS
    symbols: list[str] = []
    for raw_symbol in raw_symbols:
        symbol = _normalize_symbol(raw_symbol)
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    if len(symbols) != 4:
        logger.warning("passive_symbols must contain 4 symbols; using defaults")
        return list(DEFAULT_PASSIVE_SYMBOLS)
    return symbols


def _load_strategy_config() -> Dict[str, Any]:
    raw = load_runtime_config(CONFIG_PATH)
    passive_symbols = _normalize_passive_symbols(raw.get("passive_symbols"))
    return {
        "cycle_interval_seconds": _normalize_positive_int(raw.get("cycle_interval_seconds", 60), 60),
        "trigger_pct_usdt": _normalize_trigger_percent(
            raw.get("trigger_pct_usdt", DEFAULT_TRIGGER_PCT_USDT),
            DEFAULT_TRIGGER_PCT_USDT,
        ),
        "fixed_leverage": _normalize_positive_int(
            raw.get("fixed_leverage", DEFAULT_FIXED_LEVERAGE),
            DEFAULT_FIXED_LEVERAGE,
        ),
        "passive_leverage": _normalize_positive_int(
            raw.get("passive_leverage", DEFAULT_PASSIVE_LEVERAGE),
            DEFAULT_PASSIVE_LEVERAGE,
        ),
        "active_leverage": _normalize_positive_int(
            raw.get("active_leverage", DEFAULT_ACTIVE_LEVERAGE),
            DEFAULT_ACTIVE_LEVERAGE,
        ),
        "stop_loss_pct": _normalize_ratio(raw.get("stop_loss_pct", 0.04), 0.04),
        "capital_usage_ratio": min(
            1.0,
            _normalize_ratio(raw.get("capital_usage_ratio", DEFAULT_CAPITAL_USAGE_RATIO), DEFAULT_CAPITAL_USAGE_RATIO),
        ),
        "rebalance_threshold_pct": _normalize_ratio(
            raw.get("rebalance_threshold_pct", DEFAULT_REBALANCE_THRESHOLD_PCT),
            DEFAULT_REBALANCE_THRESHOLD_PCT,
        ),
        "active_rescreen_interval_hours": _normalize_positive_float(
            raw.get("active_rescreen_interval_hours", DEFAULT_ACTIVE_RESCREEN_INTERVAL_HOURS),
            DEFAULT_ACTIVE_RESCREEN_INTERVAL_HOURS,
        ),
        "ai_prompt_timeframe": str(
            raw.get("ai_prompt_timeframe", DEFAULT_AI_PROMPT_TIMEFRAME) or DEFAULT_AI_PROMPT_TIMEFRAME
        ).strip()
        or DEFAULT_AI_PROMPT_TIMEFRAME,
        "ai_prompt_candle_count": _normalize_positive_int(
            raw.get("ai_prompt_candle_count", DEFAULT_AI_PROMPT_CANDLE_COUNT),
            DEFAULT_AI_PROMPT_CANDLE_COUNT,
        ),
        "deepseek_model": str(raw.get("deepseek_model") or DEFAULT_DEEPSEEK_MODEL).strip()
        or DEFAULT_DEEPSEEK_MODEL,
        "deepseek_reasoning_effort": _normalize_reasoning_effort(raw.get("deepseek_reasoning_effort")),
        "deepseek_max_tokens": _normalize_positive_int(
            raw.get("deepseek_max_tokens", DEFAULT_DEEPSEEK_MAX_TOKENS),
            DEFAULT_DEEPSEEK_MAX_TOKENS,
        ),
        "deepseek_timeout_seconds": _normalize_positive_float(
            raw.get("deepseek_timeout_seconds", DEFAULT_DEEPSEEK_TIMEOUT_SECONDS),
            DEFAULT_DEEPSEEK_TIMEOUT_SECONDS,
        ),
        "passive_symbols": passive_symbols,
        "screener_quote": str(raw.get("screener_quote") or "USDT").strip().upper() or "USDT",
        "screener_timeout": _normalize_positive_float(raw.get("screener_timeout", 30.0), 30.0),
        "screener_retries": max(0, _safe_int(raw.get("screener_retries", 3), 3)),
        "screener_request_sleep": max(0.0, _safe_float(raw.get("screener_request_sleep", 0.10), 0.10) or 0.0),
    }


def _build_portfolio_slots(config: Dict[str, Any]) -> list[PortfolioSlot]:
    passive_symbols = list(config["passive_symbols"])
    passive_labels = ["passive_cl", "passive_copper", "passive_qqq", "passive_btc"]
    slots = [
        PortfolioSlot(
            slot_id=slot_id,
            label=symbol,
            kind="passive",
            symbol=symbol,
            target_margin_ratio=0.125,
        )
        for slot_id, symbol in zip(passive_labels, passive_symbols)
    ]
    slots.extend(
        [
            PortfolioSlot(
                slot_id="active_1",
                label="active1",
                kind="active",
                target_margin_ratio=0.125,
                active_screening_mode="crypto",
            ),
            PortfolioSlot(
                slot_id="active_2",
                label="active2",
                kind="active",
                target_margin_ratio=0.125,
                active_screening_mode="tradfi",
            ),
            PortfolioSlot(
                slot_id="active_3",
                label="active3",
                kind="active",
                target_margin_ratio=0.125,
                active_screening_mode="crypto",
            ),
            PortfolioSlot(
                slot_id="active_4",
                label="active4",
                kind="active",
                target_margin_ratio=0.125,
                active_screening_mode="tradfi",
            ),
        ]
    )
    return slots


def _leverage_for_slot(slot: PortfolioSlot, config: Dict[str, Any]) -> int:
    if slot.kind == "passive":
        return _normalize_positive_int(
            config.get("passive_leverage", DEFAULT_PASSIVE_LEVERAGE),
            DEFAULT_PASSIVE_LEVERAGE,
        )
    if slot.kind == "active":
        return _normalize_positive_int(
            config.get("active_leverage", DEFAULT_ACTIVE_LEVERAGE),
            DEFAULT_ACTIVE_LEVERAGE,
        )
    logger.warning("Unknown slot kind for leverage resolution: slot_id=%s kind=%s", slot.slot_id, slot.kind)
    return DEFAULT_ACTIVE_LEVERAGE


def _current_time_ms() -> int:
    return int(time.time() * 1000)


def _resolve_as_of_ms(as_of_ms: Optional[int]) -> int:
    resolved = _safe_int(as_of_ms, 0)
    return resolved if resolved > 0 else _current_time_ms()


def _utc_iso_from_ms(as_of_ms: Optional[int]) -> str:
    resolved = _resolve_as_of_ms(as_of_ms)
    return datetime.fromtimestamp(resolved / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc_iso_ms(value: Any) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _normalize_trigger_price(value: Any) -> Optional[float]:
    parsed = _safe_float(value, None)
    if parsed is None or parsed <= 0.0:
        return None
    return float(round(float(parsed), TRIGGER_PRICE_DIGITS))


def _format_price(value: Any) -> Optional[float]:
    parsed = _safe_float(value, None)
    if parsed is None or parsed <= 0.0:
        return None
    return float(parsed)


def _normalize_ai_decision(value: Any) -> Optional[str]:
    normalized = str(value or "").strip().upper()
    return normalized if normalized in MANAGED_DECISIONS else None


def _decision_to_position_direction(value: Any) -> Optional[str]:
    normalized = _normalize_ai_decision(value)
    if normalized == "LONG":
        return "long"
    if normalized == "SHORT":
        return "short"
    return None


def _decision_to_order_side(value: Any) -> Optional[str]:
    normalized = _normalize_ai_decision(value)
    if normalized == "LONG":
        return "Buy"
    if normalized == "SHORT":
        return "Sell"
    return None


def _create_cycle_dir(base_dir: str = DB_DIR) -> str:
    os.makedirs(base_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    cycle_dir = os.path.join(base_dir, f"{timestamp}_{int(time.time() * 1_000_000) % 1_000_000:06d}Z")
    os.makedirs(cycle_dir, exist_ok=True)
    _prune_old_cycle_dirs(base_dir)
    return cycle_dir


def _prune_old_cycle_dirs(base_dir: str, *, max_dirs: int = MAX_DB_CYCLE_DIRS) -> None:
    if max_dirs < 1:
        return
    try:
        child_dirs = [
            (entry.stat(follow_symlinks=False).st_mtime, entry.name, entry.path)
            for entry in os.scandir(base_dir)
            if entry.is_dir(follow_symlinks=False)
        ]
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning("Failed to scan cycle directories in %s: %s", base_dir, exc)
        return
    overflow_count = len(child_dirs) - int(max_dirs)
    if overflow_count <= 0:
        return
    child_dirs.sort(key=lambda item: (item[0], item[1]))
    for _, _, dir_path in child_dirs[:overflow_count]:
        try:
            shutil.rmtree(dir_path)
        except OSError as exc:
            logger.warning("Failed to remove old cycle directory %s: %s", dir_path, exc)


def _slot_artifact_dir(cycle_dir: str, slot_id: str) -> str:
    path = os.path.join(cycle_dir, str(slot_id or "slot"))
    os.makedirs(path, exist_ok=True)
    return path


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False, default=str)


def _persist_cycle_output(result: Dict[str, Any]) -> None:
    cycle_dir = str(result.get("cycle_dir") or "").strip()
    if not cycle_dir:
        return
    try:
        _write_json(os.path.join(cycle_dir, "portfolio_cycle_output.json"), result)
    except Exception as exc:
        logger.warning("Failed to persist cycle output for %s: %s", cycle_dir, exc)


def _persist_screener_output(slot_dir: str, payload: Dict[str, Any]) -> None:
    try:
        _write_json(os.path.join(slot_dir, "active_screener_output.json"), payload)
    except Exception as exc:
        logger.warning("Failed to persist active screener output for %s: %s", slot_dir, exc)


def _has_changed_stop_sync(payload: Dict[str, Any]) -> bool:
    stop_sync = payload.get("stop_sync")
    return isinstance(stop_sync, dict) and bool(stop_sync.get("changed"))


def _has_material_position_record(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    action = str(payload.get("action") or "").strip()
    if action in MATERIAL_POSITION_RECORD_ACTIONS or _has_changed_stop_sync(payload):
        return True
    for child_key in ("execution", "close"):
        if _has_material_position_record(payload.get(child_key)):
            return True
    return False


def _should_persist_cycle_output(result: Dict[str, Any]) -> bool:
    if bool(result.get("ai_triggered")):
        return True
    if bool(result.get("screening_triggered")):
        return True
    if result.get("unmanaged_position_closes"):
        return True
    slot_results = result.get("slot_results")
    if not isinstance(slot_results, list):
        return False
    return any(_has_material_position_record(slot_result) for slot_result in slot_results)


def _emit_notification(notification_callback: NotificationCallback, event_name: str, payload: Dict[str, Any]) -> None:
    if not callable(notification_callback):
        return
    try:
        notification_callback(event_name, dict(payload or {}))
    except Exception as exc:
        logger.warning("Notification callback failed | event=%s error=%s", event_name, exc)


def _serialize_close_prices(
    candles: Sequence[Dict[str, Any]],
    *,
    limit: int,
    latest_price_override: Optional[float] = None,
) -> list[float]:
    visible = list(candles or [])[-limit:]
    close_prices: list[float] = []
    for candle in visible:
        close_price = _format_price(candle.get("close"))
        if close_price is None:
            raise ValueError("invalid candle found while serializing close-price data")
        close_prices.append(close_price)
    live_price = _format_price(latest_price_override)
    if close_prices and live_price is not None:
        close_prices[-1] = live_price
    return close_prices


def _fetch_prompt_market_context(
    *,
    symbol: str,
    ai_prompt_timeframe: str,
    ai_prompt_candle_count: int,
    as_of_ms: Optional[int],
    reference_price: Optional[float] = None,
) -> Dict[str, Any]:
    resolved_timeframe = str(ai_prompt_timeframe or DEFAULT_AI_PROMPT_TIMEFRAME).strip() or DEFAULT_AI_PROMPT_TIMEFRAME
    resolved_count = max(1, int(ai_prompt_candle_count))
    raw_klines = fetch_klines(
        symbol,
        resolved_timeframe,
        resolved_count + 2,
        as_of_ms=_resolve_as_of_ms(as_of_ms),
    )
    candles = parse_klines(raw_klines)
    prompt_candles = list(candles or [])[-resolved_count:]
    if len(prompt_candles) < resolved_count:
        raise ValueError(f"not enough prompt candles for {symbol}: have={len(prompt_candles)} need={resolved_count}")
    return {
        "timeframes": {
            resolved_timeframe: _serialize_close_prices(
                prompt_candles,
                limit=resolved_count,
                latest_price_override=reference_price,
            )
        },
        "ai_prompt_timeframe": resolved_timeframe,
        "ai_prompt_candle_count": resolved_count,
    }


def _build_trigger_levels(anchor_price: float, trigger_pct_usdt: float) -> Dict[str, float]:
    normalized_anchor_price = _normalize_trigger_price(anchor_price)
    if normalized_anchor_price is None:
        raise ValueError("anchor price must be positive")
    trigger_ratio = float(trigger_pct_usdt) / 100.0
    next_trigger_down = _normalize_trigger_price(normalized_anchor_price * (1.0 - trigger_ratio))
    next_trigger_up = _normalize_trigger_price(normalized_anchor_price * (1.0 + trigger_ratio))
    if next_trigger_down is None or next_trigger_up is None or next_trigger_down >= next_trigger_up:
        raise ValueError("trigger percent produced invalid price levels")
    return {
        "trigger_price": normalized_anchor_price,
        "next_trigger_down": next_trigger_down,
        "next_trigger_up": next_trigger_up,
    }


def _determine_ai_trigger(
    *,
    has_position: bool,
    current_price: float,
    slot_state: Dict[str, Any],
    trigger_pct_usdt: float,
) -> Dict[str, Any]:
    current_trigger_price = _normalize_trigger_price(current_price)
    if current_trigger_price is None:
        raise ValueError("current_price must be positive")
    if not has_position:
        next_levels = _build_trigger_levels(current_price, trigger_pct_usdt)
        return {
            "should_trigger": True,
            "reason": "no_position",
            **next_levels,
        }
    if _normalize_ai_decision(slot_state.get("last_ai_decision")) is None:
        next_levels = _build_trigger_levels(current_price, trigger_pct_usdt)
        return {
            "should_trigger": True,
            "reason": "missing_ai_decision",
            **next_levels,
        }

    active_trigger_down = _normalize_trigger_price(slot_state.get("next_trigger_down"))
    active_trigger_up = _normalize_trigger_price(slot_state.get("next_trigger_up"))
    has_valid_window = active_trigger_down is not None and active_trigger_up is not None and active_trigger_down < active_trigger_up
    if not has_valid_window:
        last_anchor = _normalize_trigger_price(slot_state.get("last_ai_trigger_price"))
        if last_anchor is not None:
            anchor_levels = _build_trigger_levels(last_anchor, trigger_pct_usdt)
            active_trigger_down = anchor_levels["next_trigger_down"]
            active_trigger_up = anchor_levels["next_trigger_up"]
            has_valid_window = True

    if not has_valid_window:
        next_levels = _build_trigger_levels(current_price, trigger_pct_usdt)
        return {
            "should_trigger": True,
            "reason": "missing_llm_anchor",
            **next_levels,
        }

    if current_price >= float(active_trigger_up):
        return {
            "should_trigger": True,
            "reason": "price_distance_reached",
            "trigger_direction": "up",
            **_build_trigger_levels(current_price, trigger_pct_usdt),
        }
    if current_price <= float(active_trigger_down):
        return {
            "should_trigger": True,
            "reason": "price_distance_reached",
            "trigger_direction": "down",
            **_build_trigger_levels(current_price, trigger_pct_usdt),
        }
    return {
        "should_trigger": False,
        "reason": "waiting_for_next_price_trigger",
        "trigger_price": None,
        "next_trigger_down": active_trigger_down,
        "next_trigger_up": active_trigger_up,
    }


def _empty_slot_state(slot: PortfolioSlot) -> Dict[str, Any]:
    state = {
        "slot_id": slot.slot_id,
        "kind": slot.kind,
        "symbol": slot.symbol if slot.kind == "passive" else None,
        "last_ai_trigger_price": None,
        "last_ai_triggered_at": None,
        "last_ai_decision": None,
        "next_trigger_down": None,
        "next_trigger_up": None,
    }
    if slot.kind == "active":
        state["entered_at"] = None
        state["last_active_rank_checked_at"] = None
        state["previous_active_symbol"] = None
    return state


def _active_screening_mode(slot: PortfolioSlot) -> str:
    return "tradfi" if str(slot.active_screening_mode or "").strip().lower() == "tradfi" else "crypto"


def _normalize_slot_state(slot: PortfolioSlot, raw_state: Any) -> Dict[str, Any]:
    state = _empty_slot_state(slot)
    if isinstance(raw_state, dict):
        preserved_keys = [
            "last_ai_trigger_price",
            "last_ai_triggered_at",
            "last_ai_decision",
            "next_trigger_down",
            "next_trigger_up",
        ]
        if slot.kind == "active":
            preserved_keys.extend(["entered_at", "last_active_rank_checked_at"])
        for key in preserved_keys:
            if key in raw_state:
                state[key] = raw_state.get(key)
        if slot.kind == "active":
            state["symbol"] = _normalize_symbol(raw_state.get("symbol"))
            state["previous_active_symbol"] = _normalize_symbol(
                raw_state.get("previous_active_symbol")
            )
    if slot.kind == "passive":
        state["symbol"] = slot.symbol
    state["last_ai_decision"] = _normalize_ai_decision(state.get("last_ai_decision"))
    return state


def _active_slot_memory_symbols(slot_state: Dict[str, Any]) -> list[str]:
    symbols: list[str] = []
    for key in ("symbol", "previous_active_symbol"):
        symbol = _normalize_symbol(slot_state.get(key))
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _active_recent_symbols_by_mode(
    slots: Sequence[PortfolioSlot],
    portfolio_state: Dict[str, Any],
) -> dict[str, set[str]]:
    raw_slots_state = portfolio_state.get("slots")
    slots_state = raw_slots_state if isinstance(raw_slots_state, dict) else {}
    grouped: dict[str, set[str]] = {}
    for slot in slots:
        if slot.kind != "active":
            continue
        slot_state = slots_state.get(slot.slot_id)
        if not isinstance(slot_state, dict):
            continue
        grouped.setdefault(_active_screening_mode(slot), set()).update(
            _active_slot_memory_symbols(slot_state)
        )
    return grouped


def _normalize_portfolio_state(previous_state: Optional[Dict[str, Any]], slots: Sequence[PortfolioSlot]) -> Dict[str, Any]:
    raw = dict(previous_state or {})
    raw_slots = raw.get("slots") if str(raw.get("version") or "") == STATE_VERSION else {}
    if not isinstance(raw_slots, dict):
        raw_slots = {}
    normalized_slots: dict[str, Dict[str, Any]] = {}
    reserved_symbols: set[str] = set()
    for slot in slots:
        slot_state = _normalize_slot_state(slot, raw_slots.get(slot.slot_id))
        state_symbol = _normalize_symbol(slot_state.get("symbol"))
        if slot.kind == "active" and state_symbol in reserved_symbols:
            slot_state = _clear_active_slot_state(slot_state)
            state_symbol = None
        if state_symbol:
            reserved_symbols.add(state_symbol)
        normalized_slots[slot.slot_id] = slot_state
    return {
        "version": STATE_VERSION,
        "slots": normalized_slots,
        "last_cycle_time": raw.get("last_cycle_time"),
        "last_minute_slot": raw.get("last_minute_slot"),
        "last_cycle_result": raw.get("last_cycle_result"),
    }


def _update_slot_trigger_state(
    slot_state: Dict[str, Any],
    *,
    ai_triggered: bool,
    trigger_info: Dict[str, Any],
    ai_decision: Optional[str] = None,
    symbol: Optional[str] = None,
    update_anchor: bool = False,
) -> Dict[str, Any]:
    updated = dict(slot_state)
    if symbol is not None:
        normalized_symbol = _normalize_symbol(symbol)
        current_symbol = _normalize_symbol(updated.get("symbol"))
        if normalized_symbol and current_symbol and normalized_symbol != current_symbol:
            updated["previous_active_symbol"] = current_symbol
        updated["symbol"] = normalized_symbol
    if ai_triggered or update_anchor:
        updated["last_ai_trigger_price"] = _normalize_trigger_price(trigger_info.get("trigger_price"))
        updated["last_ai_triggered_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    normalized_decision = _normalize_ai_decision(ai_decision)
    if normalized_decision:
        updated["last_ai_decision"] = normalized_decision
    updated["next_trigger_down"] = _normalize_trigger_price(trigger_info.get("next_trigger_down"))
    updated["next_trigger_up"] = _normalize_trigger_price(trigger_info.get("next_trigger_up"))
    return updated


def _clear_active_slot_state(slot_state: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(slot_state)
    previous_symbol = _normalize_symbol(updated.get("symbol")) or _normalize_symbol(
        updated.get("previous_active_symbol")
    )
    updated["symbol"] = None
    updated["last_ai_trigger_price"] = None
    updated["last_ai_triggered_at"] = None
    updated["last_ai_decision"] = None
    updated["next_trigger_down"] = None
    updated["next_trigger_up"] = None
    updated["entered_at"] = None
    updated["last_active_rank_checked_at"] = None
    updated["previous_active_symbol"] = previous_symbol
    return updated


def _active_rescreen_interval_ms(config: Dict[str, Any]) -> int:
    hours = _normalize_positive_float(
        config.get("active_rescreen_interval_hours", DEFAULT_ACTIVE_RESCREEN_INTERVAL_HOURS),
        DEFAULT_ACTIVE_RESCREEN_INTERVAL_HOURS,
    )
    return max(1, int(hours * 60.0 * 60.0 * 1000.0))


def _active_rank_review_reference_ms(slot_state: Dict[str, Any]) -> Optional[int]:
    return _parse_utc_iso_ms(slot_state.get("last_active_rank_checked_at")) or _parse_utc_iso_ms(
        slot_state.get("entered_at")
    )


def _active_rank_review_due(
    slot_state: Dict[str, Any],
    *,
    config: Dict[str, Any],
    as_of_ms: int,
) -> bool:
    reference_ms = _active_rank_review_reference_ms(slot_state)
    if reference_ms is None:
        return False
    elapsed_ms = max(0, int(as_of_ms) - int(reference_ms))
    return elapsed_ms >= _active_rescreen_interval_ms(config)


def _ensure_active_entry_timestamp(
    slot_state: Dict[str, Any],
    *,
    as_of_ms: int,
) -> tuple[Dict[str, Any], bool]:
    if str(slot_state.get("entered_at") or "").strip():
        return dict(slot_state), False
    now_iso = _utc_iso_from_ms(as_of_ms)
    updated = dict(slot_state)
    updated["entered_at"] = now_iso
    updated["last_active_rank_checked_at"] = now_iso
    return updated, True


def _mark_active_rank_checked(
    slot_state: Dict[str, Any],
    *,
    as_of_ms: int,
) -> Dict[str, Any]:
    updated = dict(slot_state)
    updated["last_active_rank_checked_at"] = _utc_iso_from_ms(as_of_ms)
    return updated


def _mark_active_entry_opened(
    slot_state: Dict[str, Any],
    *,
    as_of_ms: int,
) -> Dict[str, Any]:
    now_iso = _utc_iso_from_ms(as_of_ms)
    updated = dict(slot_state)
    updated["entered_at"] = now_iso
    updated["last_active_rank_checked_at"] = now_iso
    return updated


def _position_direction(position: Optional[Dict[str, Any]]) -> Optional[str]:
    metrics = calculate_position_metrics(position)
    direction = str(metrics.get("direction") or "").strip().lower()
    return direction if direction in {"long", "short"} else None


def _position_side(position: Optional[Dict[str, Any]]) -> Optional[str]:
    metrics = calculate_position_metrics(position)
    side = str(metrics.get("side") or "").strip()
    return side or None


def _position_size(position: Optional[Dict[str, Any]]) -> float:
    metrics = calculate_position_metrics(position)
    return abs(_safe_float(metrics.get("size"), 0.0) or 0.0)


def _position_symbol(position: Optional[Dict[str, Any]]) -> Optional[str]:
    metrics = calculate_position_metrics(position)
    return _normalize_symbol(metrics.get("symbol") or (position or {}).get("symbol"))


def _position_notional(position: Optional[Dict[str, Any]], reference_price: Optional[float] = None) -> float:
    metrics = calculate_position_metrics(position)
    size = abs(_safe_float(metrics.get("size"), 0.0) or 0.0)
    ref = _format_price(reference_price)
    if size > 0.0 and ref is not None:
        return float(size * ref)
    return abs(_safe_float(metrics.get("position_value"), 0.0) or 0.0)


def _positions_by_symbol(positions: Sequence[Dict[str, Any]]) -> dict[str, Dict[str, Any]]:
    mapped: dict[str, Dict[str, Any]] = {}
    for position in positions:
        symbol = _position_symbol(position)
        if symbol:
            mapped[symbol] = dict(position)
    return mapped


def _position_leverage(position: Optional[Dict[str, Any]]) -> Optional[int]:
    parsed = _safe_float(calculate_position_metrics(position).get("leverage"), None)
    if parsed is None or parsed <= 0.0:
        return None
    return int(parsed)


def _ensure_symbol_leverage(
    *,
    api_key: str,
    api_secret: str,
    symbol: str,
    leverage: int,
    current_position: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_symbol = _normalize_symbol(symbol)
    requested_leverage = _normalize_positive_int(leverage, DEFAULT_ACTIVE_LEVERAGE)
    if not normalized_symbol:
        return {
            "success": False,
            "action": "invalid_symbol_for_leverage",
            "requested_leverage": requested_leverage,
        }

    current_leverage = _position_leverage(current_position)
    if current_leverage == requested_leverage:
        return {
            "success": True,
            "action": "leverage_already_configured",
            "symbol": normalized_symbol,
            "requested_leverage": requested_leverage,
            "actual_leverage": current_leverage,
        }

    try:
        actual_leverage = set_leverage(api_key, api_secret, normalized_symbol, requested_leverage)
    except Exception as exc:
        logger.error(
            "set_leverage(%s) raised before slot execution | requested_leverage=%s error=%s",
            normalized_symbol,
            requested_leverage,
            exc,
        )
        return {
            "success": False,
            "action": "set_leverage_failed",
            "symbol": normalized_symbol,
            "requested_leverage": requested_leverage,
            "current_leverage": current_leverage,
            "error": str(exc),
        }

    actual_leverage_int = _normalize_positive_int(actual_leverage, 0)
    if actual_leverage_int != requested_leverage:
        return {
            "success": False,
            "action": "set_leverage_failed",
            "symbol": normalized_symbol,
            "requested_leverage": requested_leverage,
            "current_leverage": current_leverage,
            "actual_leverage": actual_leverage,
            "error": "leverage_mismatch",
        }

    return {
        "success": True,
        "action": "set_leverage_configured",
        "symbol": normalized_symbol,
        "requested_leverage": requested_leverage,
        "previous_leverage": current_leverage,
        "actual_leverage": actual_leverage_int,
    }


def _resolve_fixed_stop_loss_price(
    *,
    direction: str,
    entry_price: float,
    stop_loss_pct: float,
) -> Optional[float]:
    if entry_price <= 0.0 or stop_loss_pct <= 0.0:
        return None
    normalized_direction = str(direction or "").strip().lower()
    if normalized_direction == "long":
        return float(entry_price * (1.0 - stop_loss_pct))
    if normalized_direction == "short":
        return float(entry_price * (1.0 + stop_loss_pct))
    return None


def _sync_fixed_stop_loss(
    *,
    api_key: str,
    api_secret: str,
    symbol: str,
    position: Dict[str, Any],
    stop_loss_pct: float,
) -> Dict[str, Any]:
    metrics = calculate_position_metrics(position)
    direction = str(metrics.get("direction") or "").strip().lower()
    side = str(metrics.get("side") or "").strip()
    entry_price = _format_price(metrics.get("entry_price"))
    current_stop = _format_price(metrics.get("stop_loss"))
    if direction not in {"long", "short"} or not side or entry_price is None:
        return {"success": False, "changed": False, "reason": "invalid_position_for_stop_sync"}
    stop_loss = _resolve_fixed_stop_loss_price(
        direction=direction,
        entry_price=entry_price,
        stop_loss_pct=stop_loss_pct,
    )
    if stop_loss is None:
        return {"success": False, "changed": False, "reason": "invalid_stop_loss"}
    sync_result = sync_existing_position_stop_loss(
        api_key,
        api_secret,
        symbol,
        side,
        stop_loss=stop_loss,
        current_stop_loss=current_stop,
    )
    enriched = dict(sync_result)
    enriched["stop_loss_pct"] = float(stop_loss_pct)
    enriched.setdefault("stop_loss", stop_loss)
    return enriched


def _build_entry_order_plan(
    *,
    symbol: str,
    desired_notional_usdt: float,
    reference_price: float,
    max_qty: Optional[float] = None,
) -> Dict[str, Any]:
    plan: Dict[str, Any] = {
        "qty": None,
        "order_notional_usdt": None,
        "min_notional_usdt": None,
        "meets_min_notional": True,
    }
    if desired_notional_usdt <= 0.0 or reference_price <= 0.0:
        return plan
    desired_notional_decimal = safe_decimal(str(desired_notional_usdt))
    reference_price_decimal = safe_decimal(str(reference_price))
    if desired_notional_decimal <= 0 or reference_price_decimal <= 0:
        return plan
    qty_value = desired_notional_decimal / reference_price_decimal
    if max_qty is not None and max_qty > 0.0:
        qty_value = min(qty_value, safe_decimal(str(max_qty)))
    adjusted_qty = adjust_qty_for_symbol(symbol, qty_value, enforce_min_qty=False)
    if adjusted_qty is None or adjusted_qty <= 0:
        return plan
    notional_check = evaluate_entry_order_notional(symbol, adjusted_qty, reference_price)
    order_notional = _safe_float(notional_check.get("order_notional"), None)
    min_notional = _safe_float(notional_check.get("min_notional"), None)
    plan["qty"] = decimal_to_str(adjusted_qty)
    if order_notional is not None and order_notional > 0.0:
        plan["order_notional_usdt"] = float(order_notional)
    if min_notional is not None and min_notional > 0.0:
        plan["min_notional_usdt"] = float(min_notional)
    plan["meets_min_notional"] = bool(notional_check.get("meets_min_notional", True))
    return plan


def _target_notional_usdt(
    *,
    account_equity: float,
    slot: PortfolioSlot,
    capital_usage_ratio: float,
    leverage: int,
) -> float:
    return float(account_equity * float(capital_usage_ratio) * float(slot.target_margin_ratio) * float(leverage))


def _available_notional_cap(*, available_balance: float, capital_usage_ratio: float, leverage: int) -> float:
    return max(0.0, float(available_balance) * float(capital_usage_ratio) * float(leverage))


def _place_direction_position(
    *,
    api_key: str,
    api_secret: str,
    symbol: str,
    decision: str,
    target_notional_usdt: float,
    reference_price: float,
    leverage: int,
    available_notional_cap: Optional[float] = None,
) -> Dict[str, Any]:
    normalized_decision = _normalize_ai_decision(decision)
    side = _decision_to_order_side(normalized_decision)
    if normalized_decision not in MANAGED_DECISIONS or side is None:
        return {"success": False, "action": "invalid_ai_decision"}
    desired_notional = float(target_notional_usdt)
    if available_notional_cap is not None and available_notional_cap > 0.0:
        desired_notional = min(desired_notional, float(available_notional_cap))
    if desired_notional <= 0.0:
        return {"success": False, "action": "insufficient_available_balance"}

    order_plan = _build_entry_order_plan(
        symbol=symbol,
        desired_notional_usdt=desired_notional,
        reference_price=reference_price,
    )
    qty = order_plan.get("qty")
    if not qty:
        return {"success": False, "action": "target_qty_below_min", "order_plan": order_plan}
    if not bool(order_plan.get("meets_min_notional", True)):
        return {
            "success": True,
            "action": "skipped_entry_below_min_notional",
            "qty": qty,
            "requested_notional_usdt": desired_notional,
            "order_plan": order_plan,
        }

    order, code, msg = place_market_entry_order(
        api_key,
        api_secret,
        symbol,
        side,
        qty,
        leverage=leverage,
    )
    if order is None:
        failure_action = "set_leverage_failed" if str(msg or "") == "set_leverage_failed" else "entry_order_failed"
        return {
            "success": False,
            "action": failure_action,
            "order_error_code": code,
            "order_error_message": msg,
            "qty": qty,
            "leverage": leverage,
            "requested_notional_usdt": desired_notional,
            "order_plan": order_plan,
        }
    return {
        "success": True,
        "action": "opened_new_position",
        "ai_decision": normalized_decision,
        "side": side,
        "order": order,
        "qty": qty,
        "leverage": leverage,
        "requested_notional_usdt": desired_notional,
        "order_plan": order_plan,
    }


def _close_existing_position(
    *,
    api_key: str,
    api_secret: str,
    position: Dict[str, Any],
    context: str,
) -> Dict[str, Any]:
    symbol = _position_symbol(position)
    side = _position_side(position)
    size = _position_size(position)
    if not symbol or not side or size <= 0.0:
        return {"success": False, "action": "invalid_existing_position"}
    close_ok = close_position(api_key, api_secret, symbol, side, str(size))
    if not close_ok:
        return {"success": False, "action": "close_position_failed", "symbol": symbol}
    wait_for_close_propagation(api_key, api_secret, [symbol], context=context)
    cancel_all_symbol_orders(api_key, api_secret, symbol)
    return {"success": True, "action": "closed_position", "symbol": symbol, "qty": size}


def _reverse_existing_position(
    *,
    api_key: str,
    api_secret: str,
    symbol: str,
    position: Dict[str, Any],
    decision: str,
    target_notional_usdt: float,
    reference_price: float,
    leverage: int,
    available_notional_cap: float,
    stop_loss_pct: float,
) -> Dict[str, Any]:
    normalized_decision = _normalize_ai_decision(decision)
    if normalized_decision not in MANAGED_DECISIONS:
        return {"success": False, "action": "invalid_ai_decision"}

    close_result = _close_existing_position(
        api_key=api_key,
        api_secret=api_secret,
        position=position,
        context="reverse_position",
    )
    if not bool(close_result.get("success")):
        return {"success": False, "action": "reverse_close_failed", "close": close_result}

    entry_result = _place_direction_position(
        api_key=api_key,
        api_secret=api_secret,
        symbol=symbol,
        decision=str(normalized_decision),
        target_notional_usdt=target_notional_usdt,
        reference_price=reference_price,
        leverage=leverage,
        available_notional_cap=available_notional_cap,
    )
    if not bool(entry_result.get("success")):
        entry_result["action"] = "reverse_reopen_failed"
        return entry_result

    synced = _sync_position_after_trade(
        api_key=api_key,
        api_secret=api_secret,
        symbol=symbol,
        stop_loss_pct=stop_loss_pct,
        expected_decision=normalized_decision,
    )
    entry_result["action"] = "reversed_position"
    return _merge_post_trade_sync_result(entry_result, synced)


def _sync_position_after_trade(
    *,
    api_key: str,
    api_secret: str,
    symbol: str,
    stop_loss_pct: float,
    expected_decision: Optional[str] = None,
) -> Dict[str, Any]:
    expected_direction = _decision_to_position_direction(expected_decision)
    synced_position = get_position_snapshot(api_key, api_secret, symbol, retries=8, sleep_seconds=0.5)
    if not isinstance(synced_position, dict):
        verification = None
        if expected_direction:
            verification = {
                "success": False,
                "action": "post_trade_position_unavailable",
                "expected_direction": expected_direction,
                "actual_direction": None,
            }
        return {"position": None, "stop_sync": None, "direction_verification": verification}

    synced_metrics = calculate_position_metrics(synced_position)
    actual_direction = str(synced_metrics.get("direction") or "").strip().lower()
    if expected_direction and actual_direction != expected_direction:
        close_result = _close_existing_position(
            api_key=api_key,
            api_secret=api_secret,
            position=synced_position,
            context="post_trade_direction_mismatch",
        )
        close_ok = bool(close_result.get("success"))
        return {
            "position": synced_metrics,
            "stop_sync": None,
            "direction_verification": {
                "success": False,
                "action": "post_trade_direction_mismatch_closed" if close_ok else "post_trade_direction_mismatch_close_failed",
                "expected_direction": expected_direction,
                "actual_direction": actual_direction or None,
                "close": close_result,
            },
        }

    stop_sync = _sync_fixed_stop_loss(
        api_key=api_key,
        api_secret=api_secret,
        symbol=symbol,
        position=synced_position,
        stop_loss_pct=stop_loss_pct,
    )
    refreshed_position = get_position_snapshot(api_key, api_secret, symbol, retries=2, sleep_seconds=0.35) or synced_position
    refreshed_metrics = calculate_position_metrics(refreshed_position)
    return {
        "position": refreshed_metrics,
        "stop_sync": stop_sync,
        "direction_verification": {
            "success": True,
            "expected_direction": expected_direction,
            "actual_direction": str(refreshed_metrics.get("direction") or "").strip().lower() or None,
        }
        if expected_direction
        else None,
    }


def _merge_post_trade_sync_result(execution: Dict[str, Any], sync_result: Dict[str, Any]) -> Dict[str, Any]:
    execution.update(sync_result)
    verification = sync_result.get("direction_verification")
    if isinstance(verification, dict) and not bool(verification.get("success")):
        execution["success"] = False
        execution["action"] = str(verification.get("action") or "post_trade_direction_verification_failed")
    return execution


def _rebalance_existing_position(
    *,
    api_key: str,
    api_secret: str,
    symbol: str,
    position: Dict[str, Any],
    decision: str,
    target_notional_usdt: float,
    reference_price: float,
    leverage: int,
    available_notional_cap: float,
    rebalance_threshold_pct: float,
    stop_loss_pct: float,
) -> Dict[str, Any]:
    normalized_decision = _normalize_ai_decision(decision)
    desired_direction = _decision_to_position_direction(normalized_decision)
    if normalized_decision not in MANAGED_DECISIONS or desired_direction is None:
        return {"success": False, "action": "invalid_ai_decision"}
    current_direction = _position_direction(position)
    if current_direction not in {"long", "short"}:
        return {"success": False, "action": "invalid_existing_position"}
    if current_direction != desired_direction:
        return _reverse_existing_position(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            position=position,
            decision=str(normalized_decision),
            target_notional_usdt=target_notional_usdt,
            reference_price=reference_price,
            leverage=leverage,
            available_notional_cap=available_notional_cap,
            stop_loss_pct=stop_loss_pct,
        )

    current_notional = _position_notional(position, reference_price)
    target_notional = max(0.0, float(target_notional_usdt))
    if target_notional <= 0.0:
        return {"success": False, "action": "invalid_target_notional"}
    diff = target_notional - current_notional
    diff_ratio = abs(diff) / target_notional
    stop_sync = _sync_fixed_stop_loss(
        api_key=api_key,
        api_secret=api_secret,
        symbol=symbol,
        position=position,
        stop_loss_pct=stop_loss_pct,
    )
    if diff_ratio < float(rebalance_threshold_pct):
        return {
            "success": True,
            "action": "kept_position_size",
            "current_notional_usdt": current_notional,
            "target_notional_usdt": target_notional,
            "rebalance_diff_usdt": diff,
            "stop_sync": stop_sync,
        }
    if diff > 0.0:
        increase_notional = min(diff, max(0.0, float(available_notional_cap)))
        if increase_notional <= 0.0:
            return {
                "success": True,
                "action": "kept_position_size",
                "current_notional_usdt": current_notional,
                "target_notional_usdt": target_notional,
                "rebalance_diff_usdt": diff,
                "requested_rebalance_notional_usdt": increase_notional,
                "skipped_rebalance_reason": "insufficient_available_balance",
                "stop_sync": stop_sync,
            }
        entry_result = _place_direction_position(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            decision=str(normalized_decision),
            target_notional_usdt=increase_notional,
            reference_price=reference_price,
            leverage=leverage,
            available_notional_cap=increase_notional,
        )
        entry_action = str(entry_result.get("action") or "")
        if entry_action in {"target_qty_below_min", "skipped_entry_below_min_notional"}:
            return {
                "success": True,
                "action": "kept_position_size",
                "current_notional_usdt": current_notional,
                "target_notional_usdt": target_notional,
                "rebalance_diff_usdt": diff,
                "requested_rebalance_notional_usdt": increase_notional,
                "skipped_rebalance_reason": entry_action,
                "rebalance_order_plan": entry_result.get("order_plan"),
                "stop_sync": stop_sync,
            }
        if not bool(entry_result.get("success")):
            return entry_result
        synced = _sync_position_after_trade(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            stop_loss_pct=stop_loss_pct,
            expected_decision=normalized_decision,
        )
        entry_result["action"] = "increased_position"
        entry_result["current_notional_usdt"] = current_notional
        entry_result["target_notional_usdt"] = target_notional
        return _merge_post_trade_sync_result(entry_result, synced)

    reduce_notional = abs(diff)
    reduce_qty = reduce_notional / reference_price if reference_price > 0.0 else 0.0
    side = _position_side(position)
    if not side or reduce_qty <= 0.0:
        return {"success": False, "action": "invalid_rebalance_reduce_qty"}
    close_ok = close_position(api_key, api_secret, symbol, side, str(reduce_qty))
    if not close_ok:
        return {"success": False, "action": "rebalance_reduce_failed"}
    synced = _sync_position_after_trade(
        api_key=api_key,
        api_secret=api_secret,
        symbol=symbol,
        stop_loss_pct=stop_loss_pct,
    )
    return {
        "success": True,
        "action": "reduced_position",
        "current_notional_usdt": current_notional,
        "target_notional_usdt": target_notional,
        "reduced_notional_usdt": reduce_notional,
        **synced,
    }


def _screen_active_candidate(
    *,
    slot: PortfolioSlot,
    config: Dict[str, Any],
    excluded_symbols: Sequence[str],
) -> Dict[str, Any]:
    if _active_screening_mode(slot) == "tradfi":
        screener_output = screen_active_tradfi_symbol(
            excluded_symbols=excluded_symbols,
            quote=str(config["screener_quote"]),
            timeout=float(config["screener_timeout"]),
            retries=int(config["screener_retries"]),
            request_sleep=float(config["screener_request_sleep"]),
            required_kline_interval=str(config["ai_prompt_timeframe"]),
            required_kline_count=int(config["ai_prompt_candle_count"]),
        )
    else:
        screener_output = screen_active_symbol(
            excluded_symbols=excluded_symbols,
            quote=str(config["screener_quote"]),
            timeout=float(config["screener_timeout"]),
            retries=int(config["screener_retries"]),
            request_sleep=float(config["screener_request_sleep"]),
            required_kline_interval=str(config["ai_prompt_timeframe"]),
            required_kline_count=int(config["ai_prompt_candle_count"]),
        )
    selection = screener_output.get("selection") if isinstance(screener_output, dict) else {}
    selected_symbol = _normalize_symbol((selection or {}).get("symbol"))
    if not selected_symbol:
        raise NoActiveCandidateError("active screener did not return a tradable candidate")
    selected = selection.get("selected") if isinstance(selection, dict) else {}
    if not isinstance(selected, dict):
        selected = {}
    screening_decision = _normalize_ai_decision(
        (selection or {}).get("screening_decision") or selected.get("screening_decision")
    )
    if screening_decision is None:
        raise NoActiveCandidateError("active screener did not return a LONG/SHORT direction")
    return {
        "symbol": selected_symbol,
        "screening_decision": screening_decision,
        "screening_direction": str(
            (selection or {}).get("screening_direction") or selected.get("screening_direction") or ""
        ).strip().lower()
        or None,
        "close_prices": list(selected.get("close_prices") or []),
        "decision_source": "active_screener",
        "selection": selection,
        "metadata": screener_output.get("metadata", {}),
        "_screener_output": screener_output,
    }


def _apply_active_screening_decision(result: Dict[str, Any], candidate: Dict[str, Any]) -> Optional[str]:
    decision = _normalize_ai_decision(candidate.get("screening_decision"))
    result["screening_triggered"] = True
    result["screening_decision"] = decision
    result["screening_direction"] = candidate.get("screening_direction")
    result["decision_source"] = candidate.get("decision_source") or "active_screener"
    if decision is not None:
        result["ai_decision"] = None
    return decision


def _visible_active_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in dict(candidate or {}).items() if key != "_screener_output"}


def _active_screening_close_prices(candidate: Dict[str, Any], reference_price: Optional[float]) -> list[float]:
    values = candidate.get("close_prices")
    if not isinstance(values, (list, tuple)) or isinstance(values, (str, bytes)):
        selected = (candidate.get("selection") or {}).get("selected") if isinstance(candidate.get("selection"), dict) else {}
        values = selected.get("close_prices") if isinstance(selected, dict) else []
    close_prices: list[float] = []
    for value in values or []:
        parsed = _format_price(value)
        if parsed is not None:
            close_prices.append(parsed)
    live_price = _format_price(reference_price)
    if close_prices and live_price is not None:
        close_prices[-1] = live_price
    return close_prices


def _emit_active_screening_after(
    *,
    notification_callback: NotificationCallback,
    slot: PortfolioSlot,
    candidate: Dict[str, Any],
    result: Dict[str, Any],
    reference_price: float,
    trigger_info: Dict[str, Any],
    config: Dict[str, Any],
    position_before: Optional[Dict[str, Any]],
) -> None:
    _emit_notification(
        notification_callback,
        "active_screening_after",
        {
            "slot_id": slot.slot_id,
            "slot_label": slot.label,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "symbol": result.get("symbol") or candidate.get("symbol"),
            "candidate_symbol": result.get("candidate_symbol") or candidate.get("symbol"),
            "previous_symbol": result.get("previous_symbol"),
            "current_price": reference_price,
            "trigger_reason": trigger_info.get("reason"),
            "trigger_price": trigger_info.get("trigger_price"),
            "next_trigger_down": trigger_info.get("next_trigger_down"),
            "next_trigger_up": trigger_info.get("next_trigger_up"),
            "decision": result.get("screening_decision"),
            "screening_decision": result.get("screening_decision"),
            "screening_direction": result.get("screening_direction"),
            "decision_source": result.get("decision_source") or "active_screener",
            "action": result.get("action"),
            "success": bool(result.get("success")),
            "execution": result.get("execution"),
            "close": result.get("close"),
            "position": result.get("position"),
            "position_before": calculate_position_metrics(position_before) if isinstance(position_before, dict) else None,
            "screener": result.get("screener"),
            "entered_at": result.get("entered_at"),
            "last_active_rank_checked_at": result.get("last_active_rank_checked_at"),
            "ai_prompt_timeframe": str(config.get("ai_prompt_timeframe") or DEFAULT_AI_PROMPT_TIMEFRAME),
            "ai_prompt_candle_count": int(config.get("ai_prompt_candle_count") or DEFAULT_AI_PROMPT_CANDLE_COUNT),
            "close_prices": _active_screening_close_prices(candidate, reference_price),
        },
    )


def _evaluate_slot_direction(
    *,
    slot: PortfolioSlot,
    symbol: str,
    reference_price: float,
    config: Dict[str, Any],
    as_of_ms: int,
    cycle_dir_factory: CycleDirFactory,
    notification_callback: NotificationCallback,
    position: Optional[Dict[str, Any]],
    trigger_info: Dict[str, Any],
    decision_mode: str,
) -> tuple[Optional[str], Dict[str, Any], Dict[str, Any]]:
    cycle_dir = cycle_dir_factory()
    slot_dir = _slot_artifact_dir(cycle_dir, slot.slot_id)
    market_context = _fetch_prompt_market_context(
        symbol=symbol,
        ai_prompt_timeframe=str(config["ai_prompt_timeframe"]),
        ai_prompt_candle_count=int(config["ai_prompt_candle_count"]),
        as_of_ms=as_of_ms,
        reference_price=reference_price,
    )
    ai_prompt_timeframe = str(market_context.get("ai_prompt_timeframe") or config["ai_prompt_timeframe"])
    close_prices = list((market_context.get("timeframes") or {}).get(ai_prompt_timeframe) or [])
    _emit_notification(
        notification_callback,
        "ai_cycle_before",
        {
            "slot_id": slot.slot_id,
            "slot_label": slot.label,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "symbol": symbol,
            "cycle_dir": slot_dir,
            "current_price": reference_price,
            "trigger_reason": trigger_info.get("reason"),
            "trigger_price": trigger_info.get("trigger_price"),
            "next_trigger_down": trigger_info.get("next_trigger_down"),
            "next_trigger_up": trigger_info.get("next_trigger_up"),
            "position": calculate_position_metrics(position) if isinstance(position, dict) else None,
            "decision_mode": decision_mode,
        },
    )
    ai_analysis: Dict[str, Any] = {}
    ai_decision = evaluate_trade_direction(
        cycle_dir=slot_dir,
        symbol=symbol,
        reference_price=reference_price,
        timeframe_ohlcv=market_context["timeframes"],
        reasoning_effort=str(config["deepseek_reasoning_effort"]),
        model=str(config["deepseek_model"]),
        max_tokens=int(config["deepseek_max_tokens"]),
        timeout_seconds=float(config["deepseek_timeout_seconds"]),
        analysis_sink=ai_analysis,
        decision_mode=decision_mode,
    )
    decision_value = str(getattr(ai_decision, "decision", "") or "").strip().upper() if ai_decision is not None else None
    if decision_value not in MANAGED_DECISIONS:
        decision_value = None
    _emit_notification(
        notification_callback,
        "ai_cycle_after",
        {
            "slot_id": slot.slot_id,
            "slot_label": slot.label,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "success": decision_value is not None,
            "symbol": symbol,
            "cycle_dir": slot_dir,
            "current_price": reference_price,
            "trigger_reason": trigger_info.get("reason"),
            "decision": decision_value,
            "analysis": dict(ai_analysis),
            "decision_mode": decision_mode,
            "ai_prompt_timeframe": ai_prompt_timeframe,
            "ai_prompt_candle_count": market_context.get("ai_prompt_candle_count"),
            "close_prices": list(close_prices),
            "position": calculate_position_metrics(position) if isinstance(position, dict) else None,
        },
    )
    return decision_value, ai_analysis, {
        "ai_prompt_timeframe": ai_prompt_timeframe,
        "ai_prompt_candle_count": market_context.get("ai_prompt_candle_count"),
        "ai_prompt_close_prices": list(close_prices),
    }


def _evaluate_slot_direction_or_none(**kwargs: Any) -> tuple[Optional[str], Dict[str, Any], Dict[str, Any], Optional[str]]:
    slot = kwargs.get("slot")
    symbol = kwargs.get("symbol")
    decision_mode = kwargs.get("decision_mode")
    try:
        decision, ai_analysis, prompt_payload = _evaluate_slot_direction(**kwargs)
        return decision, ai_analysis, prompt_payload, None
    except Exception as exc:
        logger.warning(
            "Slot AI direction evaluation failed | %s",
            format_log_details(
                {
                    "slot_id": getattr(slot, "slot_id", None),
                    "symbol": symbol,
                    "decision_mode": decision_mode,
                    "error": str(exc),
                }
            ),
            exc_info=True,
        )
        return None, {"error": str(exc)}, {}, str(exc)


def _reference_price(symbol: str) -> Optional[float]:
    payload = get_reference_price(symbol)
    return _format_price((payload or {}).get("price"))


def _managed_state_symbols(state: Dict[str, Any]) -> set[str]:
    slots_state = state.get("slots") if isinstance(state.get("slots"), dict) else {}
    symbols: set[str] = set()
    for slot_state in slots_state.values():
        if isinstance(slot_state, dict):
            symbol = _normalize_symbol(slot_state.get("symbol"))
            if symbol:
                symbols.add(symbol)
    return symbols


def _close_unmanaged_positions(
    *,
    api_key: str,
    api_secret: str,
    positions: Sequence[Dict[str, Any]],
    managed_symbols: set[str],
) -> list[Dict[str, Any]]:
    closed: list[Dict[str, Any]] = []
    for position in positions:
        symbol = _position_symbol(position)
        if not symbol or symbol in managed_symbols:
            continue
        close_result = _close_existing_position(
            api_key=api_key,
            api_secret=api_secret,
            position=position,
            context="close_unmanaged_position",
        )
        close_result["symbol"] = symbol
        closed.append(close_result)
    return closed


def _slot_result_base(slot: PortfolioSlot, symbol: Optional[str]) -> Dict[str, Any]:
    return {
        "slot_id": slot.slot_id,
        "slot_label": slot.label,
        "slot_kind": slot.kind,
        "symbol": symbol,
        "leverage": None,
        "leverage_sync": None,
        "success": False,
        "action": "init",
        "ai_triggered": False,
        "ai_decision": None,
        "screening_triggered": False,
        "screening_decision": None,
        "screening_direction": None,
        "decision_source": None,
        "trigger_reason": None,
        "trigger_price": None,
        "next_trigger_down": None,
        "next_trigger_up": None,
        "current_price": None,
        "position": None,
        "position_before": None,
        "execution": None,
        "close": None,
        "screener": None,
        "candidate_symbol": None,
        "previous_symbol": None,
        "entered_at": None,
        "last_active_rank_checked_at": None,
    }


def _run_passive_slot(
    *,
    slot: PortfolioSlot,
    slot_state: Dict[str, Any],
    config: Dict[str, Any],
    api_key: str,
    api_secret: str,
    account_overview: Dict[str, float],
    position: Optional[Dict[str, Any]],
    as_of_ms: int,
    cycle_dir_factory: CycleDirFactory,
    notification_callback: NotificationCallback,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    symbol = str(slot.symbol)
    result = _slot_result_base(slot, symbol)
    result["position_before"] = calculate_position_metrics(position) if isinstance(position, dict) else None
    leverage = _leverage_for_slot(slot, config)
    result["leverage"] = leverage
    if isinstance(position, dict):
        leverage_sync = _ensure_symbol_leverage(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            leverage=leverage,
            current_position=position,
        )
        result["leverage_sync"] = leverage_sync
        if not bool(leverage_sync.get("success")):
            result["action"] = str(leverage_sync.get("action") or "set_leverage_failed")
            result["position"] = calculate_position_metrics(position)
            return result, slot_state
        position = dict(position)
        position["leverage"] = _normalize_positive_int(leverage_sync.get("actual_leverage"), leverage)

    reference_price = _reference_price(symbol)
    if reference_price is None:
        result["action"] = "reference_price_unavailable"
        return result, slot_state
    result["current_price"] = reference_price
    trigger_info = _determine_ai_trigger(
        has_position=isinstance(position, dict),
        current_price=reference_price,
        slot_state=slot_state,
        trigger_pct_usdt=float(config["trigger_pct_usdt"]),
    )
    result.update(
        {
            "trigger_reason": trigger_info.get("reason"),
            "trigger_price": trigger_info.get("trigger_price"),
            "next_trigger_down": trigger_info.get("next_trigger_down"),
            "next_trigger_up": trigger_info.get("next_trigger_up"),
        }
    )

    target_notional = _target_notional_usdt(
        account_equity=float(account_overview["equity"]),
        slot=slot,
        capital_usage_ratio=float(config["capital_usage_ratio"]),
        leverage=leverage,
    )
    available_cap = _available_notional_cap(
        available_balance=float(account_overview.get("available_balance", 0.0)),
        capital_usage_ratio=float(config["capital_usage_ratio"]),
        leverage=leverage,
    )
    result["target_notional_usdt"] = target_notional

    if not bool(trigger_info.get("should_trigger")):
        if isinstance(position, dict):
            last_decision = _normalize_ai_decision(slot_state.get("last_ai_decision"))
            if last_decision is None:
                result["action"] = "missing_ai_decision_for_rebalance"
                result["position"] = calculate_position_metrics(position)
                return result, slot_state
            execution = _rebalance_existing_position(
                api_key=api_key,
                api_secret=api_secret,
                symbol=symbol,
                position=position,
                decision=last_decision,
                target_notional_usdt=target_notional,
                reference_price=reference_price,
                leverage=leverage,
                available_notional_cap=available_cap,
                rebalance_threshold_pct=float(config["rebalance_threshold_pct"]),
                stop_loss_pct=float(config["stop_loss_pct"]),
            )
            result["execution"] = execution
            result["position"] = execution.get("position") or calculate_position_metrics(position)
            result["stop_sync"] = execution.get("stop_sync")
            result["action"] = str(execution.get("action") or "kept_position_size")
            result["success"] = bool(execution.get("success"))
        else:
            result["action"] = "waiting_for_position"
            result["success"] = True
        return result, _update_slot_trigger_state(
            slot_state,
            ai_triggered=False,
            trigger_info=trigger_info,
            symbol=symbol,
        )

    decision, ai_analysis, prompt_payload, ai_error = _evaluate_slot_direction_or_none(
        slot=slot,
        symbol=symbol,
        reference_price=reference_price,
        config=config,
        as_of_ms=as_of_ms,
        cycle_dir_factory=cycle_dir_factory,
        notification_callback=notification_callback,
        position=position,
        trigger_info=trigger_info,
        decision_mode="passive_direction",
    )
    result["ai_triggered"] = True
    result["ai_decision"] = decision
    result["ai_analysis"] = ai_analysis
    result.update(prompt_payload)
    if ai_error:
        result["error"] = ai_error
    if decision is None:
        result["action"] = "ai_decision_failed"
        return result, slot_state

    if isinstance(position, dict):
        execution = _rebalance_existing_position(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            position=position,
            decision=decision,
            target_notional_usdt=target_notional,
            reference_price=reference_price,
            leverage=leverage,
            available_notional_cap=available_cap,
            rebalance_threshold_pct=float(config["rebalance_threshold_pct"]),
            stop_loss_pct=float(config["stop_loss_pct"]),
        )
    else:
        execution = _place_direction_position(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            decision=decision,
            target_notional_usdt=target_notional,
            reference_price=reference_price,
            leverage=leverage,
            available_notional_cap=available_cap,
        )
        if bool(execution.get("success")) and str(execution.get("action")) == "opened_new_position":
            _merge_post_trade_sync_result(
                execution,
                _sync_position_after_trade(
                    api_key=api_key,
                    api_secret=api_secret,
                    symbol=symbol,
                    stop_loss_pct=float(config["stop_loss_pct"]),
                    expected_decision=decision,
                ),
            )
    result["execution"] = execution
    result["success"] = bool(execution.get("success"))
    result["action"] = str(execution.get("action") or "execution_failed")
    result["position"] = execution.get("position")
    result["stop_sync"] = execution.get("stop_sync")
    updated_state = _update_slot_trigger_state(
        slot_state,
        ai_triggered=True,
        trigger_info=trigger_info,
        ai_decision=decision,
        symbol=symbol,
    )
    return result, updated_state


def _active_exclusions(
    *,
    config: Dict[str, Any],
    open_positions: Sequence[Dict[str, Any]],
    reserved_symbols: set[str],
    recent_universe_symbols: Sequence[str] = (),
    old_symbol: Optional[str] = None,
    allowed_symbol: Optional[str] = None,
) -> list[str]:
    allowed = _normalize_symbol(allowed_symbol)
    excluded = set(config["passive_symbols"])
    excluded.update(symbol for symbol in reserved_symbols if _normalize_symbol(symbol) != allowed)
    excluded.update(
        symbol
        for symbol in (_normalize_symbol(value) for value in recent_universe_symbols)
        if symbol and symbol != allowed
    )
    for position in open_positions:
        symbol = _position_symbol(position)
        if symbol and symbol != allowed:
            excluded.add(symbol)
    if old_symbol and _normalize_symbol(old_symbol) != allowed:
        excluded.add(old_symbol)
    return sorted(excluded)


def _run_active_slot(
    *,
    slot: PortfolioSlot,
    slot_state: Dict[str, Any],
    config: Dict[str, Any],
    api_key: str,
    api_secret: str,
    account_overview: Dict[str, float],
    open_positions: Sequence[Dict[str, Any]],
    reserved_symbols: set[str],
    as_of_ms: int,
    cycle_dir_factory: CycleDirFactory,
    notification_callback: NotificationCallback,
    recent_universe_symbols: Sequence[str] = (),
) -> tuple[Dict[str, Any], Dict[str, Any], Optional[str]]:
    state_symbol = _normalize_symbol(slot_state.get("symbol"))
    slot_universe_symbols = {
        symbol
        for symbol in (
            _normalize_symbol(value)
            for value in list(recent_universe_symbols) + _active_slot_memory_symbols(slot_state)
        )
        if symbol
    }
    positions_by_symbol = _positions_by_symbol(open_positions)
    position = positions_by_symbol.get(state_symbol) if state_symbol else None
    current_symbol = _position_symbol(position) if isinstance(position, dict) else None
    result = _slot_result_base(slot, current_symbol or state_symbol)
    result["position_before"] = calculate_position_metrics(position) if isinstance(position, dict) else None
    leverage = _leverage_for_slot(slot, config)
    result["leverage"] = leverage
    target_notional = _target_notional_usdt(
        account_equity=float(account_overview["equity"]),
        slot=slot,
        capital_usage_ratio=float(config["capital_usage_ratio"]),
        leverage=leverage,
    )
    available_cap = _available_notional_cap(
        available_balance=float(account_overview.get("available_balance", 0.0)),
        capital_usage_ratio=float(config["capital_usage_ratio"]),
        leverage=leverage,
    )
    result["target_notional_usdt"] = target_notional
    candidate: Optional[Dict[str, Any]] = None

    if isinstance(position, dict) and current_symbol:
        leverage_sync = _ensure_symbol_leverage(
            api_key=api_key,
            api_secret=api_secret,
            symbol=current_symbol,
            leverage=leverage,
            current_position=position,
        )
        result["leverage_sync"] = leverage_sync
        if not bool(leverage_sync.get("success")):
            result["action"] = str(leverage_sync.get("action") or "set_leverage_failed")
            result["position"] = calculate_position_metrics(position)
            return result, slot_state, current_symbol
        position = dict(position)
        position["leverage"] = _normalize_positive_int(leverage_sync.get("actual_leverage"), leverage)

        reference_price = _reference_price(current_symbol)
        result["symbol"] = current_symbol
        result["current_price"] = reference_price
        tracked_state, backfilled_entry_time = _ensure_active_entry_timestamp(slot_state, as_of_ms=as_of_ms)
        tracked_state["symbol"] = current_symbol
        result["entered_at"] = tracked_state.get("entered_at")
        result["last_active_rank_checked_at"] = tracked_state.get("last_active_rank_checked_at")

        if backfilled_entry_time or not _active_rank_review_due(tracked_state, config=config, as_of_ms=as_of_ms):
            stop_sync = _sync_fixed_stop_loss(
                api_key=api_key,
                api_secret=api_secret,
                symbol=current_symbol,
                position=position,
                stop_loss_pct=float(config["stop_loss_pct"]),
            )
            result["position"] = calculate_position_metrics(position)
            result["stop_sync"] = stop_sync
            result["success"] = bool(stop_sync.get("success"))
            result["action"] = "kept_active_position_until_stop_loss" if result["success"] else "stop_loss_sync_failed"
            updated_state = dict(tracked_state)
            updated_state["symbol"] = current_symbol
            return result, updated_state, current_symbol

        trigger_info = {
            "should_trigger": True,
            "reason": "active_rank_review_due",
            "trigger_price": _normalize_trigger_price(reference_price),
            "next_trigger_down": None,
            "next_trigger_up": None,
        }
        result.update(
            {
                "trigger_reason": trigger_info.get("reason"),
                "trigger_price": trigger_info.get("trigger_price"),
                "next_trigger_down": trigger_info.get("next_trigger_down"),
                "next_trigger_up": trigger_info.get("next_trigger_up"),
            }
        )
        try:
            candidate = _screen_active_candidate(
                slot=slot,
                config=config,
                excluded_symbols=_active_exclusions(
                    config=config,
                    open_positions=open_positions,
                    reserved_symbols=reserved_symbols,
                    recent_universe_symbols=slot_universe_symbols,
                    allowed_symbol=current_symbol,
                ),
            )
        except NoActiveCandidateError as exc:
            stop_sync = _sync_fixed_stop_loss(
                api_key=api_key,
                api_secret=api_secret,
                symbol=current_symbol,
                position=position,
                stop_loss_pct=float(config["stop_loss_pct"]),
            )
            result["position"] = calculate_position_metrics(position)
            result["stop_sync"] = stop_sync
            result["success"] = bool(stop_sync.get("success"))
            result["action"] = (
                "active_rank_review_failed_position_kept" if result["success"] else "stop_loss_sync_failed"
            )
            result["error"] = str(exc)
            updated_state = dict(tracked_state)
            updated_state["symbol"] = current_symbol
            return result, updated_state, current_symbol
        except Exception as exc:
            stop_sync = _sync_fixed_stop_loss(
                api_key=api_key,
                api_secret=api_secret,
                symbol=current_symbol,
                position=position,
                stop_loss_pct=float(config["stop_loss_pct"]),
            )
            result["position"] = calculate_position_metrics(position)
            result["stop_sync"] = stop_sync
            result["success"] = bool(stop_sync.get("success"))
            result["action"] = (
                "active_rank_review_failed_position_kept" if result["success"] else "stop_loss_sync_failed"
            )
            result["error"] = str(exc)
            updated_state = dict(tracked_state)
            updated_state["symbol"] = current_symbol
            return result, updated_state, current_symbol

        candidate_symbol = str(candidate["symbol"])
        screener_output = candidate.get("_screener_output")
        result["previous_symbol"] = current_symbol
        result["candidate_symbol"] = candidate_symbol
        result["screener"] = _visible_active_candidate(candidate)
        slot_dir = _slot_artifact_dir(cycle_dir_factory(), slot.slot_id)
        if isinstance(screener_output, dict):
            _persist_screener_output(slot_dir, screener_output)
        decision = _apply_active_screening_decision(result, candidate)
        if decision is None:
            stop_sync = _sync_fixed_stop_loss(
                api_key=api_key,
                api_secret=api_secret,
                symbol=current_symbol,
                position=position,
                stop_loss_pct=float(config["stop_loss_pct"]),
            )
            result["position"] = calculate_position_metrics(position)
            result["stop_sync"] = stop_sync
            result["success"] = bool(stop_sync.get("success"))
            result["action"] = (
                "active_rank_review_failed_position_kept" if result["success"] else "stop_loss_sync_failed"
            )
            updated_state = dict(tracked_state)
            updated_state["symbol"] = current_symbol
            return result, updated_state, current_symbol

        if candidate_symbol == current_symbol:
            stop_sync = _sync_fixed_stop_loss(
                api_key=api_key,
                api_secret=api_secret,
                symbol=current_symbol,
                position=position,
                stop_loss_pct=float(config["stop_loss_pct"]),
            )
            result["position"] = calculate_position_metrics(position)
            result["stop_sync"] = stop_sync
            result["success"] = bool(stop_sync.get("success"))
            result["action"] = "active_rank_review_position_kept" if result["success"] else "stop_loss_sync_failed"
            updated_state = _mark_active_rank_checked(tracked_state, as_of_ms=as_of_ms)
            updated_state["symbol"] = current_symbol
            result["last_active_rank_checked_at"] = updated_state.get("last_active_rank_checked_at")
            _emit_active_screening_after(
                notification_callback=notification_callback,
                slot=slot,
                candidate=candidate,
                result=result,
                reference_price=reference_price,
                trigger_info=trigger_info,
                config=config,
                position_before=position,
            )
            return result, updated_state, current_symbol

        candidate_reference_price = _reference_price(candidate_symbol)
        result["symbol"] = candidate_symbol
        result["current_price"] = candidate_reference_price
        if candidate_reference_price is None:
            stop_sync = _sync_fixed_stop_loss(
                api_key=api_key,
                api_secret=api_secret,
                symbol=current_symbol,
                position=position,
                stop_loss_pct=float(config["stop_loss_pct"]),
            )
            result["position"] = calculate_position_metrics(position)
            result["stop_sync"] = stop_sync
            result["success"] = bool(stop_sync.get("success"))
            result["action"] = (
                "active_rank_review_failed_position_kept" if result["success"] else "stop_loss_sync_failed"
            )
            result["error"] = "candidate_reference_price_unavailable"
            updated_state = dict(tracked_state)
            updated_state["symbol"] = current_symbol
            _emit_active_screening_after(
                notification_callback=notification_callback,
                slot=slot,
                candidate=candidate,
                result=result,
                reference_price=candidate_reference_price,
                trigger_info=trigger_info,
                config=config,
                position_before=position,
            )
            return result, updated_state, current_symbol

        close_result = _close_existing_position(
            api_key=api_key,
            api_secret=api_secret,
            position=position,
            context="active_rank_review_switch",
        )
        result["close"] = close_result
        if not bool(close_result.get("success")):
            stop_sync = _sync_fixed_stop_loss(
                api_key=api_key,
                api_secret=api_secret,
                symbol=current_symbol,
                position=position,
                stop_loss_pct=float(config["stop_loss_pct"]),
            )
            result["position"] = calculate_position_metrics(position)
            result["stop_sync"] = stop_sync
            result["success"] = False
            result["action"] = "switch_close_failed"
            updated_state = dict(tracked_state)
            updated_state["symbol"] = current_symbol
            _emit_active_screening_after(
                notification_callback=notification_callback,
                slot=slot,
                candidate=candidate,
                result=result,
                reference_price=candidate_reference_price,
                trigger_info=trigger_info,
                config=config,
                position_before=position,
            )
            return result, updated_state, current_symbol

        fresh_overview = get_account_overview(api_key, api_secret) or account_overview
        available_cap = _available_notional_cap(
            available_balance=float(fresh_overview.get("available_balance", account_overview.get("available_balance", 0.0))),
            capital_usage_ratio=float(config["capital_usage_ratio"]),
            leverage=leverage,
        )
        execution = _place_direction_position(
            api_key=api_key,
            api_secret=api_secret,
            symbol=candidate_symbol,
            decision=decision,
            target_notional_usdt=target_notional,
            reference_price=candidate_reference_price,
            leverage=leverage,
            available_notional_cap=available_cap,
        )
        entry_opened = bool(execution.get("success")) and str(execution.get("action") or "") == "opened_new_position"
        if entry_opened:
            _merge_post_trade_sync_result(
                execution,
                _sync_position_after_trade(
                    api_key=api_key,
                    api_secret=api_secret,
                    symbol=candidate_symbol,
                    stop_loss_pct=float(config["stop_loss_pct"]),
                    expected_decision=decision,
                ),
            )
        result["execution"] = execution
        result["stop_sync"] = execution.get("stop_sync")
        result["position"] = execution.get("position")
        if entry_opened and bool(execution.get("success")):
            result["success"] = True
            result["action"] = "switched_active_position_by_rank"
            updated_state = _update_slot_trigger_state(
                _mark_active_entry_opened(_clear_active_slot_state(tracked_state), as_of_ms=as_of_ms),
                ai_triggered=False,
                trigger_info=trigger_info,
                ai_decision=decision,
                symbol=candidate_symbol,
                update_anchor=True,
            )
            reserved_symbol = candidate_symbol
        else:
            execution["action"] = str(execution.get("action") or "active_rank_switch_entry_failed")
            if str(execution.get("action")) == "skipped_entry_below_min_notional":
                execution["action"] = "active_rank_switch_entry_failed"
            result["success"] = False
            result["action"] = "active_rank_switch_entry_failed"
            updated_state = _clear_active_slot_state(tracked_state)
            reserved_symbol = None
        result["last_active_rank_checked_at"] = updated_state.get("last_active_rank_checked_at")
        result["entered_at"] = updated_state.get("entered_at")
        _emit_active_screening_after(
            notification_callback=notification_callback,
            slot=slot,
            candidate=candidate,
            result=result,
            reference_price=candidate_reference_price,
            trigger_info=trigger_info,
            config=config,
            position_before=position,
        )
        return result, updated_state, reserved_symbol

    if candidate is None:
        try:
            candidate = _screen_active_candidate(
                slot=slot,
                config=config,
                excluded_symbols=_active_exclusions(
                    config=config,
                    open_positions=open_positions,
                    reserved_symbols=reserved_symbols,
                    recent_universe_symbols=slot_universe_symbols,
                    old_symbol=state_symbol,
                ),
            )
        except NoActiveCandidateError as exc:
            result["success"] = True
            result["action"] = "waiting_for_active_candidate"
            result["error"] = str(exc)
            return result, slot_state, None
        except Exception as exc:
            result["action"] = "screener_selection_failed"
            result["error"] = str(exc)
            return result, slot_state, None

    candidate_symbol = str(candidate["symbol"])
    screener_output = candidate.get("_screener_output")
    result["symbol"] = candidate_symbol
    result["candidate_symbol"] = candidate_symbol
    result["screener"] = _visible_active_candidate(candidate)
    reference_price = _reference_price(candidate_symbol)
    if reference_price is None:
        result["action"] = "candidate_reference_price_unavailable"
        return result, slot_state, candidate_symbol
    result["current_price"] = reference_price
    trigger_info = _determine_ai_trigger(
        has_position=False,
        current_price=reference_price,
        slot_state=_clear_active_slot_state(slot_state),
        trigger_pct_usdt=float(config["trigger_pct_usdt"]),
    )
    result.update(
        {
            "trigger_reason": "active_candidate_selected",
            "trigger_price": trigger_info.get("trigger_price"),
            "next_trigger_down": trigger_info.get("next_trigger_down"),
            "next_trigger_up": trigger_info.get("next_trigger_up"),
        }
    )
    trigger_info["reason"] = "active_candidate_selected"
    slot_dir = _slot_artifact_dir(cycle_dir_factory(), slot.slot_id)
    if isinstance(screener_output, dict):
        _persist_screener_output(slot_dir, screener_output)
    decision = _apply_active_screening_decision(result, candidate)
    if decision is None:
        result["action"] = "candidate_screening_direction_unavailable"
        return result, slot_state, candidate_symbol
    fresh_overview = get_account_overview(api_key, api_secret) or account_overview
    available_cap = _available_notional_cap(
        available_balance=float(fresh_overview.get("available_balance", account_overview.get("available_balance", 0.0))),
        capital_usage_ratio=float(config["capital_usage_ratio"]),
        leverage=leverage,
    )
    execution = _place_direction_position(
        api_key=api_key,
        api_secret=api_secret,
        symbol=candidate_symbol,
        decision=decision,
        target_notional_usdt=target_notional,
        reference_price=reference_price,
        leverage=leverage,
        available_notional_cap=available_cap,
    )
    if bool(execution.get("success")) and str(execution.get("action")) == "opened_new_position":
        _merge_post_trade_sync_result(
            execution,
            _sync_position_after_trade(
                api_key=api_key,
                api_secret=api_secret,
                symbol=candidate_symbol,
                stop_loss_pct=float(config["stop_loss_pct"]),
                expected_decision=decision,
            ),
        )
    result["execution"] = execution
    result["success"] = bool(execution.get("success"))
    result["action"] = str(execution.get("action") or "execution_failed")
    result["position"] = execution.get("position")
    result["stop_sync"] = execution.get("stop_sync")
    opened_active_position = result["success"] and str(result["action"]) == "opened_new_position"
    base_active_state = _clear_active_slot_state(slot_state)
    if opened_active_position:
        base_active_state = _mark_active_entry_opened(base_active_state, as_of_ms=as_of_ms)
    updated_state = _update_slot_trigger_state(
        base_active_state,
        ai_triggered=False,
        trigger_info=trigger_info,
        ai_decision=decision,
        symbol=candidate_symbol if opened_active_position else None,
        update_anchor=True,
    )
    result["entered_at"] = updated_state.get("entered_at")
    result["last_active_rank_checked_at"] = updated_state.get("last_active_rank_checked_at")
    _emit_active_screening_after(
        notification_callback=notification_callback,
        slot=slot,
        candidate=candidate,
        result=result,
        reference_price=reference_price,
        trigger_info=trigger_info,
        config=config,
        position_before=position,
    )
    return result, updated_state, candidate_symbol


def run_portfolio_cycle(
    *,
    state: Optional[Dict[str, Any]] = None,
    as_of_ms: Optional[int] = None,
    notification_callback: NotificationCallback = None,
) -> Dict[str, Any]:
    """Run one full eight-slot portfolio cycle and return a serializable result payload."""
    config = _load_strategy_config()
    slots = _build_portfolio_slots(config)
    resolved_as_of_ms = _resolve_as_of_ms(as_of_ms)
    portfolio_state = _normalize_portfolio_state(state, slots)
    cycle_dir: Optional[str] = None
    result: Dict[str, Any] = {
        "version": STATE_VERSION,
        "success": False,
        "action": "init",
        "cycle_dir": cycle_dir,
        "ai_triggered": False,
        "screening_triggered": False,
        "slot_results": [],
        "state_update": portfolio_state,
        "config_summary": {
            "fixed_leverage": config["fixed_leverage"],
            "passive_leverage": config["passive_leverage"],
            "active_leverage": config["active_leverage"],
            "capital_usage_ratio": config["capital_usage_ratio"],
            "trigger_pct_usdt": config["trigger_pct_usdt"],
            "stop_loss_pct": config["stop_loss_pct"],
            "active_rescreen_interval_hours": config["active_rescreen_interval_hours"],
            "passive_symbols": list(config["passive_symbols"]),
            "ai_prompt_timeframe": config["ai_prompt_timeframe"],
            "ai_prompt_candle_count": config["ai_prompt_candle_count"],
            "deepseek_model": config["deepseek_model"],
            "deepseek_reasoning_effort": config["deepseek_reasoning_effort"],
        },
    }

    def ensure_cycle_dir() -> str:
        nonlocal cycle_dir
        if not cycle_dir:
            cycle_dir = _create_cycle_dir()
            result["cycle_dir"] = cycle_dir
        return cycle_dir

    try:
        api_key, api_secret = get_binance_credentials()
    except ValueError as exc:
        result["action"] = "credentials_error"
        result["error"] = str(exc)
        return result

    account_overview = get_account_overview(api_key, api_secret)
    if not isinstance(account_overview, dict):
        result["action"] = "account_overview_unavailable"
        return result
    result["account_overview"] = dict(account_overview)

    positions = get_positions(api_key, api_secret)
    if positions is None:
        result["action"] = "positions_fetch_failed"
        return result

    managed_symbols = _managed_state_symbols(portfolio_state)
    unmanaged_closes = _close_unmanaged_positions(
        api_key=api_key,
        api_secret=api_secret,
        positions=positions,
        managed_symbols=managed_symbols,
    )
    if unmanaged_closes:
        result["unmanaged_position_closes"] = unmanaged_closes
        positions = get_positions(api_key, api_secret) or []

    reserved_symbols: set[str] = set()
    recent_active_symbols_by_mode = _active_recent_symbols_by_mode(slots, portfolio_state)
    for slot in slots:
        positions = get_positions(api_key, api_secret)
        if positions is None:
            slot_result = _slot_result_base(slot, slot.symbol)
            slot_result["action"] = "positions_fetch_failed"
            result["slot_results"].append(slot_result)
            continue
        account_overview = get_account_overview(api_key, api_secret) or account_overview
        slot_state = dict(portfolio_state["slots"].get(slot.slot_id) or _empty_slot_state(slot))
        positions_map = _positions_by_symbol(positions)

        if slot.kind == "passive":
            position = positions_map.get(str(slot.symbol))
            slot_result, updated_slot_state = _run_passive_slot(
                slot=slot,
                slot_state=slot_state,
                config=config,
                api_key=api_key,
                api_secret=api_secret,
                account_overview=account_overview,
                position=position,
                as_of_ms=resolved_as_of_ms,
                cycle_dir_factory=ensure_cycle_dir,
                notification_callback=notification_callback,
            )
            portfolio_state["slots"][slot.slot_id] = updated_slot_state
            if slot.symbol:
                reserved_symbols.add(slot.symbol)
        else:
            slot_result, updated_slot_state, reserved_active_symbol = _run_active_slot(
                slot=slot,
                slot_state=slot_state,
                config=config,
                api_key=api_key,
                api_secret=api_secret,
                account_overview=account_overview,
                open_positions=positions,
                reserved_symbols=reserved_symbols,
                as_of_ms=resolved_as_of_ms,
                cycle_dir_factory=ensure_cycle_dir,
                notification_callback=notification_callback,
                recent_universe_symbols=recent_active_symbols_by_mode.get(
                    _active_screening_mode(slot),
                    set(),
                ),
            )
            portfolio_state["slots"][slot.slot_id] = updated_slot_state
            recent_active_symbols_by_mode.setdefault(_active_screening_mode(slot), set()).update(
                _active_slot_memory_symbols(updated_slot_state)
            )
            if reserved_active_symbol:
                reserved_symbols.add(reserved_active_symbol)

        result["slot_results"].append(slot_result)
        if bool(slot_result.get("ai_triggered")):
            result["ai_triggered"] = True
        if bool(slot_result.get("screening_triggered")):
            result["screening_triggered"] = True

    result["success"] = all(bool(slot_result.get("success")) for slot_result in result["slot_results"])
    result["action"] = "portfolio_cycle_completed" if result["success"] else "portfolio_cycle_partial_failure"
    result["state_update"] = portfolio_state
    if result["slot_results"]:
        last_decision_result = next(
            (
                slot_result
                for slot_result in reversed(result["slot_results"])
                if bool(slot_result.get("ai_triggered")) or bool(slot_result.get("screening_triggered"))
            ),
            result["slot_results"][-1],
        )
        for key in (
            "slot_id",
            "slot_label",
            "symbol",
            "current_price",
            "ai_decision",
            "trigger_reason",
            "trigger_price",
            "next_trigger_down",
            "next_trigger_up",
            "position",
            "screening_decision",
            "screening_direction",
            "decision_source",
        ):
            result[key] = last_decision_result.get(key)
    if _should_persist_cycle_output(result):
        ensure_cycle_dir()
        _persist_cycle_output(result)
    logger.info(
        "Portfolio cycle completed | %s",
        format_log_details(
            {
                "success": result["success"],
                "action": result["action"],
                "ai_triggered": result["ai_triggered"],
                "screening_triggered": result["screening_triggered"],
                "slot_count": len(result["slot_results"]),
                "cycle_dir": result.get("cycle_dir"),
            }
        ),
    )
    return result


__all__ = [
    "PortfolioSlot",
    "STATE_VERSION",
    "_build_entry_order_plan",
    "_build_portfolio_slots",
    "_determine_ai_trigger",
    "_load_strategy_config",
    "_resolve_fixed_stop_loss_price",
    "_target_notional_usdt",
    "run_portfolio_cycle",
]
