"""Scheduling and notification orchestration for Binance DeepSeek Trader."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

from src.infra.logger import format_log_details, get_logger
from src.infra.price_chart import build_close_price_line_chart_png
from src.infra.telegram import (
    escape_telegram_html,
    sanitize_telegram_html,
    send_telegram_message,
    send_telegram_photo,
)
from src.strategy.portfolio_strategy import STATE_VERSION, run_portfolio_cycle
from src.strategy.runtime_config import load_runtime_config

logger = get_logger("scheduler")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STATE_FILE = os.path.join(ROOT_DIR, "scheduler_state.json")
CONFIG_PATH = os.path.join(ROOT_DIR, "setting.yaml")
SCHEDULE_SECOND_OFFSET = 10
MARKUP_TAG_PATTERN = re.compile(
    r"</?(?:a|b|blockquote|code|em|i|ins|pre|s|span|strike|strong|tg-spoiler|u)(?:\s+[^>]*)?>",
    flags=re.IGNORECASE,
)

TRIGGER_REASON_LABELS = {
    "no_position": "No open position",
    "missing_llm_anchor": "Missing LLM anchor",
    "price_distance_reached": "Price distance reached",
    "waiting_for_next_price_trigger": "Waiting for next price level",
    "active_candidate_selected": "Active candidate selected",
}

ACTION_LABELS = {
    "portfolio_cycle_completed": "Portfolio cycle completed",
    "portfolio_cycle_partial_failure": "Portfolio cycle partial failure",
    "kept_position_size": "Position kept, stop checked",
    "kept_position_by_ai": "Position kept by AI",
    "opened_new_position": "Opened new position",
    "reversed_position": "Reversed position",
    "increased_position": "Increased position",
    "reduced_position": "Reduced position",
    "closed_position": "Closed position",
    "screener_selection_failed": "Active screener failed",
    "candidate_ai_decision_failed": "Candidate AI decision failed",
    "ai_decision_failed": "AI decision failed",
    "entry_order_failed": "Entry order failed",
    "reference_price_unavailable": "Reference price unavailable",
}


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _compact_log_details(details: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in details.items() if value is not None}


def _build_cycle_completion_log_details(result: Dict[str, Any]) -> Dict[str, Any]:
    return _compact_log_details(
        {
            "success": bool(result.get("success")),
            "action": result.get("action"),
            "ai_triggered": bool(result.get("ai_triggered")),
            "slot_count": len(result.get("slot_results") or []),
            "cycle_dir": result.get("cycle_dir"),
        }
    )


class TradingScheduler:
    """Persist scheduler state and run portfolio cycles on a fixed cadence."""

    def __init__(self) -> None:
        self.is_running = False
        self._shutdown_requested = False
        self.state_file_path = os.path.abspath(STATE_FILE)
        self.state = self.load_state()
        logger.info("Scheduler state file path: %s", self.state_file_path)
        logger.info("TradingScheduler initialized in Binance DeepSeek Trader mode")

    def _default_state(self) -> Dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "slots": {},
            "last_cycle_time": None,
            "last_minute_slot": None,
            "last_cycle_result": None,
        }

    def _load_config(self) -> Dict[str, Any]:
        return load_runtime_config(CONFIG_PATH)

    def _get_cycle_interval_seconds(self) -> int:
        config = self._load_config()
        return max(1, _safe_int(config.get("cycle_interval_seconds", 60), 60))

    def load_state(self) -> Dict[str, Any]:
        default_state = self._default_state()
        try:
            if os.path.exists(self.state_file_path):
                with open(self.state_file_path, "r", encoding="utf-8") as file_obj:
                    loaded = json.load(file_obj)
                if isinstance(loaded, dict) and str(loaded.get("version") or "") == STATE_VERSION:
                    merged = dict(default_state)
                    merged.update(loaded)
                    return merged
        except json.JSONDecodeError as exc:
            logger.error("Corrupted state file: %s", exc)
        except Exception as exc:
            logger.error("Error loading scheduler state: %s", exc)
        return default_state

    def save_state(self) -> None:
        try:
            with open(self.state_file_path, "w", encoding="utf-8") as file_obj:
                json.dump(self.state, file_obj, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error("Error saving scheduler state: %s", exc)

    def _merge_state_update(self, update: Optional[Dict[str, Any]]) -> None:
        if isinstance(update, dict):
            self.state = dict(update)
            self.state["version"] = STATE_VERSION

    def _summarize_cycle_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "time": datetime.now(timezone.utc).isoformat(),
            "success": bool(result.get("success")),
            "action": result.get("action"),
            "ai_triggered": bool(result.get("ai_triggered")),
            "slot_count": len(result.get("slot_results") or []),
            "cycle_dir": result.get("cycle_dir"),
        }

    def _format_timestamp(self, timestamp_value: Optional[str]) -> str:
        try:
            parsed = datetime.fromisoformat(timestamp_value) if timestamp_value else datetime.now(timezone.utc)
        except Exception:
            parsed = datetime.now(timezone.utc)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    def _format_float(self, value: Any, digits: int = 2) -> str:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return "-"
        return f"{parsed:,.{digits}f}"

    def _format_pct(self, value: Any) -> str:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return "-"
        return f"{parsed * 100.0:.2f}%"

    def _format_usdt(self, value: Any, *, digits: int = 2) -> str:
        formatted = self._format_float(value, digits=digits)
        return f"{formatted} USDT" if formatted != "-" else "-"

    def _strip_markup(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        cleaned = MARKUP_TAG_PATTERN.sub("", text)
        cleaned = cleaned.replace("**", "").replace("__", "").replace("`", "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _clip_text(self, value: Any, *, limit: int) -> str:
        text = self._strip_markup(value)
        if not text:
            return "-"
        return text if len(text) <= limit else f"{text[:limit - 3]}..."

    def _format_html_title(self, title: str) -> str:
        return f"<b>{escape_telegram_html(title)}</b>"

    def _format_html_value(
        self,
        value: Any,
        *,
        code: bool = False,
        bold: bool = False,
        preserve_html: bool = False,
    ) -> str:
        text = str(value or "").strip()
        if not text or text in {"-", "None"}:
            return "<i>None</i>"
        safe_text = sanitize_telegram_html(text) if preserve_html else escape_telegram_html(text)
        if code:
            return f"<code>{safe_text}</code>"
        if bold:
            return f"<b>{safe_text}</b>"
        return safe_text

    def _format_html_line(
        self,
        label: str,
        value: Any,
        *,
        code: bool = False,
        bold: bool = False,
        preserve_html: bool = False,
    ) -> str:
        return (
            f"<b>{escape_telegram_html(label)}:</b> "
            f"{self._format_html_value(value, code=code, bold=bold, preserve_html=preserve_html)}"
        )

    def _humanize_code_label(self, value: Any) -> str:
        text = self._strip_markup(value)
        return text.replace("_", " ") if text else "None"

    def _translate_trigger_reason(self, value: Any) -> str:
        key = str(value or "").strip()
        return TRIGGER_REASON_LABELS.get(key, self._humanize_code_label(key))

    def _translate_action(self, value: Any) -> str:
        key = str(value or "").strip()
        return ACTION_LABELS.get(key, self._humanize_code_label(key))

    def _format_trigger_window(self, down_price: Any, up_price: Any) -> str:
        down_text = self._format_usdt(down_price, digits=2)
        up_text = self._format_usdt(up_price, digits=2)
        if down_text == "-" and up_text == "-":
            return "-"
        if down_text == "-":
            return up_text
        if up_text == "-":
            return down_text
        return f"{down_text} ~ {up_text}"

    def _format_position_summary(self, position: Any) -> str:
        if not isinstance(position, dict):
            return "None"
        symbol = str(position.get("symbol") or "").strip().upper()
        base_asset = symbol[:-4] if symbol.endswith("USDT") and len(symbol) > 4 else ""
        direction = str(position.get("direction") or "").strip().upper()
        size = self._format_float(position.get("size"), 3)
        entry_price = self._format_float(position.get("entry_price"))
        stop_loss = self._format_float(position.get("stop_loss"))
        segments = []
        if direction:
            segments.append(direction)
        if size != "-":
            segments.append(f"{size} {base_asset}".strip())
        if entry_price != "-":
            segments.append(f"Entry {entry_price}")
        if stop_loss != "-":
            segments.append(f"Stop {stop_loss}")
        return " / ".join(segments) if segments else "None"

    def _build_message(
        self,
        *,
        title: str,
        summary_lines: Sequence[str],
        sections: Sequence[tuple[str, Sequence[str], bool]],
    ) -> str:
        parts = [title, ""]
        parts.extend(summary_lines)
        for section_title, lines, compact in sections:
            visible_lines = [line for line in lines if str(line or "").strip()]
            if not visible_lines:
                continue
            parts.append("")
            parts.append(section_title)
            if compact:
                parts.extend(visible_lines)
            else:
                parts.append("\n".join(visible_lines))
        return "\n".join(parts).strip()

    def _reasoning_text(self, analysis: Dict[str, Any]) -> str:
        reasoning = str(analysis.get("reasoning") or "").strip()
        if reasoning:
            return reasoning
        details = analysis.get("reasoning_details")
        if isinstance(details, list) and details:
            try:
                return json.dumps(details, ensure_ascii=False, indent=2)
            except Exception:
                return str(details)
        return ""

    def _build_ai_cycle_before_message(self, payload: Dict[str, Any]) -> str:
        return self._build_message(
            title=self._format_html_title("Binance DeepSeek Trader | AI Cycle Start"),
            summary_lines=[
                self._format_html_line("Slot", payload.get("slot_label") or payload.get("slot_id"), code=True),
                self._format_html_line("Symbol", payload.get("symbol"), code=True),
                self._format_html_line("Price", self._format_usdt(payload.get("current_price"))),
                self._format_html_line("Trigger", self._translate_trigger_reason(payload.get("trigger_reason"))),
            ],
            sections=[
                (
                    self._format_html_title("Position"),
                    [self._format_html_line("Current", self._format_position_summary(payload.get("position")))],
                    True,
                )
            ],
        )

    def _build_ai_cycle_after_message(self, payload: Dict[str, Any]) -> str:
        analysis = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else {}
        reasoning = self._reasoning_text(analysis)
        sections: list[tuple[str, Sequence[str], bool]] = [
            (
                self._format_html_title("Position"),
                [self._format_html_line("Current", self._format_position_summary(payload.get("position")))],
                True,
            )
        ]
        if reasoning:
            sections.append(
                (
                    self._format_html_title("DeepSeek Reasoning"),
                    [sanitize_telegram_html(reasoning)],
                    False,
                )
            )
        return self._build_message(
            title=self._format_html_title("AI Decision"),
            summary_lines=[
                self._format_html_line("Slot", payload.get("slot_label") or payload.get("slot_id"), code=True),
                self._format_html_line("Symbol", payload.get("symbol"), code=True),
                self._format_html_line("Decision", payload.get("decision") or "None", bold=True),
            ],
            sections=sections,
        )

    def _build_cycle_completed_message(self, payload: Dict[str, Any]) -> str:
        slot_lines = []
        for slot_result in payload.get("slot_results") or []:
            if not isinstance(slot_result, dict):
                continue
            label = slot_result.get("slot_label") or slot_result.get("slot_id")
            symbol = slot_result.get("symbol") or "-"
            action = self._translate_action(slot_result.get("action"))
            decision = slot_result.get("ai_decision") or slot_result.get("position_exit_ai_decision") or "-"
            slot_lines.append(
                self._format_html_line(
                    str(label),
                    f"{symbol} / {action} / {decision}",
                    code=False,
                )
            )
        return self._build_message(
            title=self._format_html_title("Binance DeepSeek Trader | Cycle Update"),
            summary_lines=[
                self._format_html_line("Action", self._translate_action(payload.get("action"))),
                self._format_html_line("AI Triggered", str(bool(payload.get("ai_triggered")))),
            ],
            sections=[(self._format_html_title("Slots"), slot_lines, True)],
        )

    def _build_exception_message(self, payload: Dict[str, Any]) -> str:
        return self._build_message(
            title=self._format_html_title("Binance DeepSeek Trader | Scheduler Error"),
            summary_lines=[
                self._format_html_line("Time", self._format_timestamp(payload.get("timestamp"))),
                self._format_html_line("Error", self._clip_text(payload.get("error"), limit=600)),
            ],
            sections=[],
        )

    def _normalize_close_price_series(self, values: Any, *, limit: int = 100) -> list[float]:
        if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
            return []
        close_prices: list[float] = []
        for value in list(values)[-max(1, int(limit)) :]:
            try:
                price = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(price) and price > 0.0:
                close_prices.append(price)
        return close_prices

    def _build_close_price_chart_caption(self, payload: Dict[str, Any], close_prices: Sequence[float]) -> str:
        first_price = float(close_prices[0])
        latest_price = float(close_prices[-1])
        change = latest_price - first_price
        change_pct = (change / first_price) * 100.0 if first_price > 0.0 else 0.0
        timeframe = str(payload.get("ai_prompt_timeframe") or "1h").strip() or "1h"
        change_text = f"{change:+,.2f} USDT ({change_pct:+.2f}%)"
        return "\n".join(
            [
                "<b>AI Close Prices Line Chart</b>",
                self._format_html_line("Symbol", payload.get("symbol"), code=True),
                self._format_html_line("Timeframe", timeframe, code=True),
                self._format_html_line("Count", len(close_prices)),
                self._format_html_line("Decision", payload.get("decision") or "None", bold=True),
                self._format_html_line("First", self._format_usdt(first_price)),
                self._format_html_line("Latest", self._format_usdt(latest_price)),
                self._format_html_line("Change", change_text),
            ]
        )

    def _emit_telegram_text(self, message: str) -> bool:
        if not str(message or "").strip():
            return False
        return bool(send_telegram_message(message))

    def _emit_telegram_close_price_chart(self, payload: Dict[str, Any]) -> bool:
        close_prices = self._normalize_close_price_series(payload.get("close_prices"), limit=100)
        if not close_prices:
            return False
        symbol = str(payload.get("symbol") or "USDT").strip().upper() or "USDT"
        timeframe = str(payload.get("ai_prompt_timeframe") or "1h").strip() or "1h"
        try:
            image_bytes = build_close_price_line_chart_png(
                close_prices,
                symbol=symbol,
                timeframe=timeframe,
                limit=100,
            )
            sent = send_telegram_photo(
                image_bytes,
                filename=f"{symbol}_{timeframe}_close_prices.png",
                caption=self._build_close_price_chart_caption(payload, close_prices),
            )
        except Exception as exc:
            logger.warning("Failed to build or send Telegram close-price chart: %s", exc)
            return False
        return bool(sent)

    def _notify_telegram_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        if event_name == "ai_cycle_before":
            sent = self._emit_telegram_text(self._build_ai_cycle_before_message(payload))
            logger.info(
                "Telegram event dispatched | %s",
                format_log_details({"event": event_name, "sent": sent, "symbol": payload.get("symbol")}),
            )
            return
        if event_name == "ai_cycle_after":
            sent = self._emit_telegram_text(self._build_ai_cycle_after_message(payload))
            chart_sent = self._emit_telegram_close_price_chart(payload)
            logger.info(
                "Telegram event dispatched | %s",
                format_log_details(
                    {
                        "event": event_name,
                        "sent": sent,
                        "chart_sent": chart_sent,
                        "symbol": payload.get("symbol"),
                        "decision": payload.get("decision"),
                    }
                ),
            )

    def _maybe_send_cycle_notifications(self, cycle_time: datetime, result: Dict[str, Any]) -> None:
        if bool(result.get("ai_triggered")) or bool(result.get("unmanaged_position_closes")):
            cycle_payload = dict(result)
            cycle_payload["timestamp"] = cycle_time.isoformat()
            self._emit_telegram_text(self._build_cycle_completed_message(cycle_payload))

    def _next_cycle_boundary(
        self,
        now_utc: datetime,
        *,
        interval_seconds: int,
        offset_seconds: int = 0,
        include_current: bool = False,
    ) -> datetime:
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        interval_seconds = max(1, int(interval_seconds))
        offset_seconds = int(offset_seconds) % interval_seconds
        epoch_seconds = int(now_utc.timestamp())
        boundary_seconds = (((epoch_seconds - offset_seconds) // interval_seconds) * interval_seconds) + offset_seconds
        boundary = datetime.fromtimestamp(boundary_seconds, tz=timezone.utc)
        if include_current and now_utc == boundary:
            return boundary
        if now_utc < boundary:
            return boundary
        return datetime.fromtimestamp(boundary_seconds + interval_seconds, tz=timezone.utc)

    def run_cycle_once(self, now_utc: Optional[datetime] = None) -> Dict[str, Any]:
        cycle_time = now_utc or datetime.now(timezone.utc)
        if cycle_time.tzinfo is None:
            cycle_time = cycle_time.replace(tzinfo=timezone.utc)
        result = run_portfolio_cycle(
            state=dict(self.state),
            as_of_ms=int(cycle_time.timestamp() * 1000),
            notification_callback=self._notify_telegram_event,
        )
        self._merge_state_update(result.get("state_update"))
        self.state["last_cycle_time"] = cycle_time.isoformat()
        self.state["last_minute_slot"] = cycle_time.replace(second=0, microsecond=0).isoformat()
        self.state["last_cycle_result"] = self._summarize_cycle_result(result)
        self.save_state()
        self._maybe_send_cycle_notifications(cycle_time, result)
        return result

    def minute_mechanical_check(self, now_utc: datetime) -> None:
        try:
            result = self.run_cycle_once(now_utc)
            logger.info(
                "Cycle completed | %s",
                format_log_details(_build_cycle_completion_log_details(result)),
            )
        except Exception as exc:
            logger.error("Error in minute_mechanical_check: %s", exc, exc_info=True)
            self._emit_telegram_text(
                self._build_exception_message({"timestamp": now_utc.isoformat(), "error": str(exc)})
            )

    def _should_run_immediate_cycle(
        self,
        now_utc: datetime,
        *,
        interval_seconds: int,
        offset_seconds: int,
    ) -> bool:
        last_slot = self.state.get("last_minute_slot")
        if not last_slot:
            return True
        try:
            last_time = datetime.fromisoformat(str(last_slot))
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=timezone.utc)
        except Exception:
            return True
        due_time = self._next_cycle_boundary(
            last_time,
            interval_seconds=interval_seconds,
            offset_seconds=offset_seconds,
            include_current=False,
        )
        return now_utc >= due_time

    def _signal_handler(self, signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        logger.info("Received %s, initiating graceful shutdown...", signal_name)
        self._shutdown_requested = True

    def run_forever(self) -> None:
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.is_running = True
        interval_seconds = self._get_cycle_interval_seconds()
        logger.info("=" * 60)
        logger.info("=== Binance DeepSeek Trader Scheduler Started ===")
        logger.info("=" * 60)
        logger.info("Cycle interval: %s seconds", interval_seconds)
        logger.info("Scheduled second offset: %s seconds", SCHEDULE_SECOND_OFFSET)
        try:
            startup_now = datetime.now(timezone.utc)
            if self._should_run_immediate_cycle(
                startup_now,
                interval_seconds=interval_seconds,
                offset_seconds=SCHEDULE_SECOND_OFFSET,
            ):
                self.minute_mechanical_check(startup_now)
            next_cycle = self._next_cycle_boundary(
                datetime.now(timezone.utc),
                interval_seconds=interval_seconds,
                offset_seconds=SCHEDULE_SECOND_OFFSET,
                include_current=False,
            )
            while self.is_running and not self._shutdown_requested:
                now_utc = datetime.now(timezone.utc)
                if now_utc >= next_cycle:
                    self.minute_mechanical_check(now_utc)
                    next_cycle = self._next_cycle_boundary(
                        datetime.now(timezone.utc),
                        interval_seconds=interval_seconds,
                        offset_seconds=SCHEDULE_SECOND_OFFSET,
                        include_current=False,
                    )
                sleep_seconds = (next_cycle - datetime.now(timezone.utc)).total_seconds()
                time.sleep(max(0.25, sleep_seconds))
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as exc:
            logger.error("Scheduler loop error: %s", exc, exc_info=True)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.is_running = False
        self._shutdown_requested = True
        self.save_state()
        logger.info("Trading scheduler stopped")
