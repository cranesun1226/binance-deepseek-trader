"""ZAI active-symbol rebalancing selector for Binance ZAI Trader."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Literal, Optional, Sequence

try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test environments
    class BaseModel:
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

        @classmethod
        def model_validate_json(cls, raw_json: str):
            return cls(**json.loads(raw_json))

        def model_dump(self) -> dict[str, Any]:
            return dict(self.__dict__)

        def model_dump_json(self, indent: Optional[int] = None) -> str:
            return json.dumps(self.model_dump(), indent=indent, ensure_ascii=False)

    def Field(default: Any = None, **_kwargs: Any) -> Any:
        return default

from src.ai.zai_trader import (
    ZAI_DEFAULT_REASONING_EFFORT,
    ZAI_DEFAULT_TIMEOUT_SECONDS,
    ZAI_DIRECTION_MODEL,
    ZAI_GENERATE_MAX_RETRIES,
    DECISION_REASON_MAX_CHARS,
    ZAIEmptyContentError,
    ZAIStructuredResponse,
    ZaiClient,
    _coerce_completion_payload,
    _extract_content_text,
    _extract_message_payload,
    _is_retryable_zai_error,
    _json_object_response_format,
    _normalize_decision_reason,
    _normalize_reasoning_details,
    _normalize_reasoning_effort,
    _normalize_timeout_seconds,
    estimate_zai_cost,
)
from src.infra.env_loader import get_zai_api_key
from src.infra.logger import format_log_details, get_logger

logger = get_logger("zai_rebalancer")

REBALANCE_REASON_MAX_CHARS = DECISION_REASON_MAX_CHARS

_SYSTEM_PROMPT = (
    "You are a USDT perpetual futures trend-following momentum portfolio selector. "
    "Two symbol candidates have equal priority. "
    "Ignore order, symbol familiarity, and wording as selection signals. "
    "Select the symbol with stronger data-backed trend-following momentum consensus and higher expected value. "
    "In the reason, compare both candidates directly and explain why the selected symbol is more rational. "
    "Use only English. "
    "Return exactly one JSON object containing only selected_symbol and reason. "
    f"Keep reason data-based and {REBALANCE_REASON_MAX_CHARS} characters or fewer."
)


class ActiveRebalanceSelection(BaseModel):
    """Structured ZAI response for choosing one active symbol."""

    selected_symbol: str = Field(description="Return exactly one symbol from the supplied candidates.")
    reason: str = Field(
        description=f"English rationale in {REBALANCE_REASON_MAX_CHARS} characters or fewer for the selected symbol."
    )


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_positive_price(value: Any, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive number") from exc
    if parsed <= 0.0:
        raise ValueError(f"{field_name} must be a positive number")
    return parsed


def _normalize_close_prices(values: Any) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError("close price payload must be a list")
    close_prices: list[float] = []
    for value in values:
        close_prices.append(_normalize_positive_price(value, field_name="close_price"))
    if not close_prices:
        raise ValueError("close price payload must not be empty")
    return close_prices


def _normalize_timeframes(value: Any) -> dict[str, list[float]]:
    if not isinstance(value, dict) or not value:
        raise ValueError("timeframes are required")
    normalized: dict[str, list[float]] = {}
    for timeframe, close_values in value.items():
        normalized_timeframe = str(timeframe or "").strip()
        if not normalized_timeframe:
            raise ValueError("timeframe key is required")
        normalized[normalized_timeframe] = _normalize_close_prices(close_values)
    return normalized


def _normalize_candidate_payload(candidate: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(candidate, dict):
        raise ValueError("candidate must be a mapping")
    symbol = _normalize_symbol(candidate.get("symbol"))
    if not symbol:
        raise ValueError("candidate symbol is required")

    normalized = {
        "symbol": symbol,
        "reference_price": _normalize_positive_price(
            candidate.get("reference_price"),
            field_name=f"{symbol}.reference_price",
        ),
        "timeframes": _normalize_timeframes(candidate.get("timeframes")),
    }

    return normalized


def _build_rebalance_input_payload(
    *,
    candidates: Sequence[Dict[str, Any]],
    timeframe: str,
    candle_count: int,
) -> Dict[str, Any]:
    if len(candidates or []) != 2:
        raise ValueError("active rebalance selection requires exactly 2 candidates")
    normalized_candidates = [_normalize_candidate_payload(candidate) for candidate in candidates]
    symbols = [candidate["symbol"] for candidate in normalized_candidates]
    if len(set(symbols)) != 2:
        raise ValueError("active rebalance candidates must be distinct symbols")

    normalized_timeframe = str(timeframe or "").strip() or "1h"
    try:
        normalized_count = int(candle_count)
    except (TypeError, ValueError):
        normalized_count = 0
    if normalized_count <= 0:
        raise ValueError("candle_count must be positive")

    normalized_candidates.sort(key=lambda row: row["symbol"])
    return {
        "task": "select_one_symbol_from_equal_priority_momentum_candidates",
        "selection_basis": (
            "Choose the candidate with stronger trend-following momentum agreement "
            f"from the same {normalized_count}-close market evidence."
        ),
        "ai_prompt_timeframe": normalized_timeframe,
        "ai_prompt_candle_count": normalized_count,
        "candidates": normalized_candidates,
    }


def _format_rebalance_prompt(payload: Dict[str, Any]) -> str:
    symbols = [str(row.get("symbol") or "").strip().upper() for row in payload.get("candidates") or []]
    return (
        "Return JSON only with exactly two fields: selected_symbol and reason.\n"
        "Choose one allowed symbol; candidates are equal priority, so ignore order, familiarity, and wording.\n"
        "Select the symbol with stronger trend-following momentum evidence and expected value.\n"
        f"Reason: English, data-based, compare both candidates directly, explain why the selected symbol is more rational, {REBALANCE_REASON_MAX_CHARS} characters or fewer.\n"
        f"Allowed selected_symbol values: {json.dumps(symbols, ensure_ascii=False, separators=(',', ':'))}.\n"
        "Example: {\"selected_symbol\":\"SYMBOLUSDT\",\"reason\":\"...\"}.\n"
        f"Market payload:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def _parse_strict_rebalance_response(
    raw_response: str,
    *,
    allowed_symbols: Sequence[str],
) -> ActiveRebalanceSelection:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError("ZAI rebalance response must be a JSON object") from exc

    if not isinstance(payload, dict):
        raise ValueError("ZAI rebalance response must be a JSON object")
    if set(payload.keys()) != {"selected_symbol", "reason"}:
        raise ValueError("ZAI rebalance response must contain only selected_symbol and reason fields")

    allowed = {_normalize_symbol(symbol) for symbol in allowed_symbols if _normalize_symbol(symbol)}
    selected_symbol = _normalize_symbol(payload.get("selected_symbol"))
    if selected_symbol not in allowed:
        raise ValueError("ZAI selected_symbol must exactly match one supplied candidate symbol")
    reason = _normalize_decision_reason(payload.get("reason"))

    canonical_response = json.dumps(
        {"selected_symbol": selected_symbol, "reason": reason},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return ActiveRebalanceSelection.model_validate_json(canonical_response)


def _save_rebalance_analysis_data(
    *,
    cycle_dir: str,
    prompt: str,
    prompt_payload: Dict[str, Any],
    raw_response: str,
    decision: ActiveRebalanceSelection,
    usage_metadata: Optional[Dict[str, Any]],
    response_payload: Optional[Dict[str, Any]],
    reasoning: str,
    reasoning_details: list[dict[str, Any]],
    model: str,
    reasoning_effort: str,
    timeout_seconds: float,
    decision_mode: str = "active_rebalance",
) -> Dict[str, str]:
    saved_paths: Dict[str, str] = {}
    try:
        if cycle_dir:
            os.makedirs(cycle_dir, exist_ok=True)
            normalized_mode = str(decision_mode or "active_rebalance").strip().lower() or "active_rebalance"
            input_path = os.path.join(cycle_dir, f"zai_ai_{normalized_mode}_input.json")
            output_path = os.path.join(cycle_dir, f"zai_ai_{normalized_mode}_output.json")
            with open(input_path, "w", encoding="utf-8") as file_obj:
                json.dump(
                    {
                        "model": model,
                        "thinking": {"type": "enabled"},
                        "reasoning_effort": reasoning_effort,
                        "response_format": _json_object_response_format(),
                        "timeout_seconds": timeout_seconds,
                        "decision_mode": normalized_mode,
                        "prompt": prompt,
                        "payload": prompt_payload,
                    },
                    file_obj,
                    indent=2,
                    ensure_ascii=False,
                )
            with open(output_path, "w", encoding="utf-8") as file_obj:
                json.dump(
                    {
                        "model": model,
                        "generation_id": (response_payload or {}).get("id"),
                        "reasoning_effort": reasoning_effort,
                        "decision_mode": normalized_mode,
                        "decision": decision.model_dump(),
                        "raw_response": raw_response,
                        "reasoning": reasoning,
                        "reasoning_details": reasoning_details,
                        "usage_metadata": usage_metadata or {},
                        "estimated_cost": estimate_zai_cost(usage_metadata, model=model),
                        "response_payload": response_payload or {},
                    },
                    file_obj,
                    indent=2,
                    ensure_ascii=False,
                )
            saved_paths = {"input_path": input_path, "output_path": output_path}
    except Exception as exc:
        logger.warning("Failed to save ZAI rebalance analysis data: %s", exc)
    return saved_paths


def _call_zai_rebalance_selection(
    *,
    prompt: str,
    reasoning_effort: str,
    allowed_symbols: Sequence[str],
    model: str = ZAI_DIRECTION_MODEL,
    max_tokens: int = 8192,
    timeout_seconds: float = ZAI_DEFAULT_TIMEOUT_SECONDS,
    context_label: str = "active_rebalance",
) -> Optional[ZAIStructuredResponse[ActiveRebalanceSelection]]:
    api_key = get_zai_api_key()
    normalized_reasoning_effort = _normalize_reasoning_effort(reasoning_effort)
    normalized_timeout_seconds = _normalize_timeout_seconds(timeout_seconds)
    client = ZaiClient(api_key=api_key)

    payload = {
        "model": str(model or ZAI_DIRECTION_MODEL).strip() or ZAI_DIRECTION_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "thinking": {"type": "enabled"},
        "reasoning_effort": normalized_reasoning_effort,
        "response_format": _json_object_response_format(),
        "max_tokens": max(1, int(max_tokens)),
        "timeout": normalized_timeout_seconds,
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, ZAI_GENERATE_MAX_RETRIES + 1):
        try:
            logger.info(
                "ZAI active rebalance call starting | %s",
                format_log_details(
                    {
                        "context": context_label,
                        "attempt": attempt,
                        "max_retries": ZAI_GENERATE_MAX_RETRIES,
                        "model": payload["model"],
                        "reasoning_effort": normalized_reasoning_effort,
                        "timeout_seconds": normalized_timeout_seconds,
                        "prompt_chars": len(prompt or ""),
                        "allowed_symbols": sorted({_normalize_symbol(symbol) for symbol in allowed_symbols}),
                    }
                ),
            )
            response = client.chat.completions.create(**payload)
            response_payload = _coerce_completion_payload(response)
            message_payload = _extract_message_payload(response_payload)
            raw_response = _extract_content_text(message_payload)
            if not raw_response:
                finish_reason = None
                choices = response_payload.get("choices")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    finish_reason = choices[0].get("finish_reason")
                raise ZAIEmptyContentError(
                    f"ZAI returned empty final content"
                    f" (context={context_label}, finish_reason={finish_reason})"
                )
            decision = _parse_strict_rebalance_response(raw_response, allowed_symbols=allowed_symbols)
            usage_metadata = response_payload.get("usage") if isinstance(response_payload.get("usage"), dict) else {}
            reasoning = str(message_payload.get("reasoning_content") or message_payload.get("reasoning") or "")
            reasoning_details = _normalize_reasoning_details(message_payload.get("reasoning_details"))

            logger.info(
                "ZAI active rebalance call succeeded | %s",
                format_log_details(
                    {
                        "context": context_label,
                        "model": payload["model"],
                        "selected_symbol": getattr(decision, "selected_symbol", None),
                        "reasoning_chars": len(reasoning),
                        "reasoning_details": len(reasoning_details),
                        "usage": usage_metadata,
                    }
                ),
            )
            return ZAIStructuredResponse(
                decision=decision,
                raw_response=raw_response,
                usage_metadata=dict(usage_metadata),
                response_payload=response_payload,
                reasoning=reasoning,
                reasoning_details=reasoning_details,
            )
        except Exception as exc:
            last_error = exc
            if not _is_retryable_zai_error(exc) or attempt >= ZAI_GENERATE_MAX_RETRIES:
                break
            sleep_seconds = min(8.0, 2.0 ** (attempt - 1))
            logger.warning(
                "ZAI active rebalance call failed (attempt %s/%s): %s. Retrying in %ss.",
                attempt,
                ZAI_GENERATE_MAX_RETRIES,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    if last_error is not None:
        logger.error("ZAI active rebalance call failed: %s", last_error, exc_info=True)
    return None


def evaluate_active_rebalance_symbol(
    *,
    cycle_dir: str,
    candidates: Sequence[Dict[str, Any]],
    timeframe: str,
    candle_count: int,
    reasoning_effort: str = ZAI_DEFAULT_REASONING_EFFORT,
    model: str = ZAI_DIRECTION_MODEL,
    max_tokens: int = 8192,
    timeout_seconds: float = ZAI_DEFAULT_TIMEOUT_SECONDS,
    analysis_sink: Optional[Dict[str, Any]] = None,
    decision_mode: str = "active_rebalance",
) -> Optional[ActiveRebalanceSelection]:
    """Request one symbol selection from two active candidates."""
    try:
        prompt_payload = _build_rebalance_input_payload(
            candidates=candidates,
            timeframe=timeframe,
            candle_count=candle_count,
        )
        prompt = _format_rebalance_prompt(prompt_payload)
    except Exception as exc:
        logger.error("Invalid ZAI active rebalance prompt payload: %s", exc)
        return None

    allowed_symbols = [candidate["symbol"] for candidate in prompt_payload["candidates"]]
    call_result = _call_zai_rebalance_selection(
        prompt=prompt,
        reasoning_effort=reasoning_effort,
        allowed_symbols=allowed_symbols,
        model=model,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        context_label=decision_mode,
    )
    if call_result is None:
        return None

    decision = call_result.decision
    selected_symbol = _normalize_symbol(getattr(decision, "selected_symbol", ""))
    if selected_symbol not in set(allowed_symbols):
        logger.error("ZAI returned invalid active rebalance selected_symbol=%s", selected_symbol)
        return None
    try:
        normalized_reason = _normalize_decision_reason(getattr(decision, "reason", ""))
    except ValueError as exc:
        logger.error("ZAI returned invalid active rebalance reason: %s", exc)
        return None

    normalized_decision = ActiveRebalanceSelection(selected_symbol=selected_symbol, reason=normalized_reason)
    normalized_reasoning_effort = _normalize_reasoning_effort(reasoning_effort)
    saved_paths = _save_rebalance_analysis_data(
        cycle_dir=cycle_dir,
        prompt=prompt,
        prompt_payload=prompt_payload,
        raw_response=call_result.raw_response or normalized_decision.model_dump_json(indent=2),
        decision=normalized_decision,
        usage_metadata=call_result.usage_metadata,
        response_payload=call_result.response_payload,
        reasoning=call_result.reasoning,
        reasoning_details=call_result.reasoning_details,
        model=model,
        reasoning_effort=normalized_reasoning_effort,
        timeout_seconds=timeout_seconds,
        decision_mode=decision_mode,
    )

    if isinstance(analysis_sink, dict):
        analysis_sink.update(
            {
                "model": model,
                "generation_id": call_result.response_payload.get("id"),
                "reasoning_effort": normalized_reasoning_effort,
                "decision_mode": decision_mode,
                "decision": normalized_decision.model_dump(),
                "raw_response": call_result.raw_response,
                "reasoning": call_result.reasoning,
                "reasoning_details": list(call_result.reasoning_details),
                "usage_metadata": dict(call_result.usage_metadata or {}),
                "estimated_cost": estimate_zai_cost(call_result.usage_metadata, model=model),
                "response_payload": dict(call_result.response_payload or {}),
                **saved_paths,
            }
        )

    return normalized_decision


__all__ = [
    "ActiveRebalanceSelection",
    "REBALANCE_REASON_MAX_CHARS",
    "evaluate_active_rebalance_symbol",
]
