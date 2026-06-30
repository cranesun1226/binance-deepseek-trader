"""ZAI direction-decision helper for Binance ZAI Trader."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Literal, Optional, TypeVar

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

try:
    from zai import ZaiClient
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test environments
    class ZaiClient:
        def __init__(self, *_args, **_kwargs):
            raise ModuleNotFoundError("zai-sdk is required to call ZAI APIs")

from src.infra.env_loader import get_zai_api_key
from src.infra.logger import format_log_details, get_logger

logger = get_logger("zai_trader")

ZAI_GENERATE_MAX_RETRIES = 3
ZAI_GLM_5_2_MODEL = "glm-5.2"
ZAI_DIRECTION_MODEL = ZAI_GLM_5_2_MODEL
ZAI_HIGH_REASONING_EFFORT = "high"
ZAI_MAX_REASONING_EFFORT = "max"
ZAI_DEFAULT_REASONING_EFFORT = ZAI_MAX_REASONING_EFFORT
ZAI_DEFAULT_TIMEOUT_SECONDS = 300.0
DECISION_REASON_MAX_CHARS = 300
_ONE_MILLION = 1_000_000
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")

_ZAI_MODEL_PRICING_USD_PER_MILLION: dict[str, dict[str, float]] = {}

_SYSTEM_PROMPT = (
    "You are a world-class USDT perpetual futures crypto trader. "
    "Use your own trader judgment and the supplied market data to choose the LONG or SHORT direction that appears more likely to produce future profit. "
    "Use only English to reason and respond. "
    "Return exactly one json object containing only the decision and reason. "
    f"The reason must be english, reasonable, data-based, and {DECISION_REASON_MAX_CHARS} characters or fewer."
)

DecisionT = TypeVar("DecisionT", bound=BaseModel)


class TradeDirectionDecision(BaseModel):
    """Structured ZAI response for one pure symbol direction decision."""

    decision: Literal["LONG", "SHORT"] = Field(
        description="Return exactly one direction decision, LONG or SHORT."
    )
    reason: str = Field(
        description=(
            f"English rationale in {DECISION_REASON_MAX_CHARS} characters or fewer to decide the LONG or SHORT position."
        )
    )


@dataclass
class ZAIStructuredResponse(Generic[DecisionT]):
    """Container for a parsed ZAI decision plus raw diagnostics."""

    decision: DecisionT
    raw_response: str
    usage_metadata: dict[str, Any]
    response_payload: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    reasoning_details: list[dict[str, Any]] = field(default_factory=list)


class ZAIEmptyContentError(ValueError):
    """Raised when ZAI returns a response without final JSON content."""


class ZAIAuthenticationError(RuntimeError):
    """Raised when ZAI rejects credentials and further immediate retries are unsafe."""


def is_zai_authentication_error(value: Any) -> bool:
    status_code = getattr(value, "status_code", None)
    if status_code is None:
        response = getattr(value, "response", None)
        status_code = getattr(response, "status_code", None)
    try:
        if int(status_code) in {401, 403}:
            return True
    except (TypeError, ValueError):
        pass

    text = str(value or "").lower()
    return any(
        pattern in text
        for pattern in (
            "error code: 401",
            "error code: 403",
            "authentication failed",
            "invalid api key",
            "invalid public key",
            '"code":"1000"',
            '"code":"1004"',
            "unauthorized",
        )
    )


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _to_jsonable(method())
            except Exception:
                pass
    return str(value)


def _safe_non_negative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


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


def _build_direction_input_payload(
    *,
    symbol: str,
    reference_price: float,
    timeframe_ohlcv: Dict[str, Any],
) -> Dict[str, Any]:
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol is required")
    if not isinstance(timeframe_ohlcv, dict) or not timeframe_ohlcv:
        raise ValueError("timeframe_ohlcv is required")

    normalized_timeframes: dict[str, list[float]] = {}
    for timeframe, close_values in timeframe_ohlcv.items():
        normalized_timeframe = str(timeframe or "").strip()
        if not normalized_timeframe:
            raise ValueError("timeframe key is required")
        normalized_timeframes[normalized_timeframe] = _normalize_close_prices(close_values)

    return {
        "symbol": normalized_symbol,
        "reference_price": _normalize_positive_price(reference_price, field_name="reference_price"),
        "timeframes": normalized_timeframes,
    }


def _format_direction_prompt(payload: Dict[str, Any]) -> str:
    symbol = str(payload.get("symbol") or "the supplied symbol").strip().upper() or "the supplied symbol"
    return (
        f"You are a world-class {symbol} trader.\n"
        "Return JSON only with exactly two fields: decision and reason.\n"
        "Use your own trader judgment and the supplied market data to choose the LONG or SHORT position that appears more likely to produce future profit.\n"
        f"The reason must be english, reasonable, data-based, and {DECISION_REASON_MAX_CHARS} characters or fewer.\n"
        "Examples: {\"decision\":\"LONG\",\"reason\":\"...\"} or {\"decision\":\"SHORT\",\"reason\":\"...\"}.\n"
        f"Market payload:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def _build_direction_prompt(
    *,
    symbol: str,
    reference_price: float,
    timeframe_ohlcv: Dict[str, Any],
) -> str:
    payload = _build_direction_input_payload(
        symbol=symbol,
        reference_price=reference_price,
        timeframe_ohlcv=timeframe_ohlcv,
    )
    return _format_direction_prompt(payload)


def _json_object_response_format() -> dict[str, str]:
    return {"type": "json_object"}


def _decision_reason_char_count(value: str) -> int:
    return len(value)


def _normalize_decision_reason(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("ZAI decision reason must be a string")
    reason = " ".join(value.strip().split())
    if not reason:
        raise ValueError("ZAI decision reason must not be empty")
    if _CJK_PATTERN.search(reason):
        raise ValueError("ZAI decision reason must use English only")
    if _decision_reason_char_count(reason) > DECISION_REASON_MAX_CHARS:
        raise ValueError(f"ZAI decision reason must be {DECISION_REASON_MAX_CHARS} characters or fewer")
    return reason


def _parse_strict_decision_response(raw_response: str, response_model: type[DecisionT]) -> DecisionT:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError("ZAI decision response must be a JSON object") from exc

    if not isinstance(payload, dict):
        raise ValueError("ZAI decision response must be a JSON object")
    if set(payload.keys()) != {"decision", "reason"}:
        raise ValueError("ZAI decision response must contain only the decision and reason fields")

    decision_value = payload.get("decision")
    if decision_value not in {"LONG", "SHORT"}:
        raise ValueError("ZAI decision must be exactly LONG or SHORT")
    reason_value = _normalize_decision_reason(payload.get("reason"))

    canonical_response = json.dumps(
        {"decision": decision_value, "reason": reason_value},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return response_model.model_validate_json(canonical_response)


def estimate_zai_cost(
    usage_metadata: Optional[dict[str, Any]],
    *,
    model: str = ZAI_DIRECTION_MODEL,
) -> Optional[dict[str, Any]]:
    usage = usage_metadata if isinstance(usage_metadata, dict) else {}
    pricing = _ZAI_MODEL_PRICING_USD_PER_MILLION.get(str(model or "").strip())
    if not usage or not pricing:
        return None

    prompt_tokens = _safe_non_negative_int(usage.get("prompt_tokens"))
    completion_tokens = _safe_non_negative_int(usage.get("completion_tokens"))

    prompt_details = usage.get("prompt_tokens_details")
    cached_tokens = 0
    if isinstance(prompt_details, dict):
        cached_tokens = _safe_non_negative_int(prompt_details.get("cached_tokens"))
    cache_hit_tokens = _safe_non_negative_int(usage.get("prompt_cache_hit_tokens")) or cached_tokens
    cache_miss_tokens = _safe_non_negative_int(usage.get("prompt_cache_miss_tokens"))
    if cache_hit_tokens + cache_miss_tokens <= 0:
        cache_miss_tokens = prompt_tokens
    elif prompt_tokens > cache_hit_tokens + cache_miss_tokens:
        cache_miss_tokens += prompt_tokens - cache_hit_tokens - cache_miss_tokens

    cache_hit_input_cost_usd = cache_hit_tokens * pricing["input_cache_hit"] / _ONE_MILLION
    cache_miss_input_cost_usd = cache_miss_tokens * pricing["input_cache_miss"] / _ONE_MILLION
    input_cost_usd = cache_hit_input_cost_usd + cache_miss_input_cost_usd
    output_cost_usd = completion_tokens * pricing["output"] / _ONE_MILLION
    return {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "prompt_cache_hit_tokens": cache_hit_tokens,
        "prompt_cache_miss_tokens": cache_miss_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": _safe_non_negative_int(usage.get("total_tokens")),
        "cache_hit_input_cost_usd": round(cache_hit_input_cost_usd, 12),
        "cache_miss_input_cost_usd": round(cache_miss_input_cost_usd, 12),
        "input_cost_usd": round(input_cost_usd, 12),
        "output_cost_usd": round(output_cost_usd, 12),
        "total_cost_usd": round(input_cost_usd + output_cost_usd, 12),
    }


def _is_retryable_zai_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    try:
        code = int(status_code)
    except (TypeError, ValueError):
        return True
    return code == 429 or 500 <= code < 600


def _extract_message_payload(response_payload: Dict[str, Any]) -> Dict[str, Any]:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first_choice = choices[0] if isinstance(choices[0], dict) else {}
    message = first_choice.get("message")
    return message if isinstance(message, dict) else {}


def _extract_content_text(message_payload: Dict[str, Any]) -> str:
    content = message_payload.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    pieces.append(text)
            elif isinstance(item, str):
                pieces.append(item)
        return "".join(pieces).strip()
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return ""


def _normalize_reasoning_details(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in (_to_jsonable(row) for row in value) if isinstance(item, dict)]


def _coerce_completion_payload(response: Any) -> Dict[str, Any]:
    payload = _to_jsonable(response)
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"ZAI returned unexpected payload: {payload!r}")


def _normalize_timeout_seconds(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return ZAI_DEFAULT_TIMEOUT_SECONDS
    if parsed <= 0.0:
        return ZAI_DEFAULT_TIMEOUT_SECONDS
    return parsed


def _normalize_reasoning_effort(value: Any) -> str:
    normalized = str(value or ZAI_DEFAULT_REASONING_EFFORT).strip().lower()
    if normalized in {"max", "xhigh"}:
        return ZAI_MAX_REASONING_EFFORT
    if normalized in {"low", "medium", "high"}:
        return ZAI_HIGH_REASONING_EFFORT
    if normalized in {"", "none", "minimal"}:
        return ZAI_DEFAULT_REASONING_EFFORT
    logger.warning(
        "Unsupported zai_reasoning_effort=%s; using %s",
        value,
        ZAI_DEFAULT_REASONING_EFFORT,
    )
    return ZAI_DEFAULT_REASONING_EFFORT


def _save_direction_analysis_data(
    *,
    cycle_dir: str,
    prompt: str,
    prompt_payload: Dict[str, Any],
    raw_response: str,
    decision: Any,
    usage_metadata: Optional[Dict[str, Any]],
    response_payload: Optional[Dict[str, Any]],
    reasoning: str,
    reasoning_details: list[dict[str, Any]],
    model: str,
    reasoning_effort: str,
    timeout_seconds: float,
    decision_mode: str = "direction",
) -> Dict[str, str]:
    saved_paths: Dict[str, str] = {}
    try:
        if cycle_dir:
            os.makedirs(cycle_dir, exist_ok=True)
            normalized_mode = str(decision_mode or "direction").strip().lower() or "direction"
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
        logger.warning("Failed to save ZAI AI analysis data: %s", exc)
    return saved_paths


def _call_zai_structured_decision(
    *,
    prompt: str,
    reasoning_effort: str,
    response_model: type[DecisionT],
    model: str = ZAI_DIRECTION_MODEL,
    max_tokens: int = 8192,
    timeout_seconds: float = ZAI_DEFAULT_TIMEOUT_SECONDS,
    context_label: str = "direction",
) -> Optional[ZAIStructuredResponse[DecisionT]]:
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
                "ZAI futures decision call starting | %s",
                format_log_details(
                    {
                        "context": context_label,
                        "attempt": attempt,
                        "max_retries": ZAI_GENERATE_MAX_RETRIES,
                        "model": payload["model"],
                        "reasoning_effort": normalized_reasoning_effort,
                        "timeout_seconds": normalized_timeout_seconds,
                        "prompt_chars": len(prompt or ""),
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
            decision = _parse_strict_decision_response(raw_response, response_model)
            usage_metadata = response_payload.get("usage") if isinstance(response_payload.get("usage"), dict) else {}
            reasoning = str(message_payload.get("reasoning_content") or message_payload.get("reasoning") or "")
            reasoning_details = _normalize_reasoning_details(message_payload.get("reasoning_details"))

            logger.info(
                "ZAI futures decision call succeeded | %s",
                format_log_details(
                    {
                        "context": context_label,
                        "model": payload["model"],
                        "decision": getattr(decision, "decision", None),
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
            if is_zai_authentication_error(exc):
                auth_error = ZAIAuthenticationError(f"ZAI authentication failed: {exc}")
                logger.error("ZAI futures %s authentication failed: %s", context_label, exc, exc_info=True)
                raise auth_error from exc
            if not _is_retryable_zai_error(exc) or attempt >= ZAI_GENERATE_MAX_RETRIES:
                break
            sleep_seconds = min(8.0, 2.0 ** (attempt - 1))
            if isinstance(exc, ZAIEmptyContentError):
                logger.info(
                    "ZAI futures %s call returned empty content (attempt %s/%s). Retrying in %ss.",
                    context_label,
                    attempt,
                    ZAI_GENERATE_MAX_RETRIES,
                    sleep_seconds,
                )
            else:
                logger.warning(
                    "ZAI futures %s call failed (attempt %s/%s): %s. Retrying in %ss.",
                    context_label,
                    attempt,
                    ZAI_GENERATE_MAX_RETRIES,
                    exc,
                    sleep_seconds,
                )
            time.sleep(sleep_seconds)

    if last_error is not None:
        logger.error("ZAI futures %s call failed: %s", context_label, last_error, exc_info=True)
    return None


def evaluate_trade_direction(
    *,
    cycle_dir: str,
    symbol: str,
    reference_price: float,
    timeframe_ohlcv: Dict[str, Any],
    reasoning_effort: str,
    model: str = ZAI_DIRECTION_MODEL,
    max_tokens: int = 8192,
    timeout_seconds: float = ZAI_DEFAULT_TIMEOUT_SECONDS,
    analysis_sink: Optional[Dict[str, Any]] = None,
    decision_mode: str = "direction",
) -> Optional[TradeDirectionDecision]:
    """Request one pure symbol LONG/SHORT direction decision and persist artifacts."""
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        logger.error("evaluate_trade_direction requires a symbol")
        return None

    try:
        prompt_payload = _build_direction_input_payload(
            symbol=normalized_symbol,
            reference_price=reference_price,
            timeframe_ohlcv=timeframe_ohlcv,
        )
        prompt = _format_direction_prompt(prompt_payload)
    except Exception as exc:
        logger.error("Invalid ZAI direction prompt payload: %s", exc)
        return None

    call_result = _call_zai_structured_decision(
        prompt=prompt,
        reasoning_effort=reasoning_effort,
        response_model=TradeDirectionDecision,
        model=model,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        context_label=decision_mode,
    )
    if call_result is None:
        return None

    decision = call_result.decision
    normalized_value = str(getattr(decision, "decision", "") or "").strip().upper()
    if normalized_value not in {"LONG", "SHORT"}:
        logger.error("ZAI returned invalid direction decision=%s", normalized_value)
        return None
    try:
        normalized_reason = _normalize_decision_reason(getattr(decision, "reason", ""))
    except ValueError as exc:
        logger.error("ZAI returned invalid direction reason: %s", exc)
        return None

    normalized_decision = TradeDirectionDecision(decision=normalized_value, reason=normalized_reason)
    saved_paths = _save_direction_analysis_data(
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
        reasoning_effort=_normalize_reasoning_effort(reasoning_effort),
        timeout_seconds=timeout_seconds,
        decision_mode=decision_mode,
    )

    if isinstance(analysis_sink, dict):
        analysis_sink.update(
            {
                "model": model,
                "generation_id": call_result.response_payload.get("id"),
                "reasoning_effort": _normalize_reasoning_effort(reasoning_effort),
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


def evaluate_entry_direction(**kwargs: Any) -> Optional[TradeDirectionDecision]:
    """Compatibility alias for a single entry-direction decision."""
    return evaluate_trade_direction(**kwargs)


__all__ = [
    "ZAI_DIRECTION_MODEL",
    "ZAI_DEFAULT_REASONING_EFFORT",
    "ZAI_DEFAULT_TIMEOUT_SECONDS",
    "ZAI_MAX_REASONING_EFFORT",
    "ZAIAuthenticationError",
    "ZAIStructuredResponse",
    "TradeDirectionDecision",
    "estimate_zai_cost",
    "evaluate_entry_direction",
    "evaluate_trade_direction",
    "is_zai_authentication_error",
]
