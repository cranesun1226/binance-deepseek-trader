"""Active USDT-M perpetual screening by close-range volatility."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test environments
    class _RequestsFallback:
        def get(self, *_args, **_kwargs):
            raise ModuleNotFoundError("requests is required to call Binance APIs")

    requests = _RequestsFallback()

from src.binance.binance_rate_limit import binance_api_call_with_retry
from src.binance.common import get_binance_futures_base_url, to_binance_kline_interval
from src.infra.logger import format_log_details, get_logger
from src.strategy.runtime_config import DEFAULT_AI_PROMPT_CANDLE_COUNT, DEFAULT_AI_PROMPT_TIMEFRAME

logger = get_logger("active_screener")

VOLATILITY_CANDIDATE_REPORT_LIMIT = 10
VOLATILITY_REJECTION_LOG_LIMIT = 20
VOLATILITY_RANKING_METRIC = "close_range_volatility"
VOLATILITY_RANKING_FORMULA = "abs(last_close-first_close)/((last_close+first_close)/2)"
SCREENING_DECISION_FORMULA = "LONG when last_close > first_close, SHORT when last_close < first_close"
EPSILON = 1e-12
RECENT_KLINE_RANGE_LOOKBACK = 2
RECENT_KLINE_RANGE_CHANGE_LIMIT = 0.04
MANAGED_SCREENING_DECISIONS = {"LONG", "SHORT"}


class NoActiveCandidateError(RuntimeError):
    """Raised when screening succeeds but no symbol matches the strategy filters."""


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class BinanceActiveMarketDataClient:
    def __init__(
        self,
        *,
        base_url: str = "",
        timeout: float = 30.0,
        retries: int = 3,
        request_sleep: float = 0.10,
    ) -> None:
        self.base_url = (base_url or get_binance_futures_base_url()).rstrip("/")
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self.request_sleep = max(0.0, float(request_sleep))

    def _json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        normalized_params = {k: v for k, v in (params or {}).items() if v is not None}

        def _make_api_call():
            return requests.get(url, params=normalized_params, timeout=self.timeout)

        response = binance_api_call_with_retry(
            _make_api_call,
            max_retries=self.retries,
            initial_delay=0.5,
            pre_call_delay=self.request_sleep,
            operation_name=f"active_screener{path}",
        )
        payload = response.json()
        if isinstance(payload, dict):
            code = safe_int(payload.get("code"), 0)
            if code < 0:
                raise RuntimeError(f"Binance active screener API error for {path}: {payload}")
        return payload

    def exchange_info(self) -> Dict[str, Any]:
        data = self._json("/fapi/v1/exchangeInfo")
        return data if isinstance(data, dict) else {}

    def klines(self, symbol: str, interval: Any, limit: int) -> list[Any]:
        normalized_symbol = str(symbol or "").strip().upper()
        if not normalized_symbol:
            return []
        data = self._json(
            "/fapi/v1/klines",
            {
                "symbol": normalized_symbol,
                "interval": to_binance_kline_interval(interval),
                "limit": max(1, int(limit)),
            },
        )
        return data if isinstance(data, list) else []


def is_tradfi_symbol_info(row: Dict[str, Any]) -> bool:
    contract_type = str(row.get("contractType") or "").strip().upper()
    if contract_type == "TRADIFI_PERPETUAL":
        return True
    subtypes = row.get("underlyingSubType")
    if isinstance(subtypes, (list, tuple, set)):
        return any(str(subtype or "").strip().lower() == "tradfi" for subtype in subtypes)
    return str(subtypes or "").strip().lower() == "tradfi"


def build_usdt_perpetual_universe(exchange_info: Dict[str, Any], quote: str = "USDT") -> set[str]:
    normalized_quote = str(quote or "USDT").strip().upper()
    universe: set[str] = set()
    for row in exchange_info.get("symbols", []) or []:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        if str(row.get("contractType") or "").upper() != "PERPETUAL":
            continue
        if str(row.get("status") or "").upper() != "TRADING":
            continue
        if str(row.get("quoteAsset") or "").upper() != normalized_quote:
            continue
        if is_tradfi_symbol_info(row):
            continue
        universe.add(symbol)
    return universe


def build_usdt_tradfi_perpetual_universe(exchange_info: Dict[str, Any], quote: str = "USDT") -> set[str]:
    normalized_quote = str(quote or "USDT").strip().upper()
    universe: set[str] = set()
    for row in exchange_info.get("symbols", []) or []:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        if str(row.get("status") or "").upper() != "TRADING":
            continue
        if str(row.get("quoteAsset") or "").upper() != normalized_quote:
            continue
        if not is_tradfi_symbol_info(row):
            continue
        universe.add(symbol)
    return universe


def _normalize_excluded_symbols(excluded_symbols: Sequence[str]) -> set[str]:
    return {str(symbol or "").strip().upper() for symbol in excluded_symbols if str(symbol or "").strip()}


def extract_kline_close_prices(klines: Sequence[Any], *, required_count: int) -> list[float]:
    required = max(1, safe_int(required_count, DEFAULT_AI_PROMPT_CANDLE_COUNT))
    closes: list[float] = []
    for row in klines or []:
        if isinstance(row, dict):
            raw_close = row.get("close")
        elif isinstance(row, (list, tuple)) and len(row) > 4:
            raw_close = row[4]
        else:
            raw_close = None
        close = safe_float(raw_close)
        if close <= 0.0:
            raise ValueError("kline close prices must be positive")
        closes.append(close)
    if len(closes) < required:
        raise ValueError(f"not enough klines: have={len(closes)} need={required}")
    return closes[-required:]


def _extract_kline_high_low(row: Any) -> tuple[float, float]:
    if isinstance(row, dict):
        raw_high = row.get("high")
        raw_low = row.get("low")
    elif isinstance(row, (list, tuple)) and len(row) > 3:
        raw_high = row[2]
        raw_low = row[3]
    else:
        raw_high = None
        raw_low = None
    return safe_float(raw_high), safe_float(raw_low)


def extract_recent_kline_range_changes(
    klines: Sequence[Any],
    *,
    lookback: int = RECENT_KLINE_RANGE_LOOKBACK,
) -> list[float]:
    """Return range changes for the latest returned klines.

    Binance futures klines are evaluated exactly as returned by the API. With the
    default 1h interval, this means the latest two returned klines include the
    currently forming 1h candle and the immediately preceding candle.
    """
    lookback_count = max(1, safe_int(lookback, RECENT_KLINE_RANGE_LOOKBACK))
    rows = list(klines or [])
    if len(rows) < lookback_count:
        raise ValueError(f"not enough recent klines for range filter: have={len(rows)} need={lookback_count}")

    changes: list[float] = []
    for row in rows[-lookback_count:]:
        high, low = _extract_kline_high_low(row)
        if high <= 0.0 or low <= 0.0:
            raise ValueError("recent kline high/low must be positive")
        if high < low:
            raise ValueError("recent kline high must be greater than or equal to low")
        midpoint = (high + low) / 2.0
        if midpoint <= EPSILON:
            raise ValueError("recent kline midpoint must be positive")
        changes.append((high - low) / midpoint)
    return changes


def validate_recent_kline_range_filter(
    klines: Sequence[Any],
    *,
    limit: float = RECENT_KLINE_RANGE_CHANGE_LIMIT,
    lookback: int = RECENT_KLINE_RANGE_LOOKBACK,
) -> Dict[str, Any]:
    safe_limit = safe_float(limit, RECENT_KLINE_RANGE_CHANGE_LIMIT)
    if safe_limit <= 0.0:
        safe_limit = RECENT_KLINE_RANGE_CHANGE_LIMIT

    changes = extract_recent_kline_range_changes(klines, lookback=lookback)
    max_change = max(changes) if changes else 0.0
    if max_change > safe_limit + EPSILON:
        raise ValueError(f"recent kline range change above limit: max_change={max_change:.6f} limit={safe_limit:.6f}")
    return {
        "recent_kline_range_lookback": max(1, safe_int(lookback, RECENT_KLINE_RANGE_LOOKBACK)),
        "recent_kline_range_change_limit": safe_limit,
        "recent_kline_range_changes": changes,
        "recent_kline_range_max_change": max_change,
    }


def calculate_close_range_volatility_metrics(symbol: str, close_prices: Sequence[Any]) -> Dict[str, Any]:
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol is required")
    prices = [safe_float(value) for value in close_prices or []]
    if not prices:
        raise ValueError("at least 1 close price is required")
    if any(price <= 0.0 for price in prices):
        raise ValueError("close prices must be positive")

    first_close = prices[0]
    last_close = prices[-1]
    max_close = max(prices)
    min_close = min(prices)
    midpoint = (last_close + first_close) / 2.0
    if midpoint <= EPSILON:
        raise ValueError("close range midpoint must be positive")
    close_range_volatility = abs(last_close - first_close) / midpoint
    net_return_pct = ((last_close - first_close) / first_close) * 100.0 if first_close > EPSILON else 0.0
    if last_close > first_close + EPSILON:
        screening_direction = "up"
        screening_decision = "LONG"
    elif last_close < first_close - EPSILON:
        screening_direction = "down"
        screening_decision = "SHORT"
    else:
        screening_direction = "flat"
        screening_decision = None

    return {
        "symbol": normalized_symbol,
        "close_count": len(prices),
        "close_prices": list(prices),
        "first_close": first_close,
        "last_close": last_close,
        "min_close": min_close,
        "max_close": max_close,
        "close_range_midpoint": midpoint,
        "close_range_volatility": close_range_volatility,
        "close_range_volatility_pct": close_range_volatility * 100.0,
        "net_return_pct": net_return_pct,
        "screening_direction": screening_direction,
        "screening_decision": screening_decision,
        "screening_decision_formula": SCREENING_DECISION_FORMULA,
        "ranking_metric": VOLATILITY_RANKING_METRIC,
        "ranking_formula": VOLATILITY_RANKING_FORMULA,
    }


def calculate_trend_metrics(symbol: str, close_prices: Sequence[Any]) -> Dict[str, Any]:
    """Backward-compatible wrapper for the previous trend-metric function name."""
    return calculate_close_range_volatility_metrics(symbol, close_prices)


def rank_volatility_candidates(rows: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    ranked = [dict(row) for row in rows]
    if not ranked:
        return []
    for row in ranked:
        row["ranking_metric"] = VOLATILITY_RANKING_METRIC
        row["ranking_formula"] = VOLATILITY_RANKING_FORMULA
    ranked.sort(
        key=lambda row: (
            -safe_float(row.get(VOLATILITY_RANKING_METRIC)),
            str(row.get("symbol") or ""),
        )
    )
    return ranked


def _has_screening_decision(candidate: Dict[str, Any]) -> bool:
    decision = str(candidate.get("screening_decision") or "").strip().upper()
    return decision in MANAGED_SCREENING_DECISIONS


def score_trend_candidates(rows: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Backward-compatible wrapper for the previous candidate scoring function name."""
    return rank_volatility_candidates(rows)


def select_active_symbol_from_volatility_candidates(
    candidates: Sequence[Dict[str, Any]],
    *,
    excluded_symbols: Sequence[str],
    report_limit: int = VOLATILITY_CANDIDATE_REPORT_LIMIT,
) -> Dict[str, Any]:
    excluded = _normalize_excluded_symbols(excluded_symbols)
    eligible = [
        dict(candidate)
        for candidate in candidates
        if str(candidate.get("symbol") or "").strip().upper()
        and str(candidate.get("symbol") or "").strip().upper() not in excluded
        and _has_screening_decision(candidate)
    ]
    ranked = rank_volatility_candidates(eligible)
    selected = ranked[0] if ranked else None
    return {
        "symbol": str(selected.get("symbol") or "").upper() if selected else None,
        "screening_decision": str(selected.get("screening_decision") or "").upper() if selected else None,
        "screening_direction": str(selected.get("screening_direction") or "").lower() if selected else None,
        "selected": selected,
        "top_candidates": ranked[: max(1, int(report_limit))],
        "candidate_count": len(ranked),
        "excluded_symbols": sorted(excluded),
        "ranking_metric": VOLATILITY_RANKING_METRIC,
        "ranking_formula": VOLATILITY_RANKING_FORMULA,
        "screening_decision_formula": SCREENING_DECISION_FORMULA,
    }


def select_active_symbol_from_trend_candidates(
    candidates: Sequence[Dict[str, Any]],
    *,
    excluded_symbols: Sequence[str],
    report_limit: int = VOLATILITY_CANDIDATE_REPORT_LIMIT,
) -> Dict[str, Any]:
    """Backward-compatible wrapper for the previous trend-selection function name."""
    return select_active_symbol_from_volatility_candidates(
        candidates,
        excluded_symbols=excluded_symbols,
        report_limit=report_limit,
    )


def _build_volatility_candidate(symbol: str, klines: Sequence[Any], *, required_count: int) -> Dict[str, Any]:
    range_filter = validate_recent_kline_range_filter(klines)
    close_prices = extract_kline_close_prices(klines, required_count=required_count)
    candidate = calculate_close_range_volatility_metrics(symbol, close_prices)
    candidate.update(range_filter)
    return candidate


def _build_trend_candidate(symbol: str, klines: Sequence[Any], *, required_count: int) -> Dict[str, Any]:
    """Backward-compatible wrapper for the previous private candidate builder name."""
    return _build_volatility_candidate(symbol, klines, required_count=required_count)


def _screen_active_universe(
    *,
    screening_mode: str,
    universe: set[str],
    client: BinanceActiveMarketDataClient,
    excluded_symbols: Sequence[str],
    required_kline_interval: Any,
    required_kline_count: int,
) -> Dict[str, Any]:
    normalized_interval = to_binance_kline_interval(required_kline_interval)
    normalized_count = max(1, safe_int(required_kline_count, DEFAULT_AI_PROMPT_CANDLE_COUNT))
    excluded = _normalize_excluded_symbols(excluded_symbols)
    symbols = sorted(symbol for symbol in universe if symbol not in excluded)
    candidates: list[Dict[str, Any]] = []
    rejection_samples: list[Dict[str, Any]] = []

    for symbol in symbols:
        try:
            klines = client.klines(symbol, normalized_interval, normalized_count)
            candidates.append(_build_volatility_candidate(symbol, klines, required_count=normalized_count))
        except Exception as exc:
            if len(rejection_samples) < VOLATILITY_REJECTION_LOG_LIMIT:
                rejection_samples.append({"symbol": symbol, "error": str(exc)})
            logger.info(
                "Active volatility candidate skipped | %s",
                format_log_details(
                    {
                        "symbol": symbol,
                        "screening_mode": screening_mode,
                        "required_kline_interval": normalized_interval,
                        "required_kline_count": normalized_count,
                        "error": str(exc),
                    }
                ),
            )

    selection = select_active_symbol_from_volatility_candidates(candidates, excluded_symbols=excluded)
    selection["screened_symbols"] = len(symbols)
    selection["rejected_count"] = len(symbols) - len(candidates)
    selection["rejection_samples"] = rejection_samples
    return selection


def _no_candidate_message(screening_mode: str, required_kline_interval: Any, required_kline_count: int) -> str:
    return (
        f"active {screening_mode} volatility screener found no candidate with at least "
        f"{required_kline_count} {to_binance_kline_interval(required_kline_interval)} valid close-price klines "
        f"and latest returned {RECENT_KLINE_RANGE_LOOKBACK}-kline range change <= "
        f"{RECENT_KLINE_RANGE_CHANGE_LIMIT * 100.0:.2f}% including the current forming kline "
        "and a non-flat screening direction"
    )


def screen_active_symbol(
    *,
    excluded_symbols: Sequence[str],
    quote: str = "USDT",
    timeout: float = 30.0,
    retries: int = 3,
    request_sleep: float = 0.10,
    required_kline_interval: Any = DEFAULT_AI_PROMPT_TIMEFRAME,
    required_kline_count: int = DEFAULT_AI_PROMPT_CANDLE_COUNT,
) -> Dict[str, Any]:
    client = BinanceActiveMarketDataClient(
        timeout=timeout,
        retries=retries,
        request_sleep=request_sleep,
    )
    logger.info(
        "Active crypto volatility screening started | %s",
        format_log_details(
            {
                "quote": quote,
                "required_kline_interval": required_kline_interval,
                "required_kline_count": required_kline_count,
                "excluded_symbols": sorted(_normalize_excluded_symbols(excluded_symbols)),
            }
        ),
    )
    exchange_info = client.exchange_info()
    universe = build_usdt_perpetual_universe(exchange_info, quote=quote)
    selection = _screen_active_universe(
        screening_mode="crypto",
        universe=universe,
        client=client,
        excluded_symbols=excluded_symbols,
        required_kline_interval=required_kline_interval,
        required_kline_count=required_kline_count,
    )
    if not selection.get("symbol"):
        raise NoActiveCandidateError(_no_candidate_message("crypto", required_kline_interval, required_kline_count))
    return {
        "metadata": {
            "captured_at": utc_now_iso(),
            "base_url": client.base_url,
            "quote": str(quote or "USDT").upper(),
            "screening_mode": "crypto",
            "universe_symbols": len(universe),
            "required_kline_interval": to_binance_kline_interval(required_kline_interval),
            "required_kline_count": max(1, safe_int(required_kline_count, DEFAULT_AI_PROMPT_CANDLE_COUNT)),
            "ranking_metric": VOLATILITY_RANKING_METRIC,
            "ranking_formula": VOLATILITY_RANKING_FORMULA,
            "screening_decision_formula": SCREENING_DECISION_FORMULA,
            "recent_kline_range_filter": {
                "lookback": RECENT_KLINE_RANGE_LOOKBACK,
                "change_limit": RECENT_KLINE_RANGE_CHANGE_LIMIT,
                "uses_latest_returned_klines": True,
                "includes_current_forming_kline": True,
            },
        },
        "selection": selection,
    }


def screen_active_tradfi_symbol(
    *,
    excluded_symbols: Sequence[str],
    quote: str = "USDT",
    timeout: float = 30.0,
    retries: int = 3,
    request_sleep: float = 0.10,
    required_kline_interval: Any = DEFAULT_AI_PROMPT_TIMEFRAME,
    required_kline_count: int = DEFAULT_AI_PROMPT_CANDLE_COUNT,
) -> Dict[str, Any]:
    client = BinanceActiveMarketDataClient(
        timeout=timeout,
        retries=retries,
        request_sleep=request_sleep,
    )
    logger.info(
        "Active TradFi volatility screening started | %s",
        format_log_details(
            {
                "quote": quote,
                "required_kline_interval": required_kline_interval,
                "required_kline_count": required_kline_count,
                "excluded_symbols": sorted(_normalize_excluded_symbols(excluded_symbols)),
            }
        ),
    )
    exchange_info = client.exchange_info()
    universe = build_usdt_tradfi_perpetual_universe(exchange_info, quote=quote)
    selection = _screen_active_universe(
        screening_mode="tradfi",
        universe=universe,
        client=client,
        excluded_symbols=excluded_symbols,
        required_kline_interval=required_kline_interval,
        required_kline_count=required_kline_count,
    )
    if not selection.get("symbol"):
        raise NoActiveCandidateError(_no_candidate_message("tradfi", required_kline_interval, required_kline_count))
    return {
        "metadata": {
            "captured_at": utc_now_iso(),
            "base_url": client.base_url,
            "quote": str(quote or "USDT").upper(),
            "screening_mode": "tradfi",
            "universe_symbols": len(universe),
            "required_kline_interval": to_binance_kline_interval(required_kline_interval),
            "required_kline_count": max(1, safe_int(required_kline_count, DEFAULT_AI_PROMPT_CANDLE_COUNT)),
            "ranking_metric": VOLATILITY_RANKING_METRIC,
            "ranking_formula": VOLATILITY_RANKING_FORMULA,
            "screening_decision_formula": SCREENING_DECISION_FORMULA,
            "recent_kline_range_filter": {
                "lookback": RECENT_KLINE_RANGE_LOOKBACK,
                "change_limit": RECENT_KLINE_RANGE_CHANGE_LIMIT,
                "uses_latest_returned_klines": True,
                "includes_current_forming_kline": True,
            },
        },
        "selection": selection,
    }


__all__ = [
    "BinanceActiveMarketDataClient",
    "NoActiveCandidateError",
    "RECENT_KLINE_RANGE_CHANGE_LIMIT",
    "RECENT_KLINE_RANGE_LOOKBACK",
    "SCREENING_DECISION_FORMULA",
    "VOLATILITY_RANKING_FORMULA",
    "VOLATILITY_RANKING_METRIC",
    "build_usdt_tradfi_perpetual_universe",
    "build_usdt_perpetual_universe",
    "calculate_close_range_volatility_metrics",
    "calculate_trend_metrics",
    "extract_kline_close_prices",
    "extract_recent_kline_range_changes",
    "is_tradfi_symbol_info",
    "rank_volatility_candidates",
    "score_trend_candidates",
    "screen_active_symbol",
    "screen_active_tradfi_symbol",
    "select_active_symbol_from_volatility_candidates",
    "select_active_symbol_from_trend_candidates",
    "validate_recent_kline_range_filter",
]
