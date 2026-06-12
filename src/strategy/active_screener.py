"""Active USDT-M perpetual screening by one-week trend quality."""

from __future__ import annotations

import math
import time
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

TREND_CANDIDATE_REPORT_LIMIT = 10
TREND_REJECTION_LOG_LIMIT = 20
TREND_STRENGTH_CAP = 100.0
HOURLY_CANDLES_PER_DAY = 24
HOURLY_CANDLES_PER_WEEK = HOURLY_CANDLES_PER_DAY * 7
EPSILON = 1e-12
RECENT_KLINE_RANGE_LOOKBACK = 2
RECENT_KLINE_RANGE_CHANGE_LIMIT = 0.04

TREND_SCORE_WEIGHTS: dict[str, float] = {
    "trend_strength": 0.22,
    "linearity": 0.18,
    "efficiency": 0.18,
    "directional_consistency": 0.15,
    "weekly_consistency": 0.12,
    "daily_consistency": 0.07,
    "adverse_score": 0.05,
    "trend_magnitude": 0.03,
}


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


def _segment_direction_share(log_prices: Sequence[float], *, segments: int, direction_multiplier: float) -> float:
    segment_count = max(1, int(segments))
    if len(log_prices) < 2:
        return 0.0
    step = max(1, len(log_prices) // segment_count)
    aligned = 0
    valid = 0
    for index in range(segment_count):
        start = index * step
        if start >= len(log_prices) - 1:
            break
        end = min((index + 1) * step, len(log_prices) - 1)
        if index == segment_count - 1:
            end = len(log_prices) - 1
        if end <= start:
            continue
        valid += 1
        if direction_multiplier * (log_prices[end] - log_prices[start]) > 0.0:
            aligned += 1
    return aligned / valid if valid else 0.0


def _scaled_consistency_segments(close_count: int, *, candles_per_period: int) -> int:
    count = max(1, safe_int(close_count, 1))
    period = max(1, safe_int(candles_per_period, 1))
    if count < 2:
        return 1
    return max(1, min(count - 1, int(round(count / period))))


def calculate_trend_metrics(symbol: str, close_prices: Sequence[Any]) -> Dict[str, Any]:
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol is required")
    prices = [safe_float(value) for value in close_prices or []]
    if len(prices) < 3:
        raise ValueError("at least 3 close prices are required")
    if any(price <= 0.0 for price in prices):
        raise ValueError("close prices must be positive")

    log_prices = [math.log(price) for price in prices]
    count = len(log_prices)
    time_mean = (count - 1) / 2.0
    price_mean = sum(log_prices) / count
    centered_time = [index - time_mean for index in range(count)]
    sxx = sum(value * value for value in centered_time)
    if sxx <= EPSILON:
        raise ValueError("not enough time variance to fit trend")

    slope = sum(centered_time[index] * (log_prices[index] - price_mean) for index in range(count)) / sxx
    if abs(slope) <= EPSILON:
        raise ValueError("trend slope is flat")
    direction_multiplier = 1.0 if slope > 0.0 else -1.0
    trend_direction = "LONG" if slope > 0.0 else "SHORT"
    intercept = price_mean - slope * time_mean

    residuals = [log_prices[index] - (intercept + slope * index) for index in range(count)]
    sse = sum(value * value for value in residuals)
    sst = sum((value - price_mean) ** 2 for value in log_prices)
    linearity = 0.0 if sst <= EPSILON else max(0.0, min(1.0, 1.0 - (sse / sst)))
    if count > 2 and sse > EPSILON:
        slope_stderr = math.sqrt((sse / (count - 2)) / sxx)
        trend_strength = min(TREND_STRENGTH_CAP, abs(slope) / slope_stderr) if slope_stderr > EPSILON else TREND_STRENGTH_CAP
    else:
        trend_strength = TREND_STRENGTH_CAP

    returns = [log_prices[index] - log_prices[index - 1] for index in range(1, count)]
    total_path = sum(abs(value) for value in returns)
    if total_path <= EPSILON:
        raise ValueError("price path is flat")
    net_log_return = log_prices[-1] - log_prices[0]
    directional_net_return = direction_multiplier * net_log_return
    if directional_net_return <= 0.0:
        raise ValueError("endpoint return is not aligned with trend slope")

    same_direction_path = sum(max(direction_multiplier * value, 0.0) for value in returns)
    directional_consistency = same_direction_path / total_path
    efficiency = abs(net_log_return) / total_path

    transformed_path = [direction_multiplier * (value - log_prices[0]) for value in log_prices]
    running_peak = transformed_path[0]
    max_adverse_excursion = 0.0
    for value in transformed_path:
        running_peak = max(running_peak, value)
        max_adverse_excursion = max(max_adverse_excursion, running_peak - value)
    adverse_ratio = max_adverse_excursion / max(directional_net_return, EPSILON)
    adverse_score = 1.0 / (1.0 + adverse_ratio)

    weekly_segments = _scaled_consistency_segments(count, candles_per_period=HOURLY_CANDLES_PER_WEEK)
    daily_segments = _scaled_consistency_segments(count, candles_per_period=HOURLY_CANDLES_PER_DAY)
    weekly_consistency = _segment_direction_share(
        log_prices,
        segments=weekly_segments,
        direction_multiplier=direction_multiplier,
    )
    daily_consistency = _segment_direction_share(
        log_prices,
        segments=daily_segments,
        direction_multiplier=direction_multiplier,
    )
    trend_log_return = slope * (count - 1)
    realized_volatility = 0.0
    if len(returns) > 1:
        return_mean = sum(returns) / len(returns)
        realized_volatility = math.sqrt(
            sum((value - return_mean) ** 2 for value in returns) / (len(returns) - 1)
        ) * math.sqrt(len(returns))

    return {
        "symbol": normalized_symbol,
        "trend_direction": trend_direction,
        "close_count": count,
        "first_close": prices[0],
        "last_close": prices[-1],
        "net_return_pct": math.expm1(net_log_return) * 100.0,
        "trend_return_pct": math.expm1(trend_log_return) * 100.0,
        "trend_magnitude": abs(trend_log_return),
        "trend_slope": slope,
        "trend_strength": trend_strength,
        "linearity": linearity,
        "efficiency": efficiency,
        "directional_consistency": directional_consistency,
        "weekly_consistency": weekly_consistency,
        "daily_consistency": daily_consistency,
        "weekly_consistency_segments": weekly_segments,
        "daily_consistency_segments": daily_segments,
        "max_adverse_excursion": max_adverse_excursion,
        "adverse_ratio": adverse_ratio,
        "adverse_score": adverse_score,
        "realized_volatility": realized_volatility,
    }


def _percentile_ranks(rows: Sequence[Dict[str, Any]], metric_name: str) -> list[float]:
    if not rows:
        return []
    if len(rows) == 1:
        return [1.0]
    indexed_values = sorted(
        (safe_float(row.get(metric_name)), index)
        for index, row in enumerate(rows)
    )
    ranks = [0.0] * len(rows)
    position = 0
    denominator = len(rows) - 1
    while position < len(indexed_values):
        end = position
        value = indexed_values[position][0]
        while end + 1 < len(indexed_values) and indexed_values[end + 1][0] == value:
            end += 1
        rank = ((position + end) / 2.0) / denominator
        for grouped_index in range(position, end + 1):
            ranks[indexed_values[grouped_index][1]] = rank
        position = end + 1
    return ranks


def score_trend_candidates(rows: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    scored = [dict(row) for row in rows]
    if not scored:
        return []
    metric_ranks = {metric_name: _percentile_ranks(scored, metric_name) for metric_name in TREND_SCORE_WEIGHTS}
    for index, row in enumerate(scored):
        components = {
            metric_name: metric_ranks[metric_name][index]
            for metric_name in TREND_SCORE_WEIGHTS
        }
        score = sum(TREND_SCORE_WEIGHTS[metric_name] * components[metric_name] for metric_name in TREND_SCORE_WEIGHTS)
        row["trend_score"] = score
        row["score_components"] = components
        row["score_weights"] = dict(TREND_SCORE_WEIGHTS)
    scored.sort(
        key=lambda row: (
            -safe_float(row.get("trend_score")),
            -safe_float(row.get("trend_strength")),
            -safe_float(row.get("linearity")),
            -safe_float(row.get("efficiency")),
            -safe_float(row.get("trend_magnitude")),
            str(row.get("symbol") or ""),
        )
    )
    return scored


def select_active_symbol_from_trend_candidates(
    candidates: Sequence[Dict[str, Any]],
    *,
    excluded_symbols: Sequence[str],
    report_limit: int = TREND_CANDIDATE_REPORT_LIMIT,
) -> Dict[str, Any]:
    excluded = _normalize_excluded_symbols(excluded_symbols)
    eligible = [
        dict(candidate)
        for candidate in candidates
        if str(candidate.get("symbol") or "").strip().upper() and str(candidate.get("symbol") or "").strip().upper() not in excluded
    ]
    scored = score_trend_candidates(eligible)
    selected = scored[0] if scored else None
    return {
        "symbol": str(selected.get("symbol") or "").upper() if selected else None,
        "selected": selected,
        "top_candidates": scored[: max(1, int(report_limit))],
        "candidate_count": len(scored),
        "excluded_symbols": sorted(excluded),
    }


def _build_trend_candidate(symbol: str, klines: Sequence[Any], *, required_count: int) -> Dict[str, Any]:
    range_filter = validate_recent_kline_range_filter(klines)
    close_prices = extract_kline_close_prices(klines, required_count=required_count)
    candidate = calculate_trend_metrics(symbol, close_prices)
    candidate.update(range_filter)
    return candidate


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
            candidates.append(_build_trend_candidate(symbol, klines, required_count=normalized_count))
        except Exception as exc:
            if len(rejection_samples) < TREND_REJECTION_LOG_LIMIT:
                rejection_samples.append({"symbol": symbol, "error": str(exc)})
            logger.info(
                "Active trend candidate skipped | %s",
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

    selection = select_active_symbol_from_trend_candidates(candidates, excluded_symbols=excluded)
    selection["screened_symbols"] = len(symbols)
    selection["rejected_count"] = len(symbols) - len(candidates)
    selection["rejection_samples"] = rejection_samples
    return selection


def _no_candidate_message(screening_mode: str, required_kline_interval: Any, required_kline_count: int) -> str:
    return (
        f"active {screening_mode} trend screener found no candidate with at least "
        f"{required_kline_count} {to_binance_kline_interval(required_kline_interval)} valid close-price klines "
        f"and recent {RECENT_KLINE_RANGE_LOOKBACK}-kline range change <= "
        f"{RECENT_KLINE_RANGE_CHANGE_LIMIT * 100.0:.2f}%"
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
        "Active crypto trend screening started | %s",
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
            "score_weights": dict(TREND_SCORE_WEIGHTS),
            "recent_kline_range_filter": {
                "lookback": RECENT_KLINE_RANGE_LOOKBACK,
                "change_limit": RECENT_KLINE_RANGE_CHANGE_LIMIT,
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
        "Active TradFi trend screening started | %s",
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
            "score_weights": dict(TREND_SCORE_WEIGHTS),
            "recent_kline_range_filter": {
                "lookback": RECENT_KLINE_RANGE_LOOKBACK,
                "change_limit": RECENT_KLINE_RANGE_CHANGE_LIMIT,
            },
        },
        "selection": selection,
    }


__all__ = [
    "BinanceActiveMarketDataClient",
    "NoActiveCandidateError",
    "TREND_SCORE_WEIGHTS",
    "RECENT_KLINE_RANGE_CHANGE_LIMIT",
    "RECENT_KLINE_RANGE_LOOKBACK",
    "build_usdt_tradfi_perpetual_universe",
    "build_usdt_perpetual_universe",
    "calculate_trend_metrics",
    "extract_kline_close_prices",
    "extract_recent_kline_range_changes",
    "is_tradfi_symbol_info",
    "score_trend_candidates",
    "screen_active_symbol",
    "screen_active_tradfi_symbol",
    "select_active_symbol_from_trend_candidates",
    "validate_recent_kline_range_filter",
]
