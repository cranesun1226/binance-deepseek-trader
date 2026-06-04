"""Active USDT-M perpetual symbol screening by 24h move target."""

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
from src.binance.common import get_binance_futures_base_url
from src.infra.logger import format_log_details, get_logger

logger = get_logger("active_screener")


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

    def ticker_24hr(self) -> list[Dict[str, Any]]:
        data = self._json("/fapi/v1/ticker/24hr")
        return data if isinstance(data, list) else []


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
        universe.add(symbol)
    return universe


def is_tradfi_symbol_info(row: Dict[str, Any]) -> bool:
    contract_type = str(row.get("contractType") or "").strip().upper()
    if contract_type == "TRADIFI_PERPETUAL":
        return True
    subtypes = row.get("underlyingSubType")
    if isinstance(subtypes, (list, tuple, set)):
        return any(str(subtype or "").strip().lower() == "tradfi" for subtype in subtypes)
    return str(subtypes or "").strip().lower() == "tradfi"


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


def normalize_ticker_row(symbol: str, ticker: Dict[str, Any], *, target_abs_change_pct: float) -> Dict[str, Any]:
    price_change_pct = safe_float(ticker.get("priceChangePercent"))
    abs_change = abs(price_change_pct)
    return {
        "symbol": symbol,
        "price_change_pct_24h": price_change_pct,
        "abs_price_change_pct_24h": abs_change,
        "target_abs_change_pct": float(target_abs_change_pct),
        "target_distance_pct": abs(abs_change - float(target_abs_change_pct)),
        "last_price": safe_float(ticker.get("lastPrice")),
        "quote_volume_24h": safe_float(ticker.get("quoteVolume")),
        "trades_24h": safe_int(ticker.get("count")),
    }


def select_active_symbol_from_tickers(
    tickers: Sequence[Dict[str, Any]],
    *,
    universe: set[str],
    target_abs_change_pct: float,
    excluded_symbols: Sequence[str],
    candidate_pool_size: int = 10,
    min_abs_change_pct: Optional[float] = None,
    max_abs_change_pct: Optional[float] = None,
) -> Dict[str, Any]:
    excluded = {str(symbol or "").strip().upper() for symbol in excluded_symbols if str(symbol or "").strip()}
    rows: list[Dict[str, Any]] = []
    for ticker in tickers:
        if not isinstance(ticker, dict):
            continue
        symbol = str(ticker.get("symbol") or "").strip().upper()
        if not symbol or symbol not in universe or symbol in excluded:
            continue
        row = normalize_ticker_row(symbol, ticker, target_abs_change_pct=target_abs_change_pct)
        if row["last_price"] <= 0.0 or row["quote_volume_24h"] <= 0.0:
            continue
        abs_change = safe_float(row.get("abs_price_change_pct_24h"))
        if min_abs_change_pct is not None and abs_change < float(min_abs_change_pct):
            continue
        if max_abs_change_pct is not None and abs_change > float(max_abs_change_pct):
            continue
        rows.append(row)

    rows.sort(
        key=lambda row: (
            safe_float(row.get("target_distance_pct"), float("inf")),
            -safe_float(row.get("quote_volume_24h")),
            str(row.get("symbol") or ""),
        )
    )
    top_candidates = rows[: max(1, int(candidate_pool_size))]
    selected = max(
        top_candidates,
        key=lambda row: (
            safe_float(row.get("quote_volume_24h")),
            -safe_float(row.get("target_distance_pct"), float("inf")),
            str(row.get("symbol") or ""),
        ),
        default=None,
    )
    return {
        "symbol": str(selected.get("symbol") or "").upper() if selected else None,
        "selected": selected,
        "top_candidates": top_candidates,
        "candidate_count": len(rows),
        "target_abs_change_pct": float(target_abs_change_pct),
        "min_abs_change_pct": min_abs_change_pct,
        "max_abs_change_pct": max_abs_change_pct,
        "excluded_symbols": sorted(excluded),
    }


def screen_active_symbol(
    *,
    target_abs_change_pct: float,
    excluded_symbols: Sequence[str],
    quote: str = "USDT",
    candidate_pool_size: int = 10,
    min_abs_change_pct: Optional[float] = None,
    max_abs_change_pct: Optional[float] = None,
    timeout: float = 30.0,
    retries: int = 3,
    request_sleep: float = 0.10,
) -> Dict[str, Any]:
    client = BinanceActiveMarketDataClient(
        timeout=timeout,
        retries=retries,
        request_sleep=request_sleep,
    )
    logger.info(
        "Active symbol screening started | %s",
        format_log_details(
            {
                "target_abs_change_pct": target_abs_change_pct,
                "min_abs_change_pct": min_abs_change_pct,
                "max_abs_change_pct": max_abs_change_pct,
                "quote": quote,
                "candidate_pool_size": candidate_pool_size,
                "excluded_symbols": sorted(
                    {str(symbol or '').strip().upper() for symbol in excluded_symbols if str(symbol or '').strip()}
                ),
            }
        ),
    )
    exchange_info = client.exchange_info()
    universe = build_usdt_perpetual_universe(exchange_info, quote=quote)
    tickers = client.ticker_24hr()
    selection = select_active_symbol_from_tickers(
        tickers,
        universe=universe,
        target_abs_change_pct=target_abs_change_pct,
        excluded_symbols=excluded_symbols,
        candidate_pool_size=candidate_pool_size,
        min_abs_change_pct=min_abs_change_pct,
        max_abs_change_pct=max_abs_change_pct,
    )
    if not selection.get("symbol"):
        if min_abs_change_pct is not None or max_abs_change_pct is not None:
            raise NoActiveCandidateError(
                f"active screener found no candidate with abs(24h change) between {min_abs_change_pct}% and {max_abs_change_pct}%"
            )
        raise NoActiveCandidateError("active screener did not return a tradable candidate")
    return {
        "metadata": {
            "captured_at": utc_now_iso(),
            "base_url": client.base_url,
            "quote": str(quote or "USDT").upper(),
            "screening_mode": "standard",
            "universe_symbols": len(universe),
            "ticker_count": len(tickers),
            "candidate_pool_size": max(1, int(candidate_pool_size)),
            "min_abs_change_pct": min_abs_change_pct,
            "max_abs_change_pct": max_abs_change_pct,
        },
        "selection": selection,
    }


def screen_active_tradfi_symbol(
    *,
    target_abs_change_pct: float,
    excluded_symbols: Sequence[str],
    min_abs_change_pct: float = 3.0,
    max_abs_change_pct: float = 5.0,
    quote: str = "USDT",
    candidate_pool_size: int = 10,
    timeout: float = 30.0,
    retries: int = 3,
    request_sleep: float = 0.10,
) -> Dict[str, Any]:
    client = BinanceActiveMarketDataClient(
        timeout=timeout,
        retries=retries,
        request_sleep=request_sleep,
    )
    logger.info(
        "Active TradFi symbol screening started | %s",
        format_log_details(
            {
                "target_abs_change_pct": target_abs_change_pct,
                "min_abs_change_pct": min_abs_change_pct,
                "max_abs_change_pct": max_abs_change_pct,
                "quote": quote,
                "candidate_pool_size": candidate_pool_size,
                "excluded_symbols": sorted(
                    {str(symbol or '').strip().upper() for symbol in excluded_symbols if str(symbol or '').strip()}
                ),
            }
        ),
    )
    exchange_info = client.exchange_info()
    universe = build_usdt_tradfi_perpetual_universe(exchange_info, quote=quote)
    tickers = client.ticker_24hr()
    selection = select_active_symbol_from_tickers(
        tickers,
        universe=universe,
        target_abs_change_pct=target_abs_change_pct,
        excluded_symbols=excluded_symbols,
        candidate_pool_size=candidate_pool_size,
        min_abs_change_pct=float(min_abs_change_pct),
        max_abs_change_pct=float(max_abs_change_pct),
    )
    if not selection.get("symbol"):
        raise NoActiveCandidateError(
            f"active TradFi screener found no candidate with abs(24h change) between {min_abs_change_pct}% and {max_abs_change_pct}%"
        )
    return {
        "metadata": {
            "captured_at": utc_now_iso(),
            "base_url": client.base_url,
            "quote": str(quote or "USDT").upper(),
            "screening_mode": "tradfi",
            "universe_symbols": len(universe),
            "ticker_count": len(tickers),
            "candidate_pool_size": max(1, int(candidate_pool_size)),
            "min_abs_change_pct": float(min_abs_change_pct),
            "max_abs_change_pct": float(max_abs_change_pct),
        },
        "selection": selection,
    }


__all__ = [
    "BinanceActiveMarketDataClient",
    "NoActiveCandidateError",
    "build_usdt_tradfi_perpetual_universe",
    "build_usdt_perpetual_universe",
    "is_tradfi_symbol_info",
    "normalize_ticker_row",
    "screen_active_symbol",
    "screen_active_tradfi_symbol",
    "select_active_symbol_from_tickers",
]
