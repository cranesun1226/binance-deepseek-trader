import math
import unittest

from src.strategy.active_screener import (
    calculate_trend_metrics,
    build_usdt_perpetual_universe,
    build_usdt_tradfi_perpetual_universe,
    evaluate_active_liquidity_filter,
    extract_kline_close_prices,
    extract_kline_quote_volumes,
    select_active_symbol_from_trend_candidates,
)
from src.strategy import active_screener


def _trend_prices(*, rate: float, count: int = 168, wiggle: float = 0.0, start: float = 100.0):
    return [start * math.exp((rate * index) + (wiggle * math.sin(index / 9.0))) for index in range(count)]


def _klines_from_closes(closes, quote_volumes=None):
    volumes = quote_volumes if quote_volumes is not None else [1_000_000.0] * len(closes)
    return [
        [0, "0", "0", "0", str(close), "0", 0, str(quote_volume)]
        for close, quote_volume in zip(closes, volumes)
    ]


def _order_book_with_depth(depth_usdt, *, mid_price=100.0):
    return {
        "bids": [["99.95", str(depth_usdt / 99.95)]],
        "asks": [["100.05", str(depth_usdt / 100.05)]],
    }


class FakeActiveClient:
    def __init__(self, *, klines_by_symbol, open_interest_by_symbol=None, order_book_by_symbol=None):
        self.klines_by_symbol = klines_by_symbol
        self.open_interest_by_symbol = open_interest_by_symbol or {}
        self.order_book_by_symbol = order_book_by_symbol or {}

    def klines(self, symbol, interval, limit):
        return self.klines_by_symbol[str(symbol).upper()]

    def open_interest(self, symbol):
        return {"openInterest": str(self.open_interest_by_symbol[str(symbol).upper()])}

    def order_book(self, symbol, limit=100):
        return self.order_book_by_symbol[str(symbol).upper()]


ACTIVE_FILTER = {
    "min_7d_avg_daily_quote_volume_usdt": 50_000_000,
    "min_7d_p10_hourly_quote_volume_usdt": 100_000,
    "min_open_interest_notional_usdt": 50_000_000,
    "min_order_book_depth_10bps_usdt": 100_000,
}


class ActiveScreenerTests(unittest.TestCase):
    def test_trend_metrics_identifies_orderly_long_trend(self):
        metrics = calculate_trend_metrics("LONGUSDT", _trend_prices(rate=0.001, wiggle=0.001))

        self.assertEqual(metrics["symbol"], "LONGUSDT")
        self.assertEqual(metrics["trend_direction"], "LONG")
        self.assertEqual(metrics["close_count"], 168)
        self.assertGreater(metrics["linearity"], 0.99)
        self.assertGreater(metrics["efficiency"], 0.95)
        self.assertEqual(metrics["weekly_consistency"], 1.0)
        self.assertEqual(metrics["weekly_consistency_segments"], 1)
        self.assertEqual(metrics["daily_consistency_segments"], 7)

    def test_trend_metrics_identifies_orderly_short_trend(self):
        metrics = calculate_trend_metrics("SHORTUSDT", _trend_prices(rate=-0.001, wiggle=0.001))

        self.assertEqual(metrics["trend_direction"], "SHORT")
        self.assertLess(metrics["net_return_pct"], 0.0)
        self.assertGreater(metrics["directional_consistency"], 0.95)

    def test_consistency_segments_scale_with_close_count(self):
        metrics = calculate_trend_metrics("LONGUSDT", _trend_prices(rate=0.001, count=168 * 4, wiggle=0.001))

        self.assertEqual(metrics["weekly_consistency_segments"], 4)
        self.assertEqual(metrics["daily_consistency_segments"], 28)

    def test_selects_highest_quality_one_week_trend_after_exclusions(self):
        candidates = [
            calculate_trend_metrics("CHOPPYUSDT", _trend_prices(rate=0.0014, wiggle=0.18)),
            calculate_trend_metrics("ORDERLYUSDT", _trend_prices(rate=0.0010, wiggle=0.001)),
            calculate_trend_metrics("EXCLUDEDUSDT", _trend_prices(rate=0.0015, wiggle=0.0)),
        ]

        selection = select_active_symbol_from_trend_candidates(
            candidates,
            excluded_symbols=["EXCLUDEDUSDT"],
        )

        self.assertEqual(selection["symbol"], "ORDERLYUSDT")
        self.assertNotIn("EXCLUDEDUSDT", [row["symbol"] for row in selection["top_candidates"]])
        self.assertGreater(
            selection["selected"]["trend_score"],
            next(row["trend_score"] for row in selection["top_candidates"] if row["symbol"] == "CHOPPYUSDT"),
        )

    def test_extract_kline_close_prices_requires_enough_positive_closes(self):
        closes = extract_kline_close_prices(_klines_from_closes([1.0, 1.1, 1.2]), required_count=2)

        self.assertEqual(closes, [1.1, 1.2])
        with self.assertRaisesRegex(ValueError, "not enough klines"):
            extract_kline_close_prices(_klines_from_closes([1.0]), required_count=2)
        with self.assertRaisesRegex(ValueError, "positive"):
            extract_kline_close_prices(_klines_from_closes([1.0, 0.0]), required_count=2)

    def test_extract_kline_quote_volumes_requires_enough_non_negative_values(self):
        volumes = extract_kline_quote_volumes(
            _klines_from_closes([1.0, 1.1, 1.2], [10.0, 20.0, 30.0]),
            required_count=2,
        )

        self.assertEqual(volumes, [20.0, 30.0])
        with self.assertRaisesRegex(ValueError, "not enough quote-volume klines"):
            extract_kline_quote_volumes(_klines_from_closes([1.0]), required_count=2)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            extract_kline_quote_volumes(_klines_from_closes([1.0], [-1.0]), required_count=1)

    def test_active_liquidity_filter_accepts_deep_liquid_candidate(self):
        klines = _klines_from_closes(
            _trend_prices(rate=0.001),
            [3_000_000.0] * 168,
        )
        client = FakeActiveClient(
            klines_by_symbol={"GOODUSDT": klines},
            open_interest_by_symbol={"GOODUSDT": 600_000.0},
            order_book_by_symbol={"GOODUSDT": _order_book_with_depth(150_000.0)},
        )

        metrics = evaluate_active_liquidity_filter(
            "GOODUSDT",
            klines,
            client=client,
            required_kline_interval="1h",
            required_kline_count=168,
            active_filter=ACTIVE_FILTER,
        )

        self.assertGreater(metrics["avg_daily_quote_volume_usdt"], 50_000_000)
        self.assertGreater(metrics["p10_hourly_quote_volume_usdt"], 100_000)
        self.assertGreater(metrics["open_interest_notional_usdt"], 50_000_000)
        self.assertGreater(metrics["order_book_depth_10bps_usdt"], 100_000)

    def test_active_liquidity_filter_rejects_low_average_quote_volume(self):
        klines = _klines_from_closes(
            _trend_prices(rate=0.001),
            [1_000.0] * 168,
        )
        client = FakeActiveClient(klines_by_symbol={"LOWUSDT": klines})

        with self.assertRaisesRegex(ValueError, "avg_daily_quote_volume_usdt"):
            evaluate_active_liquidity_filter(
                "LOWUSDT",
                klines,
                client=client,
                required_kline_interval="1h",
                required_kline_count=168,
                active_filter=ACTIVE_FILTER,
            )

    def test_active_liquidity_filter_rejects_low_p10_hourly_quote_volume(self):
        quote_volumes = [50_000.0] * 20 + [3_000_000.0] * 148
        klines = _klines_from_closes(_trend_prices(rate=0.001), quote_volumes)
        client = FakeActiveClient(klines_by_symbol={"THINUSDT": klines})

        with self.assertRaisesRegex(ValueError, "p10_hourly_quote_volume_usdt"):
            evaluate_active_liquidity_filter(
                "THINUSDT",
                klines,
                client=client,
                required_kline_interval="1h",
                required_kline_count=168,
                active_filter=ACTIVE_FILTER,
            )

    def test_active_liquidity_filter_rejects_low_open_interest(self):
        klines = _klines_from_closes(
            _trend_prices(rate=0.001),
            [3_000_000.0] * 168,
        )
        client = FakeActiveClient(
            klines_by_symbol={"LOWOIUSDT": klines},
            open_interest_by_symbol={"LOWOIUSDT": 10_000.0},
            order_book_by_symbol={"LOWOIUSDT": _order_book_with_depth(150_000.0)},
        )

        with self.assertRaisesRegex(ValueError, "open_interest_notional_usdt"):
            evaluate_active_liquidity_filter(
                "LOWOIUSDT",
                klines,
                client=client,
                required_kline_interval="1h",
                required_kline_count=168,
                active_filter=ACTIVE_FILTER,
            )

    def test_active_liquidity_filter_rejects_shallow_order_book_depth(self):
        klines = _klines_from_closes(
            _trend_prices(rate=0.001),
            [3_000_000.0] * 168,
        )
        client = FakeActiveClient(
            klines_by_symbol={"SHALLOWUSDT": klines},
            open_interest_by_symbol={"SHALLOWUSDT": 600_000.0},
            order_book_by_symbol={"SHALLOWUSDT": _order_book_with_depth(10_000.0)},
        )

        with self.assertRaisesRegex(ValueError, "order_book_depth_10bps_usdt"):
            evaluate_active_liquidity_filter(
                "SHALLOWUSDT",
                klines,
                client=client,
                required_kline_interval="1h",
                required_kline_count=168,
                active_filter=ACTIVE_FILTER,
            )

    def test_screen_active_universe_scores_only_candidates_passing_active_filter(self):
        good_klines = _klines_from_closes(
            _trend_prices(rate=0.001),
            [3_000_000.0] * 168,
        )
        low_volume_klines = _klines_from_closes(
            _trend_prices(rate=0.0015),
            [1_000.0] * 168,
        )
        client = FakeActiveClient(
            klines_by_symbol={
                "GOODUSDT": good_klines,
                "LOWVOLUSDT": low_volume_klines,
            },
            open_interest_by_symbol={"GOODUSDT": 600_000.0},
            order_book_by_symbol={"GOODUSDT": _order_book_with_depth(150_000.0)},
        )

        selection = active_screener._screen_active_universe(
            screening_mode="crypto",
            universe={"GOODUSDT", "LOWVOLUSDT"},
            client=client,
            excluded_symbols=[],
            required_kline_interval="1h",
            required_kline_count=168,
            active_filter=ACTIVE_FILTER,
        )

        self.assertEqual(selection["symbol"], "GOODUSDT")
        self.assertEqual(selection["candidate_count"], 1)
        self.assertEqual(selection["rejected_count"], 1)
        self.assertEqual(selection["rejection_samples"][0]["symbol"], "LOWVOLUSDT")
        self.assertIn("avg_daily_quote_volume_usdt", selection["rejection_samples"][0]["error"])

    def test_universe_keeps_trading_crypto_usdt_perpetuals_only(self):
        exchange_info = {
            "symbols": [
                {"symbol": "BTCUSDT", "contractType": "PERPETUAL", "status": "TRADING", "quoteAsset": "USDT"},
                {
                    "symbol": "XAUUSDT",
                    "contractType": "PERPETUAL",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "underlyingSubType": ["TradFi"],
                },
                {"symbol": "ETHUSDT", "contractType": "CURRENT_QUARTER", "status": "TRADING", "quoteAsset": "USDT"},
                {"symbol": "XRPBUSD", "contractType": "PERPETUAL", "status": "TRADING", "quoteAsset": "BUSD"},
                {"symbol": "OLDUSDT", "contractType": "PERPETUAL", "status": "BREAK", "quoteAsset": "USDT"},
            ]
        }

        self.assertEqual(build_usdt_perpetual_universe(exchange_info), {"BTCUSDT"})

    def test_tradfi_universe_uses_tradfi_contract_metadata(self):
        exchange_info = {
            "symbols": [
                {
                    "symbol": "XAUUSDT",
                    "contractType": "TRADIFI_PERPETUAL",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "underlyingSubType": ["TradFi"],
                },
                {
                    "symbol": "QQQUSDT",
                    "contractType": "TRADIFI_PERPETUAL",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "underlyingSubType": ["TradFi"],
                },
                {
                    "symbol": "BTCUSDT",
                    "contractType": "PERPETUAL",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "underlyingSubType": ["PoW"],
                },
                {
                    "symbol": "OLDTRADFIUSDT",
                    "contractType": "TRADIFI_PERPETUAL",
                    "status": "BREAK",
                    "quoteAsset": "USDT",
                    "underlyingSubType": ["TradFi"],
                },
            ]
        }

        self.assertEqual(build_usdt_tradfi_perpetual_universe(exchange_info), {"QQQUSDT", "XAUUSDT"})


if __name__ == "__main__":
    unittest.main()
