import math
import unittest

from src.strategy.active_screener import (
    RECENT_KLINE_RANGE_CHANGE_LIMIT,
    _build_trend_candidate,
    calculate_trend_metrics,
    build_usdt_perpetual_universe,
    build_usdt_tradfi_perpetual_universe,
    extract_kline_close_prices,
    extract_recent_kline_range_changes,
    select_active_symbol_from_trend_candidates,
    validate_recent_kline_range_filter,
)


def _trend_prices(*, rate: float, count: int = 168, wiggle: float = 0.0, start: float = 100.0):
    return [start * math.exp((rate * index) + (wiggle * math.sin(index / 9.0))) for index in range(count)]


def _klines_from_closes(closes):
    return [
        [index, str(close), str(close), str(close), str(close)]
        for index, close in enumerate(closes)
    ]


def _kline(index, *, close, high=None, low=None):
    high_value = close if high is None else high
    low_value = close if low is None else low
    return [index, str(close), str(high_value), str(low_value), str(close)]


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

    def test_recent_kline_range_filter_uses_latest_two_klines(self):
        klines = [
            _kline(0, close=100.0, high=140.0, low=100.0),
            _kline(1, close=101.0, high=102.0, low=100.0),
            {"timestamp": 2, "close": "103", "high": "104", "low": "102"},
        ]

        changes = extract_recent_kline_range_changes(klines)
        result = validate_recent_kline_range_filter(klines)

        self.assertEqual(len(changes), 2)
        self.assertAlmostEqual(changes[0], 2.0 / 101.0)
        self.assertAlmostEqual(changes[1], 2.0 / 103.0)
        self.assertLess(result["recent_kline_range_max_change"], RECENT_KLINE_RANGE_CHANGE_LIMIT)

    def test_build_trend_candidate_rejects_current_or_previous_kline_above_four_percent(self):
        for index in (-2, -1):
            with self.subTest(index=index):
                klines = _klines_from_closes(_trend_prices(rate=0.001))
                klines[index][2] = "105"
                klines[index][3] = "100"

                with self.assertRaisesRegex(ValueError, "range change above limit"):
                    _build_trend_candidate("VOLATILEUSDT", klines, required_count=168)

    def test_build_trend_candidate_keeps_recent_range_metadata_when_filter_passes(self):
        candidate = _build_trend_candidate(
            "ORDERLYUSDT",
            _klines_from_closes(_trend_prices(rate=0.001)),
            required_count=168,
        )

        self.assertEqual(candidate["symbol"], "ORDERLYUSDT")
        self.assertEqual(candidate["recent_kline_range_changes"], [0.0, 0.0])
        self.assertEqual(candidate["recent_kline_range_change_limit"], RECENT_KLINE_RANGE_CHANGE_LIMIT)

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
