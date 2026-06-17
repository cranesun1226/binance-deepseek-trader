import math
import unittest

from src.strategy.active_screener import (
    ENTRY_EXTREME_DISTANCE_LIMIT,
    ENTRY_EXTREME_LOOKBACK,
    TREND_SCORE_WEIGHTS,
    _build_trend_candidate,
    build_usdt_perpetual_universe,
    build_usdt_tradfi_perpetual_universe,
    calculate_trend_metrics,
    extract_kline_close_prices,
    extract_recent_kline_extremes,
    select_active_symbol_from_trend_candidates,
    validate_directional_entry_filter,
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
    def test_trend_score_weights_balance_return_quality_and_win_rate_inputs(self):
        self.assertAlmostEqual(sum(TREND_SCORE_WEIGHTS.values()), 1.0)
        self.assertEqual(
            TREND_SCORE_WEIGHTS,
            {
                "trend_strength": 0.20,
                "r_squared": 0.16,
                "efficiency": 0.16,
                "directional_consistency": 0.17,
                "daily_consistency": 0.12,
                "adverse_score": 0.14,
                "trend_magnitude": 0.05,
            },
        )
        self.assertNotIn("weekly_consistency", TREND_SCORE_WEIGHTS)

    def test_trend_metrics_identifies_orderly_long_trend(self):
        metrics = calculate_trend_metrics("LONGUSDT", _trend_prices(rate=0.001, wiggle=0.001))

        self.assertEqual(metrics["symbol"], "LONGUSDT")
        self.assertEqual(metrics["trend_direction"], "LONG")
        self.assertEqual(metrics["screening_decision"], "LONG")
        self.assertEqual(metrics["screening_direction"], "up")
        self.assertEqual(metrics["close_count"], 168)
        self.assertGreater(metrics["r_squared"], 0.99)
        self.assertGreater(metrics["efficiency"], 0.95)
        self.assertEqual(metrics["weekly_consistency"], 1.0)
        self.assertEqual(metrics["weekly_consistency_segments"], 1)
        self.assertEqual(metrics["daily_consistency_segments"], 7)
        self.assertEqual(metrics["ranking_metric"], "trend_score")

    def test_trend_metrics_identifies_orderly_short_trend(self):
        metrics = calculate_trend_metrics("SHORTUSDT", _trend_prices(rate=-0.001, wiggle=0.001))

        self.assertEqual(metrics["trend_direction"], "SHORT")
        self.assertEqual(metrics["screening_decision"], "SHORT")
        self.assertEqual(metrics["screening_direction"], "down")
        self.assertLess(metrics["net_return_pct"], 0.0)
        self.assertGreater(metrics["directional_consistency"], 0.95)

    def test_trend_metrics_rejects_flat_paths(self):
        with self.assertRaisesRegex(ValueError, "flat"):
            calculate_trend_metrics("FLATUSDT", [100.0] * 168)

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
        self.assertEqual(selection["screening_decision"], "LONG")
        self.assertEqual(selection["screening_direction"], "up")
        self.assertEqual(selection["ranking_metric"], "trend_score")
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

    def test_directional_entry_filter_allows_long_when_24h_min_low_is_inside_four_percent(self):
        klines = [_kline(index, close=100.0, high=101.0, low=96.0) for index in range(ENTRY_EXTREME_LOOKBACK)]

        result = validate_directional_entry_filter(klines, decision="LONG", reference_price=100.0)

        self.assertTrue(result["entry_filter_passed"])
        self.assertEqual(result["recent_kline_min_low"], 96.0)
        self.assertEqual(result["entry_filter_long_min_low_threshold"], 96.0)
        self.assertEqual(result["entry_filter_distance_limit"], ENTRY_EXTREME_DISTANCE_LIMIT)

    def test_directional_entry_filter_rejects_long_when_24h_min_low_breaks_four_percent(self):
        klines = [_kline(index, close=100.0, high=101.0, low=96.0) for index in range(ENTRY_EXTREME_LOOKBACK)]
        klines[-1][3] = "95.99"

        with self.assertRaisesRegex(ValueError, "long entry filter failed"):
            validate_directional_entry_filter(klines, decision="LONG", reference_price=100.0)

    def test_directional_entry_filter_allows_short_when_24h_max_high_is_inside_four_percent(self):
        klines = [_kline(index, close=100.0, high=104.0, low=99.0) for index in range(ENTRY_EXTREME_LOOKBACK)]

        result = validate_directional_entry_filter(klines, decision="SHORT", reference_price=100.0)

        self.assertTrue(result["entry_filter_passed"])
        self.assertEqual(result["recent_kline_max_high"], 104.0)
        self.assertEqual(result["entry_filter_short_max_high_threshold"], 104.0)

    def test_directional_entry_filter_rejects_short_when_24h_max_high_breaks_four_percent(self):
        klines = [_kline(index, close=100.0, high=104.0, low=99.0) for index in range(ENTRY_EXTREME_LOOKBACK)]
        klines[-1][2] = "104.01"

        with self.assertRaisesRegex(ValueError, "short entry filter failed"):
            validate_directional_entry_filter(klines, decision="SHORT", reference_price=100.0)

    def test_build_trend_candidate_keeps_entry_filter_metadata_when_filter_passes(self):
        closes = _trend_prices(rate=0.001)
        klines = _klines_from_closes(closes)
        for row in klines[-ENTRY_EXTREME_LOOKBACK:]:
            row[2] = str(float(row[4]) * 1.001)
            row[3] = str(float(row[4]) * 0.999)

        candidate = _build_trend_candidate("ORDERLYUSDT", klines, required_count=168, reference_price=closes[-1])

        self.assertEqual(candidate["symbol"], "ORDERLYUSDT")
        self.assertEqual(candidate["screening_decision"], "LONG")
        self.assertEqual(candidate["ranking_metric"], "trend_score")
        self.assertTrue(candidate["entry_filter_passed"])

    def test_extract_recent_kline_extremes_uses_latest_24_klines(self):
        klines = [_kline(index, close=100.0, high=100.0 + index, low=100.0 - index) for index in range(30)]

        extremes = extract_recent_kline_extremes(klines)

        self.assertEqual(extremes["recent_kline_extreme_lookback"], 24.0)
        self.assertEqual(extremes["recent_kline_min_low"], 71.0)
        self.assertEqual(extremes["recent_kline_max_high"], 129.0)

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
