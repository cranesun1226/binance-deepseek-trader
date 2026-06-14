import math
import unittest

from src.strategy.active_screener import (
    RECENT_KLINE_RANGE_CHANGE_LIMIT,
    _build_volatility_candidate,
    calculate_close_range_volatility_metrics,
    build_usdt_perpetual_universe,
    build_usdt_tradfi_perpetual_universe,
    extract_kline_close_prices,
    extract_recent_kline_range_changes,
    select_active_symbol_from_volatility_candidates,
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
    def test_close_range_volatility_metrics_uses_first_last_midpoint(self):
        metrics = calculate_close_range_volatility_metrics("RANGEUSDT", [100.0, 110.0, 90.0, 105.0])

        self.assertEqual(metrics["symbol"], "RANGEUSDT")
        self.assertEqual(metrics["close_count"], 4)
        self.assertEqual(metrics["first_close"], 100.0)
        self.assertEqual(metrics["last_close"], 105.0)
        self.assertEqual(metrics["min_close"], 90.0)
        self.assertEqual(metrics["max_close"], 110.0)
        self.assertEqual(metrics["close_range_midpoint"], 102.5)
        self.assertAlmostEqual(metrics["close_range_volatility"], 5.0 / 102.5)
        self.assertAlmostEqual(metrics["close_range_volatility_pct"], (5.0 / 102.5) * 100.0)
        self.assertEqual(metrics["screening_direction"], "up")
        self.assertEqual(metrics["screening_decision"], "LONG")
        self.assertEqual(metrics["ranking_metric"], "close_range_volatility")

    def test_close_range_volatility_metrics_allows_flat_paths(self):
        metrics = calculate_close_range_volatility_metrics("FLATUSDT", [100.0] * 168)

        self.assertEqual(metrics["close_count"], 168)
        self.assertEqual(metrics["close_range_volatility"], 0.0)
        self.assertEqual(metrics["screening_direction"], "flat")
        self.assertIsNone(metrics["screening_decision"])

    def test_selects_highest_close_range_volatility_after_exclusions(self):
        candidates = [
            calculate_close_range_volatility_metrics("LOWVOLUSDT", [100.0] * 167 + [102.0]),
            calculate_close_range_volatility_metrics("HIGHVOLUSDT", [100.0] * 167 + [130.0]),
            calculate_close_range_volatility_metrics("FLATVOLUSDT", [100.0, 150.0, 50.0, 100.0]),
            calculate_close_range_volatility_metrics("EXCLUDEDUSDT", [100.0] * 167 + [200.0]),
        ]

        selection = select_active_symbol_from_volatility_candidates(
            candidates,
            excluded_symbols=["EXCLUDEDUSDT"],
        )

        self.assertEqual(selection["symbol"], "HIGHVOLUSDT")
        self.assertEqual(selection["screening_decision"], "LONG")
        self.assertEqual(selection["screening_direction"], "up")
        self.assertEqual(selection["ranking_metric"], "close_range_volatility")
        self.assertNotIn("EXCLUDEDUSDT", [row["symbol"] for row in selection["top_candidates"]])
        self.assertNotIn("FLATVOLUSDT", [row["symbol"] for row in selection["top_candidates"]])
        self.assertGreater(
            selection["selected"]["close_range_volatility"],
            next(row["close_range_volatility"] for row in selection["top_candidates"] if row["symbol"] == "LOWVOLUSDT"),
        )
        self.assertNotIn("trend_score", selection["selected"])

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

    def test_build_volatility_candidate_rejects_current_forming_or_previous_kline_above_four_percent(self):
        for index in (-2, -1):
            with self.subTest(index=index):
                klines = _klines_from_closes(_trend_prices(rate=0.001))
                klines[index][2] = "105"
                klines[index][3] = "100"

                with self.assertRaisesRegex(ValueError, "range change above limit"):
                    _build_volatility_candidate("VOLATILEUSDT", klines, required_count=168)

    def test_build_volatility_candidate_keeps_recent_range_metadata_when_filter_passes(self):
        candidate = _build_volatility_candidate(
            "ORDERLYUSDT",
            _klines_from_closes(_trend_prices(rate=0.001)),
            required_count=168,
        )

        self.assertEqual(candidate["symbol"], "ORDERLYUSDT")
        self.assertIn("close_range_volatility", candidate)
        self.assertEqual(candidate["recent_kline_range_changes"], [0.0, 0.0])
        self.assertEqual(candidate["recent_kline_range_change_limit"], RECENT_KLINE_RANGE_CHANGE_LIMIT)

    def test_universe_keeps_trading_crypto_usdt_perpetuals_only(self):
        exchange_info = {
            "symbols": [
                {"symbol": "BTCUSDT", "contractType": "PERPETUAL", "status": "TRADING", "quoteAsset": "USDT"},
                {
                    "symbol": "COPPERUSDT",
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
                    "symbol": "COPPERUSDT",
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

        self.assertEqual(build_usdt_tradfi_perpetual_universe(exchange_info), {"QQQUSDT", "COPPERUSDT"})


if __name__ == "__main__":
    unittest.main()
