import math
import unittest

from src.strategy.active_screener import (
    calculate_trend_metrics,
    build_usdt_perpetual_universe,
    build_usdt_tradfi_perpetual_universe,
    extract_kline_close_prices,
    select_active_symbol_from_trend_candidates,
)


def _trend_prices(*, rate: float, count: int = 672, wiggle: float = 0.0, start: float = 100.0):
    return [start * math.exp((rate * index) + (wiggle * math.sin(index / 9.0))) for index in range(count)]


def _klines_from_closes(closes):
    return [[0, "0", "0", "0", str(close)] for close in closes]


class ActiveScreenerTests(unittest.TestCase):
    def test_trend_metrics_identifies_orderly_long_trend(self):
        metrics = calculate_trend_metrics("LONGUSDT", _trend_prices(rate=0.001, wiggle=0.001))

        self.assertEqual(metrics["symbol"], "LONGUSDT")
        self.assertEqual(metrics["trend_direction"], "LONG")
        self.assertEqual(metrics["close_count"], 672)
        self.assertGreater(metrics["linearity"], 0.99)
        self.assertGreater(metrics["efficiency"], 0.95)
        self.assertEqual(metrics["weekly_consistency"], 1.0)

    def test_trend_metrics_identifies_orderly_short_trend(self):
        metrics = calculate_trend_metrics("SHORTUSDT", _trend_prices(rate=-0.001, wiggle=0.001))

        self.assertEqual(metrics["trend_direction"], "SHORT")
        self.assertLess(metrics["net_return_pct"], 0.0)
        self.assertGreater(metrics["directional_consistency"], 0.95)

    def test_selects_highest_quality_four_week_trend_after_exclusions(self):
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
