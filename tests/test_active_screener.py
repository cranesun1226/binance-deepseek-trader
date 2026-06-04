import unittest

from src.strategy.active_screener import (
    build_usdt_perpetual_universe,
    build_usdt_tradfi_perpetual_universe,
    select_active_symbol_from_tickers,
)


class ActiveScreenerTests(unittest.TestCase):
    def test_selects_highest_volume_inside_closest_ten_and_excludes_symbols(self):
        tickers = [
            {
                "symbol": f"SYM{idx}USDT",
                "priceChangePercent": str(4.0 + (idx * 0.01)),
                "lastPrice": "10",
                "quoteVolume": str(1000 + idx),
                "count": "100",
            }
            for idx in range(12)
        ]
        tickers[2]["quoteVolume"] = "999999"
        tickers[9]["quoteVolume"] = "888888"
        tickers[10]["quoteVolume"] = "99999999"
        universe = {row["symbol"] for row in tickers}

        selection = select_active_symbol_from_tickers(
            tickers,
            universe=universe,
            target_abs_change_pct=4.0,
            excluded_symbols=["SYM2USDT"],
            candidate_pool_size=9,
        )

        self.assertEqual(selection["symbol"], "SYM9USDT")
        self.assertNotIn("SYM2USDT", [row["symbol"] for row in selection["top_candidates"]])
        self.assertNotIn("SYM10USDT", [row["symbol"] for row in selection["top_candidates"]])

    def test_abs_change_target_is_bidirectional(self):
        tickers = [
            {"symbol": "LONGUSDT", "priceChangePercent": "4.1", "lastPrice": "1", "quoteVolume": "10"},
            {"symbol": "SHORTUSDT", "priceChangePercent": "-4.0", "lastPrice": "1", "quoteVolume": "20"},
        ]

        selection = select_active_symbol_from_tickers(
            tickers,
            universe={"LONGUSDT", "SHORTUSDT"},
            target_abs_change_pct=4.0,
            excluded_symbols=[],
            candidate_pool_size=10,
        )

        self.assertEqual(selection["symbol"], "SHORTUSDT")

    def test_universe_keeps_trading_usdt_perpetuals_only(self):
        exchange_info = {
            "symbols": [
                {"symbol": "BTCUSDT", "contractType": "PERPETUAL", "status": "TRADING", "quoteAsset": "USDT"},
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

    def test_tradfi_band_limits_pool_to_eligible_count_when_less_than_ten(self):
        tickers = [
            {
                "symbol": f"TRADFI{idx}USDT",
                "priceChangePercent": str(3.1 + (idx * 0.2)),
                "lastPrice": "10",
                "quoteVolume": str(1000 + idx),
            }
            for idx in range(7)
        ]
        tickers[5]["quoteVolume"] = "999999"
        tickers.append(
            {
                "symbol": "OUTSIDEUSDT",
                "priceChangePercent": "5.8",
                "lastPrice": "10",
                "quoteVolume": "999999999",
            }
        )
        universe = {row["symbol"] for row in tickers}

        selection = select_active_symbol_from_tickers(
            tickers,
            universe=universe,
            target_abs_change_pct=4.0,
            excluded_symbols=[],
            candidate_pool_size=10,
            min_abs_change_pct=3.0,
            max_abs_change_pct=5.0,
        )

        self.assertEqual(len(selection["top_candidates"]), 7)
        self.assertEqual(selection["symbol"], "TRADFI5USDT")
        self.assertNotIn("OUTSIDEUSDT", [row["symbol"] for row in selection["top_candidates"]])

    def test_tradfi_band_limits_pool_to_ten_when_more_than_ten(self):
        tickers = [
            {
                "symbol": f"TRADFI{idx}USDT",
                "priceChangePercent": str(4.0 + (idx * 0.01)),
                "lastPrice": "10",
                "quoteVolume": str(1000 + idx),
            }
            for idx in range(12)
        ]
        tickers[9]["quoteVolume"] = "999999"
        tickers[10]["quoteVolume"] = "999999999"
        universe = {row["symbol"] for row in tickers}

        selection = select_active_symbol_from_tickers(
            tickers,
            universe=universe,
            target_abs_change_pct=4.0,
            excluded_symbols=[],
            candidate_pool_size=10,
            min_abs_change_pct=3.0,
            max_abs_change_pct=5.0,
        )

        self.assertEqual(len(selection["top_candidates"]), 10)
        self.assertEqual(selection["symbol"], "TRADFI9USDT")
        self.assertNotIn("TRADFI10USDT", [row["symbol"] for row in selection["top_candidates"]])


if __name__ == "__main__":
    unittest.main()
