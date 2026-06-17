import unittest
from unittest.mock import patch

from src.binance import common


class BinanceCommonTests(unittest.TestCase):
    def test_binance_futures_base_url_defaults_to_production(self):
        with patch("src.binance.common._load_config_value", return_value=None):
            self.assertEqual(common.get_binance_futures_base_url(), common.BINANCE_FUTURES_BASE_URL)

    def test_binance_futures_base_url_uses_testnet_flag(self):
        def config_value(key):
            return {"BINANCE_FUTURES_TESTNET": "true"}.get(key)

        with patch("src.binance.common._load_config_value", side_effect=config_value):
            self.assertEqual(common.get_binance_futures_base_url(), common.BINANCE_FUTURES_TESTNET_BASE_URL)

    def test_binance_futures_base_url_override_takes_precedence(self):
        def config_value(key):
            return {
                "BINANCE_FUTURES_BASE_URL": "https://example.binance.local/",
                "BINANCE_FUTURES_TESTNET": "true",
            }.get(key)

        with patch("src.binance.common._load_config_value", side_effect=config_value):
            self.assertEqual(common.get_binance_futures_base_url(), "https://example.binance.local")


if __name__ == "__main__":
    unittest.main()
