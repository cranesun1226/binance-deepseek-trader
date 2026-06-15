import unittest
from decimal import Decimal
from unittest.mock import patch

from src.binance import trade_position


class TradePositionTests(unittest.TestCase):
    def test_short_position_negative_notional_becomes_positive_position_value(self):
        payload = trade_position._normalize_position_risk_payload(
            {
                "symbol": "ETHUSDT",
                "positionAmt": "-2",
                "entryPrice": "100.5",
                "notional": "-201",
            }
        )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["side"], "Sell")
        self.assertEqual(payload["positionValue"], 201.0)

        metrics = trade_position.calculate_position_metrics(payload)
        self.assertEqual(metrics["direction"], "short")
        self.assertEqual(metrics["position_value"], 201.0)

    def test_tradfi_agreement_error_is_not_retried(self):
        with patch(
            "src.binance.trade_position.adjust_qty_for_symbol",
            return_value=Decimal("3.16"),
        ), patch(
            "src.binance.trade_position.set_leverage",
            return_value=1,
        ), patch(
            "src.binance.trade_position._signed_post_expect_key",
            return_value=(None, -4411, "Please sign TradFi-Perps agreement contract fapi."),
        ) as signed_post, patch("src.binance.trade_position.logger"):
            order, code, message = trade_position.place_market_entry_order(
                "api-key",
                "api-secret",
                "CLUSDT",
                "Sell",
                "3.16",
                leverage=1,
            )

        self.assertIsNone(order)
        self.assertEqual(code, -4411)
        self.assertIn("TradFi-Perps", message)
        self.assertEqual(signed_post.call_count, 1)

    def test_region_restriction_error_is_not_retried(self):
        region_message = (
            "Dear user, as per our Terms of Use and compliance with local regulations, "
            "this feature is currently not available in your region."
        )
        with patch(
            "src.binance.trade_position.adjust_qty_for_symbol",
            return_value=Decimal("0.31"),
        ), patch(
            "src.binance.trade_position.set_leverage",
            return_value=2,
        ), patch(
            "src.binance.trade_position._signed_post_expect_key",
            return_value=(None, -4412, region_message),
        ) as signed_post, patch("src.binance.trade_position.logger"):
            order, code, message = trade_position.place_market_entry_order(
                "api-key",
                "api-secret",
                "SKHYNIXUSDT",
                "Buy",
                "0.31",
                leverage=2,
            )

        self.assertIsNone(order)
        self.assertEqual(code, -4412)
        self.assertIn("not available in your region", message)
        self.assertEqual(signed_post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
