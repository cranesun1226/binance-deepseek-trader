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

    def test_market_entry_reconciles_unknown_execution_by_client_order_id(self):
        observed_query = {}

        def query_order(_api_key, _api_secret, symbol, client_order_id):
            observed_query["symbol"] = symbol
            observed_query["client_order_id"] = client_order_id
            return {"orderId": 123, "clientOrderId": client_order_id, "status": "FILLED"}

        with patch(
            "src.binance.trade_position.adjust_qty_for_symbol",
            return_value=Decimal("1"),
        ), patch(
            "src.binance.trade_position.set_leverage",
            return_value=1,
        ), patch(
            "src.binance.trade_position._signed_request",
            side_effect=trade_position.BinanceExecutionStatusUnknown("unknown", status_code=503),
        ) as signed_request, patch(
            "src.binance.trade_position._query_order_by_client_id",
            side_effect=query_order,
        ), patch("src.binance.trade_position.time.sleep"), patch("src.binance.trade_position.logger"):
            order, code, message = trade_position.place_market_entry_order(
                "api-key",
                "api-secret",
                "BTCUSDT",
                "Buy",
                "1",
                leverage=2,
            )

        params = dict(signed_request.call_args.kwargs["params"])
        self.assertEqual(order["status"], "FILLED")
        self.assertIsNone(code)
        self.assertEqual(message, "")
        self.assertEqual(observed_query["symbol"], "BTCUSDT")
        self.assertEqual(params["newClientOrderId"], observed_query["client_order_id"])

    def test_close_position_reconciles_unknown_execution_by_client_order_id(self):
        observed_query = {}

        def query_order(_api_key, _api_secret, symbol, client_order_id):
            observed_query["symbol"] = symbol
            observed_query["client_order_id"] = client_order_id
            return {"orderId": 456, "clientOrderId": client_order_id, "status": "FILLED"}

        with patch("src.binance.trade_position._get_position_amt", return_value=2.0), patch(
            "src.binance.trade_position._adjust_close_qty_for_symbol",
            return_value=Decimal("2"),
        ), patch(
            "src.binance.trade_position._signed_request",
            side_effect=trade_position.BinanceExecutionStatusUnknown("unknown", status_code=503),
        ) as signed_request, patch(
            "src.binance.trade_position._query_order_by_client_id",
            side_effect=query_order,
        ), patch("src.binance.trade_position.time.sleep"), patch("src.binance.trade_position.logger"):
            closed = trade_position.close_position("api-key", "api-secret", "BTCUSDT", "Buy", "2")

        params = dict(signed_request.call_args.kwargs["params"])
        self.assertTrue(closed)
        self.assertEqual(params["reduceOnly"], "true")
        self.assertEqual(params["newClientOrderId"], observed_query["client_order_id"])
        self.assertEqual(observed_query["symbol"], "BTCUSDT")

    def test_stop_loss_reconciles_unknown_execution_by_client_algo_id(self):
        observed_query = {}

        def query_algo(_api_key, _api_secret, client_algo_id):
            observed_query["client_algo_id"] = client_algo_id
            return {"algoId": 789, "clientAlgoId": client_algo_id, "algoStatus": "NEW"}

        with patch("src.binance.trade_position.adjust_price_for_symbol", return_value=96.0), patch(
            "src.binance.trade_position._cancel_stop_orders",
            return_value=True,
        ), patch(
            "src.binance.trade_position._signed_request",
            side_effect=trade_position.BinanceExecutionStatusUnknown("unknown", status_code=503),
        ) as signed_request, patch(
            "src.binance.trade_position._query_algo_order_by_client_id",
            side_effect=query_algo,
        ), patch("src.binance.trade_position.time.sleep"), patch("src.binance.trade_position.logger"):
            result = trade_position.sync_existing_position_stop_loss(
                "api-key",
                "api-secret",
                "BTCUSDT",
                "Buy",
                stop_loss=96.0,
            )

        params = dict(signed_request.call_args.kwargs["params"])
        self.assertTrue(result["success"])
        self.assertEqual(result["order"]["algoStatus"], "NEW")
        self.assertEqual(params["clientAlgoId"], observed_query["client_algo_id"])

    def test_stop_loss_trigger_price_uses_plain_decimal_for_low_priced_symbols(self):
        with patch("src.binance.trade_position.adjust_price_for_symbol", return_value=0.0000995), patch(
            "src.binance.trade_position._cancel_stop_orders",
            return_value=True,
        ), patch(
            "src.binance.trade_position._signed_post_expect_key",
            return_value=({"algoId": 789, "clientAlgoId": "client-id", "algoStatus": "NEW"}, None, ""),
        ) as signed_post, patch("src.binance.trade_position.logger"):
            result = trade_position.sync_existing_position_stop_loss(
                "api-key",
                "api-secret",
                "SPELLUSDT",
                "Sell",
                stop_loss=0.0000995,
            )

        params = dict(signed_post.call_args.kwargs["params"])
        self.assertTrue(result["success"])
        self.assertEqual(params["triggerPrice"], "0.0000995")
        self.assertNotIn("e", params["triggerPrice"].lower())

    def test_unknown_execution_reconciliation_ignores_ineffective_order_status(self):
        with patch("src.binance.trade_position.time.sleep"), patch("src.binance.trade_position.logger"):
            reconciled = trade_position._reconcile_unknown_execution(
                operation_name="place_market_entry_order(BTCUSDT)",
                reconcile_on_unknown=lambda: {
                    "orderId": 999,
                    "clientOrderId": "bdt_1",
                    "status": "EXPIRED",
                },
            )

        self.assertIsNone(reconciled)


if __name__ == "__main__":
    unittest.main()
