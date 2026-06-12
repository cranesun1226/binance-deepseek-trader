import os
import tempfile
import unittest
from unittest.mock import patch

from src.strategy import portfolio_strategy


def _config():
    return {
        "cycle_interval_seconds": 60,
        "trigger_pct_usdt": 1.0,
        "fixed_leverage": 1,
        "stop_loss_pct": 0.04,
        "capital_usage_ratio": 0.99,
        "rebalance_threshold_pct": 0.03,
        "ai_prompt_timeframe": "1h",
        "ai_prompt_candle_count": 168,
        "deepseek_model": "deepseek-v4-flash",
        "deepseek_reasoning_effort": "max",
        "deepseek_max_tokens": 8192,
        "deepseek_timeout_seconds": 300.0,
        "passive_symbols": ["CLUSDT", "XAUUSDT", "QQQUSDT", "BTCUSDT"],
        "screener_quote": "USDT",
        "screener_timeout": 30.0,
        "screener_retries": 3,
        "screener_request_sleep": 0.1,
    }


def _active_slot():
    return portfolio_strategy.PortfolioSlot(
        slot_id="active_1",
        label="active1",
        kind="active",
        target_margin_ratio=0.25,
    )


def _active2_slot():
    return portfolio_strategy.PortfolioSlot(
        slot_id="active_2",
        label="active2",
        kind="active",
        target_margin_ratio=0.25,
        active_screening_mode="tradfi",
    )


def _passive_slot():
    return portfolio_strategy.PortfolioSlot(
        slot_id="passive_cl",
        label="CLUSDT",
        kind="passive",
        target_margin_ratio=0.125,
        symbol="CLUSDT",
    )


def _long_position(symbol="ETHUSDT"):
    return {
        "symbol": symbol,
        "positionAmt": "2",
        "side": "Buy",
        "entryPrice": "100",
        "markPrice": "100.5",
        "leverage": "1",
        "positionValue": "201",
    }


def _short_position(symbol="ETHUSDT"):
    return {
        "symbol": symbol,
        "positionAmt": "-2",
        "side": "Sell",
        "entryPrice": "100",
        "markPrice": "99.5",
        "leverage": "1",
        "positionValue": "199",
    }


class PortfolioFlowTests(unittest.TestCase):
    def test_active1_candidate_screening_uses_crypto_trend_screener(self):
        with patch(
            "src.strategy.portfolio_strategy.screen_active_symbol",
            return_value={
                "metadata": {"screening_mode": "crypto"},
                "selection": {"symbol": "ETHUSDT", "selected": {"symbol": "ETHUSDT"}},
            },
        ) as mocked_crypto, patch("src.strategy.portfolio_strategy.screen_active_tradfi_symbol") as mocked_tradfi:
            candidate = portfolio_strategy._screen_active_candidate(
                slot=_active_slot(),
                config=_config(),
                excluded_symbols=["BTCUSDT"],
            )

        self.assertEqual(candidate["symbol"], "ETHUSDT")
        mocked_crypto.assert_called_once()
        mocked_tradfi.assert_not_called()
        self.assertEqual(mocked_crypto.call_args.kwargs["quote"], "USDT")
        self.assertEqual(mocked_crypto.call_args.kwargs["required_kline_interval"], "1h")
        self.assertEqual(mocked_crypto.call_args.kwargs["required_kline_count"], 168)

    def test_active2_candidate_screening_uses_tradfi_screener(self):
        with patch(
            "src.strategy.portfolio_strategy.screen_active_tradfi_symbol",
            return_value={
                "metadata": {"screening_mode": "tradfi"},
                "selection": {"symbol": "ESUSDT", "selected": {"symbol": "ESUSDT"}},
            },
        ) as mocked_tradfi, patch("src.strategy.portfolio_strategy.screen_active_symbol") as mocked_crypto:
            candidate = portfolio_strategy._screen_active_candidate(
                slot=_active2_slot(),
                config=_config(),
                excluded_symbols=["CLUSDT"],
            )

        self.assertEqual(candidate["symbol"], "ESUSDT")
        mocked_tradfi.assert_called_once()
        mocked_crypto.assert_not_called()
        self.assertEqual(mocked_tradfi.call_args.kwargs["quote"], "USDT")
        self.assertEqual(mocked_tradfi.call_args.kwargs["required_kline_interval"], "1h")
        self.assertEqual(mocked_tradfi.call_args.kwargs["required_kline_count"], 168)

    def test_prompt_market_context_fetches_padding_and_serializes_requested_count(self):
        raw_klines = [[index, "0", "0", "0", str(float(index + 1))] for index in range(170)]

        with patch("src.strategy.portfolio_strategy.fetch_klines", return_value=raw_klines) as mocked_fetch:
            context = portfolio_strategy._fetch_prompt_market_context(
                symbol="BTCUSDT",
                ai_prompt_timeframe="1h",
                ai_prompt_candle_count=168,
                as_of_ms=123456,
                reference_price=999.0,
            )

        mocked_fetch.assert_called_once_with("BTCUSDT", "1h", 170, as_of_ms=123456)
        close_prices = context["timeframes"]["1h"]
        self.assertEqual(context["ai_prompt_candle_count"], 168)
        self.assertEqual(len(close_prices), 168)
        self.assertEqual(close_prices[0], 3.0)
        self.assertEqual(close_prices[-1], 999.0)

    def test_post_trade_direction_mismatch_is_closed_and_marked_failed(self):
        with patch(
            "src.strategy.portfolio_strategy.get_position_snapshot",
            return_value=_short_position("BTCUSDT"),
        ), patch("src.strategy.portfolio_strategy.close_position", return_value=True) as mocked_close, patch(
            "src.strategy.portfolio_strategy.wait_for_close_propagation"
        ), patch("src.strategy.portfolio_strategy.cancel_all_symbol_orders"):
            result = portfolio_strategy._sync_position_after_trade(
                api_key="key",
                api_secret="secret",
                symbol="BTCUSDT",
                stop_loss_pct=0.04,
                expected_decision="LONG",
            )

        self.assertEqual(result["direction_verification"]["action"], "post_trade_direction_mismatch_closed")
        self.assertEqual(result["direction_verification"]["expected_direction"], "long")
        self.assertEqual(result["direction_verification"]["actual_direction"], "short")
        mocked_close.assert_called_once()

    def test_non_ai_cycle_does_not_create_db_artifact(self):
        slot = _passive_slot()
        slot_state = {"slot_id": "passive_cl", "kind": "passive", "symbol": "CLUSDT"}
        slot_result = {
            "slot_id": "passive_cl",
            "slot_label": "CLUSDT",
            "symbol": "CLUSDT",
            "success": True,
            "action": "kept_position_size",
            "ai_triggered": False,
        }

        with patch("src.strategy.portfolio_strategy._load_strategy_config", return_value=_config()), patch(
            "src.strategy.portfolio_strategy._build_portfolio_slots", return_value=[slot]
        ), patch("src.strategy.portfolio_strategy.get_binance_credentials", return_value=("key", "secret")), patch(
            "src.strategy.portfolio_strategy.get_account_overview",
            return_value={"equity": 1000.0, "available_balance": 500.0},
        ), patch("src.strategy.portfolio_strategy.get_positions", return_value=[]), patch(
            "src.strategy.portfolio_strategy._run_passive_slot", return_value=(slot_result, slot_state)
        ), patch(
            "src.strategy.portfolio_strategy._create_cycle_dir",
            side_effect=AssertionError("non-AI cycle should not create db artifact"),
        ):
            result = portfolio_strategy.run_portfolio_cycle(state={"version": portfolio_strategy.STATE_VERSION})

        self.assertTrue(result["success"])
        self.assertFalse(result["ai_triggered"])
        self.assertIsNone(result["cycle_dir"])

    def test_ai_cycle_persists_db_artifact(self):
        slot = _passive_slot()
        slot_state = {
            "slot_id": "passive_cl",
            "kind": "passive",
            "symbol": "CLUSDT",
            "last_ai_decision": "SHORT",
        }
        slot_result = {
            "slot_id": "passive_cl",
            "slot_label": "CLUSDT",
            "symbol": "CLUSDT",
            "success": True,
            "action": "kept_position_by_ai",
            "ai_triggered": True,
            "ai_decision": "SHORT",
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.strategy.portfolio_strategy._load_strategy_config", return_value=_config()
        ), patch("src.strategy.portfolio_strategy._build_portfolio_slots", return_value=[slot]), patch(
            "src.strategy.portfolio_strategy.get_binance_credentials", return_value=("key", "secret")
        ), patch(
            "src.strategy.portfolio_strategy.get_account_overview",
            return_value={"equity": 1000.0, "available_balance": 500.0},
        ), patch("src.strategy.portfolio_strategy.get_positions", return_value=[]), patch(
            "src.strategy.portfolio_strategy._run_passive_slot", return_value=(slot_result, slot_state)
        ), patch(
            "src.strategy.portfolio_strategy._create_cycle_dir", return_value=temp_dir
        ):
            result = portfolio_strategy.run_portfolio_cycle(state={"version": portfolio_strategy.STATE_VERSION})
            artifact_exists = os.path.exists(os.path.join(temp_dir, "portfolio_cycle_output.json"))

        self.assertTrue(result["success"])
        self.assertTrue(result["ai_triggered"])
        self.assertEqual(result["cycle_dir"], temp_dir)
        self.assertTrue(artifact_exists)

    def test_material_position_event_persists_db_artifact_without_ai(self):
        slot = _passive_slot()
        slot_state = {"slot_id": "passive_cl", "kind": "passive", "symbol": "CLUSDT"}
        slot_result = {
            "slot_id": "passive_cl",
            "slot_label": "CLUSDT",
            "symbol": "CLUSDT",
            "success": True,
            "action": "kept_position_size",
            "ai_triggered": False,
            "execution": {
                "success": True,
                "action": "kept_position_size",
                "stop_sync": {"success": True, "changed": True},
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.strategy.portfolio_strategy._load_strategy_config", return_value=_config()
        ), patch("src.strategy.portfolio_strategy._build_portfolio_slots", return_value=[slot]), patch(
            "src.strategy.portfolio_strategy.get_binance_credentials", return_value=("key", "secret")
        ), patch(
            "src.strategy.portfolio_strategy.get_account_overview",
            return_value={"equity": 1000.0, "available_balance": 500.0},
        ), patch("src.strategy.portfolio_strategy.get_positions", return_value=[]), patch(
            "src.strategy.portfolio_strategy._run_passive_slot", return_value=(slot_result, slot_state)
        ), patch(
            "src.strategy.portfolio_strategy._create_cycle_dir", return_value=temp_dir
        ):
            result = portfolio_strategy.run_portfolio_cycle(state={"version": portfolio_strategy.STATE_VERSION})
            artifact_exists = os.path.exists(os.path.join(temp_dir, "portfolio_cycle_output.json"))

        self.assertTrue(result["success"])
        self.assertFalse(result["ai_triggered"])
        self.assertEqual(result["cycle_dir"], temp_dir)
        self.assertTrue(artifact_exists)

    def test_passive_opposite_direction_reverses_same_symbol_without_rescreening(self):
        slot = _passive_slot()
        slot_state = {
            "slot_id": "passive_cl",
            "kind": "passive",
            "symbol": "CLUSDT",
            "last_ai_trigger_price": 100.0,
            "next_trigger_down": 99.0,
            "next_trigger_up": 101.0,
            "last_ai_decision": "LONG",
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.strategy.portfolio_strategy._reference_price", return_value=101.5
        ), patch(
            "src.strategy.portfolio_strategy._evaluate_slot_direction",
            return_value=("SHORT", {"reasoning": "reverse"}, {}),
        ) as mocked_ai, patch(
            "src.strategy.portfolio_strategy._rebalance_existing_position",
            return_value={
                "success": True,
                "action": "reversed_position",
                "position": {"symbol": "CLUSDT", "direction": "short"},
            },
        ) as mocked_rebalance, patch(
            "src.strategy.portfolio_strategy._screen_active_candidate"
        ) as mocked_screen:
            result, updated_state = portfolio_strategy._run_passive_slot(
                slot=slot,
                slot_state=slot_state,
                config=_config(),
                api_key="key",
                api_secret="secret",
                account_overview={"equity": 1000.0, "available_balance": 500.0},
                position=_long_position("CLUSDT"),
                as_of_ms=1,
                cycle_dir_factory=lambda: temp_dir,
                notification_callback=None,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["symbol"], "CLUSDT")
        self.assertEqual(result["action"], "reversed_position")
        self.assertEqual(result["ai_decision"], "SHORT")
        self.assertEqual(updated_state["symbol"], "CLUSDT")
        self.assertEqual(updated_state["last_ai_decision"], "SHORT")
        mocked_ai.assert_called_once()
        mocked_rebalance.assert_called_once()
        self.assertEqual(mocked_rebalance.call_args.kwargs["symbol"], "CLUSDT")
        self.assertEqual(mocked_rebalance.call_args.kwargs["decision"], "SHORT")
        mocked_screen.assert_not_called()

    def test_active_existing_same_direction_does_not_rescreen(self):
        slot = _active_slot()
        slot_state = {
            "slot_id": "active_1",
            "kind": "active",
            "symbol": "ETHUSDT",
            "last_ai_trigger_price": 100.0,
            "next_trigger_down": 99.0,
            "next_trigger_up": 101.0,
            "last_ai_decision": "LONG",
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.strategy.portfolio_strategy._reference_price", return_value=101.5
        ), patch(
            "src.strategy.portfolio_strategy._evaluate_slot_direction",
            return_value=("LONG", {"reasoning": "keep"}, {}),
        ) as mocked_ai, patch(
            "src.strategy.portfolio_strategy._rebalance_existing_position",
            return_value={"success": True, "action": "kept_position_size", "position": {"symbol": "ETHUSDT"}},
        ) as mocked_rebalance, patch(
            "src.strategy.portfolio_strategy._screen_active_candidate"
        ) as mocked_screen:
            result, updated_state, active_symbol = portfolio_strategy._run_active_slot(
                slot=slot,
                slot_state=slot_state,
                config=_config(),
                api_key="key",
                api_secret="secret",
                account_overview={"equity": 1000.0, "available_balance": 500.0},
                open_positions=[_long_position()],
                reserved_symbols=set(),
                as_of_ms=1,
                cycle_dir_factory=lambda: temp_dir,
                notification_callback=None,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "kept_position_by_ai")
        self.assertEqual(active_symbol, "ETHUSDT")
        self.assertEqual(updated_state["symbol"], "ETHUSDT")
        mocked_ai.assert_called_once()
        mocked_rebalance.assert_called_once()
        mocked_screen.assert_not_called()

    def test_active_screener_failure_does_not_create_db_artifact_without_position_event(self):
        slot = _active_slot()
        slot_state = {
            "slot_id": "active_1",
            "kind": "active",
            "symbol": None,
        }

        def fail_cycle_dir():
            raise AssertionError("screener failure should not create db artifact")

        with patch(
            "src.strategy.portfolio_strategy._screen_active_candidate",
            side_effect=RuntimeError("screener unavailable"),
        ):
            result, updated_state, active_symbol = portfolio_strategy._run_active_slot(
                slot=slot,
                slot_state=slot_state,
                config=_config(),
                api_key="key",
                api_secret="secret",
                account_overview={"equity": 1000.0, "available_balance": 500.0},
                open_positions=[],
                reserved_symbols=set(),
                as_of_ms=1,
                cycle_dir_factory=fail_cycle_dir,
                notification_callback=None,
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["action"], "screener_selection_failed")
        self.assertIsNone(active_symbol)
        self.assertEqual(updated_state, slot_state)

    def test_active_without_qualified_kline_candidate_waits_quietly(self):
        slot = _active_slot()
        slot_state = {
            "slot_id": "active_1",
            "kind": "active",
            "symbol": None,
        }

        def fail_cycle_dir():
            raise AssertionError("waiting for active candidate should not create db artifact")

        with patch(
            "src.strategy.portfolio_strategy._screen_active_candidate",
            side_effect=portfolio_strategy.NoActiveCandidateError(
                "active crypto trend screener found no candidate with at least 168 1h valid close-price klines"
            ),
        ):
            result, updated_state, active_symbol = portfolio_strategy._run_active_slot(
                slot=slot,
                slot_state=slot_state,
                config=_config(),
                api_key="key",
                api_secret="secret",
                account_overview={"equity": 1000.0, "available_balance": 500.0},
                open_positions=[],
                reserved_symbols=set(),
                as_of_ms=1,
                cycle_dir_factory=fail_cycle_dir,
                notification_callback=None,
            )

        self.assertTrue(result["success"])
        self.assertFalse(result["ai_triggered"])
        self.assertEqual(result["action"], "waiting_for_active_candidate")
        self.assertIsNone(active_symbol)
        self.assertEqual(updated_state, slot_state)

    def test_active_opposite_direction_closes_then_screens_new_candidate(self):
        slot = _active_slot()
        slot_state = {
            "slot_id": "active_1",
            "kind": "active",
            "symbol": "ETHUSDT",
            "last_ai_trigger_price": 100.0,
            "next_trigger_down": 99.0,
            "next_trigger_up": 101.0,
            "last_ai_decision": "LONG",
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.strategy.portfolio_strategy._reference_price",
            side_effect=[101.5, 55.0],
        ), patch(
            "src.strategy.portfolio_strategy._evaluate_slot_direction",
            side_effect=[("SHORT", {"reasoning": "exit"}, {}), ("LONG", {"reasoning": "enter"}, {})],
        ), patch(
            "src.strategy.portfolio_strategy._close_existing_position",
            return_value={"success": True, "action": "closed_position", "symbol": "ETHUSDT"},
        ) as mocked_close, patch(
            "src.strategy.portfolio_strategy._screen_active_candidate",
            return_value={"symbol": "SOLUSDT", "selection": {}, "metadata": {}},
        ) as mocked_screen, patch(
            "src.strategy.portfolio_strategy.get_account_overview",
            return_value={"equity": 1000.0, "available_balance": 500.0},
        ), patch(
            "src.strategy.portfolio_strategy._place_direction_position",
            return_value={"success": True, "action": "opened_new_position", "position": {"symbol": "SOLUSDT"}},
        ) as mocked_place, patch(
            "src.strategy.portfolio_strategy._sync_position_after_trade",
            return_value={"position": {"symbol": "SOLUSDT"}, "stop_sync": {"success": True}},
        ):
            result, updated_state, active_symbol = portfolio_strategy._run_active_slot(
                slot=slot,
                slot_state=slot_state,
                config=_config(),
                api_key="key",
                api_secret="secret",
                account_overview={"equity": 1000.0, "available_balance": 500.0},
                open_positions=[_long_position()],
                reserved_symbols={"CLUSDT"},
                as_of_ms=1,
                cycle_dir_factory=lambda: temp_dir,
                notification_callback=None,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["symbol"], "SOLUSDT")
        self.assertEqual(updated_state["symbol"], "SOLUSDT")
        self.assertEqual(active_symbol, "SOLUSDT")
        mocked_close.assert_called_once()
        mocked_screen.assert_called_once()
        self.assertIn("ETHUSDT", mocked_screen.call_args.kwargs["excluded_symbols"])
        mocked_place.assert_called_once()


if __name__ == "__main__":
    unittest.main()
