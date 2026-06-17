import os
import tempfile
import unittest
from unittest.mock import patch

from src.strategy import portfolio_strategy


def _config():
    return {
        "cycle_interval_seconds": 60,
        "trigger_pct_usdt": 1.0,
        "active_leverage": 2,
        "stop_loss_pct": 0.04,
        "capital_usage_ratio": 0.99,
        "rebalance_threshold_pct": 0.02,
        "active_rescreen_interval_hours": 24,
        "ai_prompt_timeframe": "1h",
        "ai_prompt_candle_count": 168,
        "deepseek_model": "deepseek-v4-flash",
        "deepseek_reasoning_effort": "max",
        "deepseek_max_tokens": 8192,
        "deepseek_timeout_seconds": 300.0,
        "screener_quote": "USDT",
        "screener_timeout": 30.0,
        "screener_retries": 3,
        "screener_request_sleep": 0.1,
    }


def _active_slot(slot_id="active_1", screening_mode="tradfi"):
    return portfolio_strategy.PortfolioSlot(
        slot_id=slot_id,
        label=slot_id.replace("_", ""),
        kind="active",
        target_margin_ratio=0.25,
        active_screening_mode=screening_mode,
    )


def _long_position(symbol="ESUSDT"):
    return {
        "symbol": symbol,
        "positionAmt": "2",
        "side": "Buy",
        "entryPrice": "100",
        "markPrice": "100.5",
        "leverage": "2",
        "positionValue": "201",
    }


def _short_position(symbol="ESUSDT"):
    return {
        "symbol": symbol,
        "positionAmt": "-2",
        "side": "Sell",
        "entryPrice": "100",
        "markPrice": "99.5",
        "leverage": "2",
        "positionValue": "199",
    }


def _close_prices(start=100.0, step=0.1, count=168):
    return [start + (step * index) for index in range(count)]


class PortfolioFlowTests(unittest.TestCase):
    def test_tradfi_active_candidate_screening_uses_tradfi_screener(self):
        with patch(
            "src.strategy.portfolio_strategy.screen_active_tradfi_symbol",
            return_value={
                "metadata": {"screening_mode": "tradfi"},
                "selection": {
                    "symbol": "ESUSDT",
                    "screening_decision": "LONG",
                    "screening_direction": "up",
                    "selected": {"symbol": "ESUSDT", "screening_decision": "LONG", "screening_direction": "up"},
                },
            },
        ) as mocked_tradfi, patch("src.strategy.portfolio_strategy.screen_active_symbol") as mocked_crypto:
            candidate = portfolio_strategy._screen_active_candidate(
                slot=_active_slot("active_1", "tradfi"),
                config=_config(),
                excluded_symbols=["BTCUSDT"],
            )

        self.assertEqual(candidate["symbol"], "ESUSDT")
        self.assertEqual(candidate["screening_decision"], "LONG")
        self.assertEqual(candidate["decision_source"], "active_screener")
        mocked_tradfi.assert_called_once()
        mocked_crypto.assert_not_called()
        self.assertEqual(mocked_tradfi.call_args.kwargs["quote"], "USDT")
        self.assertEqual(mocked_tradfi.call_args.kwargs["required_kline_interval"], "1h")
        self.assertEqual(mocked_tradfi.call_args.kwargs["required_kline_count"], 168)

    def test_crypto_active_candidate_screening_uses_crypto_screener(self):
        with patch(
            "src.strategy.portfolio_strategy.screen_active_symbol",
            return_value={
                "metadata": {"screening_mode": "crypto"},
                "selection": {
                    "symbol": "BTCUSDT",
                    "screening_decision": "SHORT",
                    "screening_direction": "down",
                    "selected": {"symbol": "BTCUSDT", "screening_decision": "SHORT", "screening_direction": "down"},
                },
            },
        ) as mocked_crypto, patch("src.strategy.portfolio_strategy.screen_active_tradfi_symbol") as mocked_tradfi:
            candidate = portfolio_strategy._screen_active_candidate(
                slot=_active_slot("active_4", "crypto"),
                config=_config(),
                excluded_symbols=["ESUSDT"],
            )

        self.assertEqual(candidate["symbol"], "BTCUSDT")
        self.assertEqual(candidate["screening_decision"], "SHORT")
        mocked_crypto.assert_called_once()
        mocked_tradfi.assert_not_called()

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

    def test_post_trade_stop_loss_failure_closes_position_and_marks_failed(self):
        stop_failure = {"success": False, "changed": False, "reason": "place_stop_loss_failed"}
        close_result = {"success": True, "action": "closed_position", "symbol": "BTCUSDT", "qty": 2.0}

        with patch(
            "src.strategy.portfolio_strategy.get_position_snapshot",
            return_value=_long_position("BTCUSDT"),
        ), patch(
            "src.strategy.portfolio_strategy._sync_fixed_stop_loss",
            return_value=stop_failure,
        ), patch(
            "src.strategy.portfolio_strategy._close_existing_position",
            return_value=close_result,
        ) as mocked_close:
            sync_result = portfolio_strategy._sync_position_after_trade(
                api_key="key",
                api_secret="secret",
                symbol="BTCUSDT",
                stop_loss_pct=0.04,
                expected_decision="LONG",
            )

        merged = portfolio_strategy._merge_post_trade_sync_result(
            {"success": True, "action": "opened_new_position"},
            sync_result,
        )

        self.assertEqual(sync_result["stop_sync"], stop_failure)
        self.assertEqual(sync_result["protection_verification"]["action"], "post_trade_stop_loss_sync_failed_closed")
        self.assertFalse(merged["success"])
        self.assertEqual(merged["action"], "post_trade_stop_loss_sync_failed_closed")
        mocked_close.assert_called_once()
        self.assertEqual(mocked_close.call_args.kwargs["context"], "post_trade_stop_loss_sync_failed")

    def test_non_ai_cycle_does_not_create_db_artifact(self):
        slot = _active_slot()
        slot_state = {"slot_id": "active_1", "kind": "active", "symbol": "ESUSDT"}
        slot_result = {
            "slot_id": "active_1",
            "slot_label": "active1",
            "symbol": "ESUSDT",
            "success": True,
            "action": "kept_active_position_until_stop_loss",
            "ai_triggered": False,
            "screening_triggered": False,
        }

        with patch("src.strategy.portfolio_strategy._load_strategy_config", return_value=_config()), patch(
            "src.strategy.portfolio_strategy._build_portfolio_slots", return_value=[slot]
        ), patch("src.strategy.portfolio_strategy.get_binance_credentials", return_value=("key", "secret")), patch(
            "src.strategy.portfolio_strategy.get_account_overview",
            return_value={"equity": 1000.0, "available_balance": 500.0},
        ), patch("src.strategy.portfolio_strategy.get_positions", return_value=[]), patch(
            "src.strategy.portfolio_strategy._run_active_slot", return_value=(slot_result, slot_state, "ESUSDT")
        ), patch(
            "src.strategy.portfolio_strategy._create_cycle_dir",
            side_effect=AssertionError("non-AI cycle should not create db artifact"),
        ):
            result = portfolio_strategy.run_portfolio_cycle(state={"version": portfolio_strategy.STATE_VERSION})

        self.assertTrue(result["success"])
        self.assertFalse(result["ai_triggered"])
        self.assertIsNone(result["cycle_dir"])

    def test_ai_cycle_persists_db_artifact(self):
        slot = _active_slot()
        slot_state = {
            "slot_id": "active_1",
            "kind": "active",
            "symbol": "ESUSDT",
            "last_ai_decision": "SHORT",
        }
        slot_result = {
            "slot_id": "active_1",
            "slot_label": "active1",
            "symbol": "ESUSDT",
            "success": True,
            "action": "opened_new_position",
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
            "src.strategy.portfolio_strategy._run_active_slot", return_value=(slot_result, slot_state, "ESUSDT")
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
        slot = _active_slot()
        slot_state = {"slot_id": "active_1", "kind": "active", "symbol": "ESUSDT"}
        slot_result = {
            "slot_id": "active_1",
            "slot_label": "active1",
            "symbol": "ESUSDT",
            "success": True,
            "action": "kept_active_position_until_stop_loss",
            "ai_triggered": False,
            "screening_triggered": False,
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
            "src.strategy.portfolio_strategy._run_active_slot", return_value=(slot_result, slot_state, "ESUSDT")
        ), patch(
            "src.strategy.portfolio_strategy._create_cycle_dir", return_value=temp_dir
        ):
            result = portfolio_strategy.run_portfolio_cycle(state={"version": portfolio_strategy.STATE_VERSION})
            artifact_exists = os.path.exists(os.path.join(temp_dir, "portfolio_cycle_output.json"))

        self.assertTrue(result["success"])
        self.assertFalse(result["ai_triggered"])
        self.assertEqual(result["cycle_dir"], temp_dir)
        self.assertTrue(artifact_exists)

    def test_active_region_restriction_records_runtime_symbol_ban_for_next_cycle(self):
        slot1 = _active_slot()
        seen_bans = []

        def run_active_slot(**kwargs):
            seen_bans.append(tuple(kwargs.get("runtime_banned_symbols") or ()))
            slot = kwargs["slot"]
            slot_state = kwargs["slot_state"]
            slot_result = portfolio_strategy._slot_result_base(slot, "SKHYNIXUSDT")
            slot_result.update(
                {
                    "success": False,
                    "action": "entry_order_failed",
                    "screening_triggered": True,
                    "screening_decision": "LONG",
                    "execution": {
                        "success": False,
                        "action": "entry_order_failed",
                        "order_error_code": -4412,
                        "order_error_message": "not available in your region",
                        "symbol_ban": {
                            "symbol": "SKHYNIXUSDT",
                            "reason": "binance_region_restricted",
                            "source": "entry_order",
                            "error_code": -4412,
                            "error_message": "not available in your region",
                        },
                    },
                }
            )
            return slot_result, slot_state, "SKHYNIXUSDT"

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.strategy.portfolio_strategy._load_strategy_config", return_value=_config()
        ), patch(
            "src.strategy.portfolio_strategy._build_portfolio_slots",
            return_value=[slot1],
        ), patch(
            "src.strategy.portfolio_strategy.get_binance_credentials", return_value=("key", "secret")
        ), patch(
            "src.strategy.portfolio_strategy.get_account_overview",
            return_value={"equity": 1000.0, "available_balance": 500.0},
        ), patch("src.strategy.portfolio_strategy.get_positions", return_value=[]), patch(
            "src.strategy.portfolio_strategy._run_active_slot", side_effect=run_active_slot
        ), patch(
            "src.strategy.portfolio_strategy._create_cycle_dir", return_value=temp_dir
        ):
            result = portfolio_strategy.run_portfolio_cycle(
                state={"version": portfolio_strategy.STATE_VERSION},
                as_of_ms=2000,
            )

        self.assertFalse(result["success"])
        ban = result["state_update"]["symbol_bans"]["SKHYNIXUSDT"]
        self.assertEqual(ban["reason"], "binance_region_restricted")
        self.assertEqual(ban["last_seen_at"], "1970-01-01T00:00:02Z")
        self.assertEqual(ban["event_count"], 1)
        self.assertEqual(seen_bans, [()])
        self.assertIn("SKHYNIXUSDT", portfolio_strategy._runtime_banned_symbols(result["state_update"]))

    def test_four_active_slots_reserve_distinct_symbols_in_one_cycle(self):
        slots = portfolio_strategy._build_portfolio_slots({})
        screen_exclusions = []

        def screen_candidate(**kwargs):
            screen_exclusions.append((kwargs["slot"].slot_id, kwargs["slot"].active_screening_mode, set(kwargs["excluded_symbols"])))
            symbols = {
                "active_1": "ESUSDT",
                "active_2": "NQUSDT",
                "active_3": "GCUSDT",
                "active_4": "BTCUSDT",
            }
            return {
                "symbol": symbols[kwargs["slot"].slot_id],
                "screening_decision": "LONG",
                "screening_direction": "up",
                "close_prices": [100.0, 101.0, 102.0],
                "decision_source": "active_screener",
                "selection": {},
                "metadata": {},
            }

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.strategy.portfolio_strategy._load_strategy_config", return_value=_config()
        ), patch(
            "src.strategy.portfolio_strategy._build_portfolio_slots",
            return_value=slots,
        ), patch(
            "src.strategy.portfolio_strategy.get_binance_credentials", return_value=("key", "secret")
        ), patch(
            "src.strategy.portfolio_strategy.get_account_overview",
            return_value={"equity": 1000.0, "available_balance": 500.0},
        ), patch("src.strategy.portfolio_strategy.get_positions", return_value=[]), patch(
            "src.strategy.portfolio_strategy._screen_active_candidate", side_effect=screen_candidate
        ), patch(
            "src.strategy.portfolio_strategy._reference_price", return_value=102.0
        ), patch(
            "src.strategy.portfolio_strategy._evaluate_slot_direction",
            return_value=(
                "LONG",
                {"decision": {"decision": "LONG", "reason": "DeepSeek confirms the active setup."}},
                {
                    "ai_prompt_timeframe": "1h",
                    "ai_prompt_candle_count": 3,
                    "ai_prompt_close_prices": [100.0, 101.0, 102.0],
                },
            ),
        ), patch(
            "src.strategy.portfolio_strategy._place_direction_position",
            return_value={"success": True, "action": "opened_new_position"},
        ), patch(
            "src.strategy.portfolio_strategy._sync_position_after_trade",
            return_value={"position": {"direction": "long"}, "stop_sync": {"success": True}},
        ), patch(
            "src.strategy.portfolio_strategy._create_cycle_dir", return_value=temp_dir
        ):
            result = portfolio_strategy.run_portfolio_cycle(
                state={"version": portfolio_strategy.STATE_VERSION},
                as_of_ms=1000,
            )

        self.assertTrue(result["success"])
        self.assertEqual([row["slot_id"] for row in result["slot_results"]], ["active_1", "active_2", "active_3", "active_4"])
        self.assertEqual([row["symbol"] for row in result["slot_results"]], ["ESUSDT", "NQUSDT", "GCUSDT", "BTCUSDT"])
        self.assertNotIn("ESUSDT", screen_exclusions[0][2])
        self.assertIn("ESUSDT", screen_exclusions[1][2])
        self.assertIn("NQUSDT", screen_exclusions[2][2])
        self.assertIn("GCUSDT", screen_exclusions[3][2])
        self.assertEqual(screen_exclusions[3][1], "crypto")

    def test_slot_exception_is_captured_and_following_slots_continue(self):
        active1 = _active_slot("active_1")
        active2 = _active_slot("active_2")

        def run_active_slot(**kwargs):
            if kwargs["slot"].slot_id == "active_1":
                raise RuntimeError("boom")
            slot_result = portfolio_strategy._slot_result_base(kwargs["slot"], None)
            slot_result.update({"success": True, "action": "waiting_for_active_candidate"})
            return slot_result, kwargs["slot_state"], None

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.strategy.portfolio_strategy._load_strategy_config", return_value=_config()
        ), patch(
            "src.strategy.portfolio_strategy._build_portfolio_slots",
            return_value=[active1, active2],
        ), patch(
            "src.strategy.portfolio_strategy.get_binance_credentials", return_value=("key", "secret")
        ), patch(
            "src.strategy.portfolio_strategy.get_account_overview",
            return_value={"equity": 1000.0, "available_balance": 500.0},
        ), patch("src.strategy.portfolio_strategy.get_positions", return_value=[]), patch(
            "src.strategy.portfolio_strategy._run_active_slot", side_effect=run_active_slot
        ), patch(
            "src.strategy.portfolio_strategy._create_cycle_dir", return_value=temp_dir
        ), patch(
            "src.strategy.portfolio_strategy.logger"
        ):
            result = portfolio_strategy.run_portfolio_cycle(
                state={"version": portfolio_strategy.STATE_VERSION},
                as_of_ms=2000,
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["slot_results"][0]["action"], "slot_execution_failed")
        self.assertEqual(result["slot_results"][0]["error"], "boom")
        self.assertEqual(result["slot_results"][1]["action"], "waiting_for_active_candidate")

    def test_active_existing_position_waiting_for_price_trigger_only_syncs_stop_loss(self):
        slot = _active_slot()
        slot_state = {
            "slot_id": "active_1",
            "kind": "active",
            "symbol": "ESUSDT",
            "active_screening_mode": "tradfi",
            "last_ai_trigger_price": 100.0,
            "next_trigger_down": 99.0,
            "next_trigger_up": 101.0,
            "last_ai_decision": "LONG",
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.strategy.portfolio_strategy._reference_price", return_value=100.5
        ), patch(
            "src.strategy.portfolio_strategy._evaluate_slot_direction",
        ) as mocked_ai, patch(
            "src.strategy.portfolio_strategy._rebalance_existing_position",
            return_value={"success": True, "action": "kept_position_size", "position": {"symbol": "ESUSDT"}},
        ) as mocked_rebalance, patch(
            "src.strategy.portfolio_strategy._screen_active_candidate"
        ) as mocked_screen, patch(
            "src.strategy.portfolio_strategy._sync_fixed_stop_loss",
            return_value={"success": True, "changed": False, "stop_loss_pct": 0.04},
        ) as mocked_stop:
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
        self.assertFalse(result["ai_triggered"])
        self.assertFalse(result["screening_triggered"])
        self.assertEqual(result["action"], "kept_active_position_until_stop_loss")
        self.assertEqual(active_symbol, "ESUSDT")
        self.assertEqual(updated_state["symbol"], "ESUSDT")
        self.assertEqual(updated_state["last_ai_decision"], "LONG")
        self.assertEqual(updated_state["entered_at"], "1970-01-01T00:00:00Z")
        self.assertEqual(updated_state["last_active_rank_checked_at"], "1970-01-01T00:00:00Z")
        mocked_ai.assert_not_called()
        mocked_rebalance.assert_not_called()
        mocked_screen.assert_not_called()
        mocked_stop.assert_called_once()

    def test_active_existing_position_uses_price_trigger_for_deepseek_review(self):
        slot = _active_slot()
        slot_state = {
            "slot_id": "active_1",
            "kind": "active",
            "symbol": "ESUSDT",
            "active_screening_mode": "tradfi",
            "last_ai_trigger_price": 100.0,
            "next_trigger_down": 99.0,
            "next_trigger_up": 101.0,
            "last_ai_decision": "LONG",
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.strategy.portfolio_strategy._reference_price", return_value=101.5
        ), patch(
            "src.strategy.portfolio_strategy._evaluate_slot_direction",
            return_value=(
                "SHORT",
                {"decision": {"decision": "SHORT", "reason": "DeepSeek flips downside."}},
                {
                    "ai_prompt_timeframe": "1h",
                    "ai_prompt_candle_count": 3,
                    "ai_prompt_close_prices": [100.0, 101.0, 101.5],
                },
            ),
        ) as mocked_ai, patch(
            "src.strategy.portfolio_strategy._rebalance_existing_position",
            return_value={
                "success": True,
                "action": "reversed_position",
                "position": {"symbol": "ESUSDT", "direction": "short"},
                "stop_sync": {"success": True},
            },
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
        self.assertTrue(result["ai_triggered"])
        self.assertFalse(result["screening_triggered"])
        self.assertEqual(result["trigger_reason"], "price_distance_reached")
        self.assertEqual(result["ai_decision"], "SHORT")
        self.assertEqual(result["action"], "reversed_position")
        self.assertEqual(active_symbol, "ESUSDT")
        self.assertEqual(updated_state["symbol"], "ESUSDT")
        self.assertEqual(updated_state["last_ai_decision"], "SHORT")
        self.assertEqual(updated_state["last_ai_trigger_price"], 101.5)
        self.assertEqual(updated_state["next_trigger_down"], 100.485)
        self.assertEqual(updated_state["next_trigger_up"], 102.515)
        mocked_ai.assert_called_once()
        mocked_screen.assert_not_called()
        mocked_rebalance.assert_called_once()
        self.assertEqual(mocked_rebalance.call_args.kwargs["decision"], "SHORT")

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
            side_effect=portfolio_strategy.NoActiveCandidateError("active tradfi trend screener found no candidate"),
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

    def test_active_without_position_excludes_stale_and_same_universe_recent_symbols(self):
        slot = _active_slot()
        slot_state = {
            "slot_id": "active_1",
            "kind": "active",
            "symbol": "ESUSDT",
            "previous_active_symbol": "NQUSDT",
            "active_screening_mode": "tradfi",
        }

        def fail_cycle_dir():
            raise AssertionError("waiting for active candidate should not create db artifact")

        with patch(
            "src.strategy.portfolio_strategy._screen_active_candidate",
            side_effect=portfolio_strategy.NoActiveCandidateError("no candidate"),
        ) as mocked_screen:
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
                recent_universe_symbols={"YMUSDT"},
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "waiting_for_active_candidate")
        self.assertIsNone(active_symbol)
        self.assertEqual(updated_state, slot_state)
        excluded_symbols = set(mocked_screen.call_args.kwargs["excluded_symbols"])
        self.assertIn("ESUSDT", excluded_symbols)
        self.assertIn("NQUSDT", excluded_symbols)
        self.assertIn("YMUSDT", excluded_symbols)

    def test_active_replacement_remembers_stale_symbol_as_previous(self):
        slot = _active_slot()
        slot_state = {
            "slot_id": "active_1",
            "kind": "active",
            "symbol": "ESUSDT",
            "previous_active_symbol": "NQUSDT",
            "active_screening_mode": "tradfi",
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.strategy.portfolio_strategy._screen_active_candidate",
            return_value={
                "symbol": "YMUSDT",
                "screening_decision": "LONG",
                "screening_direction": "up",
                "close_prices": [300.0, 305.0, 310.0],
                "decision_source": "active_screener",
                "selection": {},
                "metadata": {},
            },
        ) as mocked_screen, patch("src.strategy.portfolio_strategy._reference_price", return_value=310.0), patch(
            "src.strategy.portfolio_strategy._evaluate_slot_direction",
            return_value=(
                "LONG",
                {"decision": {"decision": "LONG", "reason": "DeepSeek confirms the active candidate."}},
                {
                    "ai_prompt_timeframe": "1h",
                    "ai_prompt_candle_count": 3,
                    "ai_prompt_close_prices": [300.0, 305.0, 310.0],
                },
            ),
        ) as mocked_ai, patch(
            "src.strategy.portfolio_strategy.get_account_overview",
            return_value={"equity": 1000.0, "available_balance": 500.0},
        ), patch(
            "src.strategy.portfolio_strategy._place_direction_position",
            return_value={"success": True, "action": "opened_new_position", "position": {"symbol": "YMUSDT"}},
        ) as mocked_place, patch(
            "src.strategy.portfolio_strategy._sync_position_after_trade",
            return_value={"position": {"symbol": "YMUSDT"}, "stop_sync": {"success": True}},
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
                cycle_dir_factory=lambda: temp_dir,
                notification_callback=None,
            )

        self.assertTrue(result["success"])
        self.assertEqual(active_symbol, "YMUSDT")
        self.assertEqual(updated_state["symbol"], "YMUSDT")
        self.assertEqual(updated_state["previous_active_symbol"], "ESUSDT")
        self.assertEqual(updated_state["last_ai_decision"], "LONG")
        mocked_ai.assert_called_once()
        mocked_place.assert_called_once()
        self.assertEqual(mocked_place.call_args.kwargs["decision"], "LONG")
        excluded_symbols = set(mocked_screen.call_args.kwargs["excluded_symbols"])
        self.assertIn("ESUSDT", excluded_symbols)
        self.assertIn("NQUSDT", excluded_symbols)

    def test_active_new_candidate_uses_deepseek_direction_after_screening(self):
        slot = _active_slot()
        slot_state = {
            "slot_id": "active_1",
            "kind": "active",
            "symbol": None,
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.strategy.portfolio_strategy._screen_active_candidate",
            return_value={
                "symbol": "NQUSDT",
                "screening_decision": "SHORT",
                "screening_direction": "down",
                "close_prices": [210.0, 205.0, 200.0],
                "decision_source": "active_screener",
                "selection": {},
                "metadata": {},
            },
        ), patch("src.strategy.portfolio_strategy._reference_price", return_value=200.0), patch(
            "src.strategy.portfolio_strategy._evaluate_slot_direction",
            return_value=(
                "LONG",
                {"decision": {"decision": "LONG", "reason": "DeepSeek sees stronger upside continuation."}},
                {
                    "ai_prompt_timeframe": "1h",
                    "ai_prompt_candle_count": 3,
                    "ai_prompt_close_prices": [198.0, 199.0, 200.0],
                },
            ),
        ) as mocked_ai, patch(
            "src.strategy.portfolio_strategy.get_account_overview",
            return_value={"equity": 1000.0, "available_balance": 500.0},
        ), patch(
            "src.strategy.portfolio_strategy._place_direction_position",
            return_value={"success": True, "action": "opened_new_position", "position": {"symbol": "NQUSDT"}},
        ) as mocked_place, patch(
            "src.strategy.portfolio_strategy._sync_position_after_trade",
            return_value={"position": {"symbol": "NQUSDT"}, "stop_sync": {"success": True}},
        ):
            notifications = []
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
                cycle_dir_factory=lambda: temp_dir,
                notification_callback=lambda event_name, payload: notifications.append((event_name, payload)),
            )

        self.assertTrue(result["success"])
        self.assertTrue(result["ai_triggered"])
        self.assertTrue(result["screening_triggered"])
        self.assertEqual(result["screening_decision"], "SHORT")
        self.assertEqual(result["ai_decision"], "LONG")
        self.assertEqual(updated_state["symbol"], "NQUSDT")
        self.assertEqual(updated_state["last_ai_decision"], "LONG")
        self.assertEqual(active_symbol, "NQUSDT")
        mocked_ai.assert_called_once()
        mocked_place.assert_called_once()
        self.assertEqual(mocked_place.call_args.kwargs["decision"], "LONG")
        self.assertEqual([event_name for event_name, _ in notifications], ["active_screening_after"])
        active_payload = notifications[0][1]
        self.assertEqual(active_payload["symbol"], "NQUSDT")
        self.assertEqual(active_payload["screening_decision"], "SHORT")
        self.assertEqual(active_payload["ai_decision"], "LONG")
        self.assertEqual(active_payload["decision_source"], "deepseek_trader")
        self.assertEqual(active_payload["close_prices"], [198.0, 199.0, 200.0])
        self.assertEqual(active_payload["execution"]["action"], "opened_new_position")

    def test_active_due_rank_review_evaluates_current_top_symbol_with_deepseek(self):
        slot = _active_slot()
        slot_state = {
            "slot_id": "active_1",
            "kind": "active",
            "symbol": "ESUSDT",
            "active_screening_mode": "tradfi",
            "last_ai_decision": "LONG",
            "entered_at": "1970-01-01T00:00:00Z",
            "last_active_rank_checked_at": "1970-01-01T00:00:00Z",
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.strategy.portfolio_strategy._reference_price",
            return_value=101.0,
        ), patch(
            "src.strategy.portfolio_strategy._screen_active_candidate",
            return_value={
                "symbol": "ESUSDT",
                "screening_decision": "LONG",
                "screening_direction": "up",
                "close_prices": _close_prices(start=90.0, step=0.1),
                "decision_source": "active_screener",
                "selection": {"symbol": "ESUSDT", "selected": {"symbol": "ESUSDT"}},
                "metadata": {},
                "_screener_output": {"selection": {"symbol": "ESUSDT"}},
            },
        ) as mocked_screen, patch(
            "src.strategy.portfolio_strategy._close_existing_position"
        ) as mocked_close, patch(
            "src.strategy.portfolio_strategy._place_direction_position"
        ) as mocked_place, patch(
            "src.strategy.portfolio_strategy._evaluate_slot_direction",
            return_value=(
                "LONG",
                {"decision": {"decision": "LONG", "reason": "DeepSeek keeps the current upside setup."}},
                {
                    "ai_prompt_timeframe": "1h",
                    "ai_prompt_candle_count": 168,
                    "ai_prompt_close_prices": _close_prices(start=90.0, step=0.1),
                },
            ),
        ) as mocked_ai, patch(
            "src.strategy.portfolio_strategy._rebalance_existing_position",
            return_value={
                "success": True,
                "action": "kept_position_size",
                "position": {"symbol": "ESUSDT", "direction": "long"},
                "stop_sync": {"success": True},
            },
        ) as mocked_rebalance:
            notifications = []
            result, updated_state, active_symbol = portfolio_strategy._run_active_slot(
                slot=slot,
                slot_state=slot_state,
                config=_config(),
                api_key="key",
                api_secret="secret",
                account_overview={"equity": 1000.0, "available_balance": 500.0},
                open_positions=[_long_position()],
                reserved_symbols=set(),
                as_of_ms=(24 * 60 * 60 * 1000) + 1000,
                cycle_dir_factory=lambda: temp_dir,
                notification_callback=lambda event_name, payload: notifications.append((event_name, payload)),
            )

        self.assertTrue(result["success"])
        self.assertTrue(result["ai_triggered"])
        self.assertTrue(result["screening_triggered"])
        self.assertEqual(result["trigger_reason"], "active_rank_review_due")
        self.assertEqual(result["action"], "kept_position_size")
        self.assertEqual(result["ai_decision"], "LONG")
        self.assertEqual(active_symbol, "ESUSDT")
        self.assertEqual(updated_state["symbol"], "ESUSDT")
        self.assertEqual(updated_state["last_ai_decision"], "LONG")
        self.assertEqual(updated_state["last_active_rank_checked_at"], "1970-01-02T00:00:01Z")
        mocked_screen.assert_called_once()
        self.assertNotIn("ESUSDT", mocked_screen.call_args.kwargs["excluded_symbols"])
        mocked_ai.assert_called_once()
        mocked_rebalance.assert_called_once()
        mocked_close.assert_not_called()
        mocked_place.assert_not_called()
        self.assertEqual([event_name for event_name, _ in notifications], ["active_screening_after"])

    def test_active_due_rank_review_feeds_trader_directions_into_rebalancer(self):
        slot = _active_slot()
        slot_state = {
            "slot_id": "active_1",
            "kind": "active",
            "symbol": "ESUSDT",
            "active_screening_mode": "tradfi",
            "last_ai_decision": "LONG",
            "entered_at": "1970-01-01T00:00:00Z",
            "last_active_rank_checked_at": "1970-01-01T00:00:00Z",
        }
        current_closes = _close_prices(start=90.0, step=0.1)
        candidate_closes = _close_prices(start=200.0, step=0.1)

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.strategy.portfolio_strategy._reference_price",
            side_effect=[101.0, 205.0],
        ), patch(
            "src.strategy.portfolio_strategy._screen_active_candidate",
            return_value={
                "symbol": "NQUSDT",
                "screening_decision": "SHORT",
                "screening_direction": "down",
                "close_prices": _close_prices(start=220.0, step=-0.1),
                "decision_source": "active_screener",
                "selection": {
                    "symbol": "NQUSDT",
                    "selected": {
                        "symbol": "NQUSDT",
                        "screening_decision": "SHORT",
                        "screening_direction": "down",
                        "r_squared": 0.92,
                    },
                },
                "metadata": {},
                "_screener_output": {"selection": {"symbol": "NQUSDT"}},
            },
        ), patch(
            "src.strategy.portfolio_strategy._evaluate_slot_direction",
            side_effect=[
                (
                    "SHORT",
                    {"decision": {"decision": "SHORT", "reason": "Current symbol lost upside quality."}},
                    {
                        "ai_prompt_timeframe": "1h",
                        "ai_prompt_candle_count": 168,
                        "ai_prompt_close_prices": current_closes,
                    },
                ),
                (
                    "LONG",
                    {"decision": {"decision": "LONG", "reason": "Candidate has better forward setup."}},
                    {
                        "ai_prompt_timeframe": "1h",
                        "ai_prompt_candle_count": 168,
                        "ai_prompt_close_prices": candidate_closes,
                    },
                ),
            ],
        ) as mocked_ai, patch(
            "src.strategy.portfolio_strategy._evaluate_active_rebalance_or_none",
            return_value=(
                "NQUSDT",
                {"decision": {"selected_symbol": "NQUSDT", "reason": "Candidate chosen."}},
                None,
            ),
        ) as mocked_rebalancer, patch(
            "src.strategy.portfolio_strategy._close_existing_position",
            return_value={"success": True, "action": "closed_position"},
        ) as mocked_close, patch(
            "src.strategy.portfolio_strategy.get_account_overview",
            return_value={"equity": 1000.0, "available_balance": 500.0},
        ), patch(
            "src.strategy.portfolio_strategy._place_direction_position",
            return_value={"success": True, "action": "opened_new_position", "position": {"symbol": "NQUSDT"}},
        ) as mocked_place, patch(
            "src.strategy.portfolio_strategy._sync_position_after_trade",
            return_value={"position": {"symbol": "NQUSDT"}, "stop_sync": {"success": True}},
        ):
            result, updated_state, active_symbol = portfolio_strategy._run_active_slot(
                slot=slot,
                slot_state=slot_state,
                config=_config(),
                api_key="key",
                api_secret="secret",
                account_overview={"equity": 1000.0, "available_balance": 500.0},
                open_positions=[_long_position()],
                reserved_symbols=set(),
                as_of_ms=(24 * 60 * 60 * 1000) + 1000,
                cycle_dir_factory=lambda: temp_dir,
                notification_callback=None,
            )

        self.assertTrue(result["success"])
        self.assertTrue(result["ai_triggered"])
        self.assertTrue(result["screening_triggered"])
        self.assertEqual(result["trigger_reason"], "active_rank_review_due")
        self.assertEqual(result["action"], "switched_active_position_by_rebalancer")
        self.assertEqual(result["ai_decision"], "LONG")
        self.assertEqual(result["selected_symbol"], "NQUSDT")
        self.assertEqual(active_symbol, "NQUSDT")
        self.assertEqual(updated_state["symbol"], "NQUSDT")
        self.assertEqual(updated_state["last_ai_decision"], "LONG")
        self.assertEqual(result["active_direction_reviews"]["current"]["decision"], "SHORT")
        self.assertEqual(result["active_direction_reviews"]["candidate"]["decision"], "LONG")
        self.assertEqual(mocked_ai.call_count, 2)
        mocked_rebalancer.assert_called_once()
        rebalancer_kwargs = mocked_rebalancer.call_args.kwargs
        self.assertEqual(rebalancer_kwargs["current_payload"]["decision"], "SHORT")
        self.assertEqual(rebalancer_kwargs["candidate_payload"]["decision"], "LONG")
        self.assertEqual(rebalancer_kwargs["current_payload"]["timeframes"]["1h"], current_closes[:-1] + [101.0])
        self.assertEqual(rebalancer_kwargs["candidate_payload"]["timeframes"]["1h"], candidate_closes[:-1] + [205.0])
        mocked_close.assert_called_once()
        mocked_place.assert_called_once()
        self.assertEqual(mocked_place.call_args.kwargs["decision"], "LONG")


if __name__ == "__main__":
    unittest.main()
