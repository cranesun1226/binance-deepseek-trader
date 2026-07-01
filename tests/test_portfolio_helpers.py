import unittest
from unittest.mock import patch

from src.strategy import portfolio_strategy


class PortfolioHelperTests(unittest.TestCase):
    def test_default_config_builds_two_active_slots_in_priority_order(self):
        slots = portfolio_strategy._build_portfolio_slots({})

        self.assertEqual([slot.slot_id for slot in slots], ["active_1", "active_2"])
        self.assertEqual([slot.kind for slot in slots], ["active"] * 2)
        self.assertEqual([slot.target_margin_ratio for slot in slots], [0.5] * 2)
        self.assertEqual([slot.active_screening_mode for slot in slots], ["tradfi", "all"])

    def test_zai_reasoning_effort_normalization_matches_api_values(self):
        self.assertEqual(portfolio_strategy._normalize_reasoning_effort("xhigh"), "max")
        self.assertEqual(portfolio_strategy._normalize_reasoning_effort("max"), "max")
        self.assertEqual(portfolio_strategy._normalize_reasoning_effort("medium"), "high")
        self.assertEqual(portfolio_strategy._normalize_reasoning_effort(""), "max")

    def test_target_notional_uses_capital_usage_ratio_and_leverage(self):
        slot = portfolio_strategy.PortfolioSlot(
            slot_id="active_1",
            label="active1",
            kind="active",
            target_margin_ratio=0.5,
        )

        target = portfolio_strategy._target_notional_usdt(
            account_equity=1000.0,
            slot=slot,
            capital_usage_ratio=0.99,
            leverage=2,
        )

        self.assertEqual(target, 990.0)

    def test_slot_leverage_defaults_active_to_1x(self):
        active = portfolio_strategy.PortfolioSlot(
            slot_id="active_1",
            label="active1",
            kind="active",
            target_margin_ratio=0.5,
        )

        self.assertEqual(portfolio_strategy._leverage_for_slot(active, {}), 1)

    def test_active_rescreen_interval_defaults_to_18_hours(self):
        self.assertEqual(
            portfolio_strategy._active_rescreen_interval_ms({}),
            18 * 60 * 60 * 1000,
        )

    def test_ensure_symbol_leverage_fails_closed_on_mismatch(self):
        with patch("src.strategy.portfolio_strategy.set_leverage", return_value=1):
            result = portfolio_strategy._ensure_symbol_leverage(
                api_key="key",
                api_secret="secret",
                symbol="CLUSDT",
                leverage=2,
                current_position={
                    "symbol": "CLUSDT",
                    "positionAmt": "1",
                    "side": "Buy",
                    "entryPrice": "100",
                    "markPrice": "100",
                    "leverage": "1",
                },
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["action"], "set_leverage_failed")
        self.assertEqual(result["requested_leverage"], 2)
        self.assertEqual(result["actual_leverage"], 1)

    def test_fixed_stop_loss_is_entry_price_five_percent(self):
        self.assertEqual(
            portfolio_strategy._resolve_fixed_stop_loss_price(
                direction="long",
                entry_price=100.0,
                stop_loss_pct=0.05,
            ),
            95.0,
        )
        self.assertEqual(
            portfolio_strategy._resolve_fixed_stop_loss_price(
                direction="short",
                entry_price=100.0,
                stop_loss_pct=0.05,
            ),
            105.0,
        )

    def test_ai_trigger_info_has_no_price_distance_window(self):
        waiting = portfolio_strategy._build_ai_trigger_info(
            reason="holding_until_rebalance",
            current_price=100.5,
            should_trigger=False,
        )
        triggered = portfolio_strategy._build_ai_trigger_info(
            reason="active_candidate_selected",
            current_price=101.01,
        )

        self.assertFalse(waiting["should_trigger"])
        self.assertEqual(waiting["reason"], "holding_until_rebalance")
        self.assertEqual(waiting["trigger_price"], 100.5)
        self.assertNotIn("next_trigger_down", waiting)
        self.assertTrue(triggered["should_trigger"])
        self.assertEqual(triggered["reason"], "active_candidate_selected")
        self.assertEqual(triggered["trigger_price"], 101.01)

    def test_normalized_state_migrates_old_ai_trigger_timestamp_only(self):
        slot = portfolio_strategy.PortfolioSlot(
            slot_id="active_1",
            label="active1",
            kind="active",
            target_margin_ratio=0.5,
        )

        state = portfolio_strategy._normalize_slot_state(
            slot,
            {
                "symbol": "ESUSDT",
                "last_ai_triggered_at": "2026-01-01T00:00:00Z",
                "last_ai_trigger_price": 100.0,
                "next_trigger_down": 99.0,
                "next_trigger_up": 101.0,
                "last_ai_decision": "LONG",
            },
        )

        self.assertEqual(state["last_ai_decision_at"], "2026-01-01T00:00:00Z")
        self.assertEqual(state["last_ai_decision"], "LONG")
        self.assertNotIn("last_ai_trigger_price", state)
        self.assertNotIn("next_trigger_down", state)
        self.assertNotIn("next_trigger_up", state)

    def test_rebalance_rejects_invalid_decision_without_inferring_direction(self):
        result = portfolio_strategy._rebalance_existing_position(
            api_key="key",
            api_secret="secret",
            symbol="XAUUSDT",
            position={
                "symbol": "XAUUSDT",
                "positionAmt": "2",
                "side": "Buy",
                "entryPrice": "100",
                "markPrice": "100",
            },
            decision="",
            target_notional_usdt=200.0,
            reference_price=100.0,
            leverage=1,
            available_notional_cap=100.0,
            rebalance_threshold_pct=0.02,
            stop_loss_pct=0.05,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["action"], "invalid_ai_decision")

    def test_rebalance_keeps_position_when_tiny_increase_is_below_min_qty(self):
        with patch(
            "src.strategy.portfolio_strategy._sync_fixed_stop_loss",
            return_value={"success": True, "changed": False},
        ), patch(
            "src.strategy.portfolio_strategy._place_direction_position",
            return_value={
                "success": False,
                "action": "target_qty_below_min",
                "order_plan": {"qty": None, "meets_min_notional": True},
            },
        ) as mocked_place:
            result = portfolio_strategy._rebalance_existing_position(
                api_key="key",
                api_secret="secret",
                symbol="BTCUSDT",
                position={
                    "symbol": "BTCUSDT",
                    "positionAmt": "-0.004",
                    "side": "Sell",
                    "entryPrice": "65000",
                    "markPrice": "65000",
                    "positionValue": "260",
                },
                decision="SHORT",
                target_notional_usdt=307.0,
                reference_price=65000.0,
                leverage=1,
                available_notional_cap=47.0,
                rebalance_threshold_pct=0.02,
                stop_loss_pct=0.05,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "kept_position_size")
        self.assertEqual(result["skipped_rebalance_reason"], "target_qty_below_min")
        self.assertEqual(result["rebalance_order_plan"], {"qty": None, "meets_min_notional": True})
        mocked_place.assert_called_once()

    def test_rebalance_keeps_position_when_no_available_cap_for_increase(self):
        with patch(
            "src.strategy.portfolio_strategy._sync_fixed_stop_loss",
            return_value={"success": True, "changed": False},
        ), patch("src.strategy.portfolio_strategy._place_direction_position") as mocked_place:
            result = portfolio_strategy._rebalance_existing_position(
                api_key="key",
                api_secret="secret",
                symbol="BTCUSDT",
                position={
                    "symbol": "BTCUSDT",
                    "positionAmt": "-0.004",
                    "side": "Sell",
                    "entryPrice": "65000",
                    "markPrice": "65000",
                    "positionValue": "260",
                },
                decision="SHORT",
                target_notional_usdt=307.0,
                reference_price=65000.0,
                leverage=1,
                available_notional_cap=0.0,
                rebalance_threshold_pct=0.02,
                stop_loss_pct=0.05,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "kept_position_size")
        self.assertEqual(result["skipped_rebalance_reason"], "insufficient_available_balance")
        mocked_place.assert_not_called()

    def test_reverse_existing_position_closes_then_opens_opposite_immediately(self):
        with patch(
            "src.strategy.portfolio_strategy._close_existing_position",
            return_value={"success": True, "action": "closed_position", "symbol": "XAUUSDT"},
        ) as mocked_close, patch(
            "src.strategy.portfolio_strategy._place_direction_position",
            return_value={"success": True, "action": "opened_new_position"},
        ) as mocked_place, patch(
            "src.strategy.portfolio_strategy._sync_position_after_trade",
            return_value={"position": {"symbol": "XAUUSDT", "direction": "short"}, "stop_sync": {"success": True}},
        ) as mocked_sync:
            result = portfolio_strategy._reverse_existing_position(
                api_key="key",
                api_secret="secret",
                symbol="XAUUSDT",
                position={
                    "symbol": "XAUUSDT",
                    "positionAmt": "2",
                    "side": "Buy",
                    "entryPrice": "100",
                    "markPrice": "100",
                },
                decision="SHORT",
                target_notional_usdt=200.0,
                reference_price=100.0,
                leverage=1,
                available_notional_cap=200.0,
                stop_loss_pct=0.05,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "reversed_position")
        mocked_close.assert_called_once()
        mocked_place.assert_called_once()
        self.assertEqual(mocked_place.call_args.kwargs["symbol"], "XAUUSDT")
        self.assertEqual(mocked_place.call_args.kwargs["decision"], "SHORT")
        mocked_sync.assert_called_once()
        self.assertEqual(mocked_sync.call_args.kwargs["expected_decision"], "SHORT")

    def test_ai_trigger_info_keeps_precision_for_low_priced_symbols(self):
        info = portfolio_strategy._build_ai_trigger_info(reason="active_candidate_selected", current_price=0.093455)

        self.assertEqual(info["trigger_price"], 0.093455)
        self.assertNotIn("next_trigger_down", info)
        self.assertNotIn("next_trigger_up", info)

    def test_removed_slots_are_dropped_from_normalized_state(self):
        slots = portfolio_strategy._build_portfolio_slots({})
        state = portfolio_strategy._normalize_portfolio_state(
            {
                "version": portfolio_strategy.STATE_VERSION,
                "slots": {
                    "retired_slot": {"symbol": "CLUSDT", "last_ai_decision": "LONG"},
                    "active_1": {"symbol": "ESUSDT", "last_ai_decision": "LONG"},
                    "active_2": {"symbol": "ESUSDT", "last_ai_decision": "SHORT"},
                    "active_3": {"symbol": "GCUSDT"},
                    "active_4": {"symbol": "BTCUSDT"},
                },
            },
            slots,
        )

        self.assertEqual(state["slots"]["active_1"]["symbol"], "ESUSDT")
        self.assertEqual(state["slots"]["active_2"]["symbol"], None)
        self.assertEqual(set(state["slots"]), {"active_1", "active_2"})
        self.assertNotIn("retired_slot", state["slots"])
        self.assertNotIn("active_3", state["slots"])
        self.assertNotIn("active_4", state["slots"])

    def test_active_recent_symbols_are_grouped_by_screening_universe(self):
        slots = portfolio_strategy._build_portfolio_slots({})
        state = portfolio_strategy._normalize_portfolio_state(
            {
                "version": portfolio_strategy.STATE_VERSION,
                "slots": {
                    "active_1": {"symbol": "ESUSDT", "previous_active_symbol": "NQUSDT"},
                    "active_2": {"symbol": "GCUSDT", "previous_active_symbol": "YMUSDT"},
                    "active_3": {"symbol": "XAUUSDT"},
                    "active_4": {"symbol": "BTCUSDT", "previous_active_symbol": "ETHUSDT"},
                },
            },
            slots,
        )

        grouped = portfolio_strategy._active_recent_symbols_by_mode(slots, state)

        self.assertEqual(grouped["tradfi"], {"ESUSDT", "NQUSDT"})
        self.assertEqual(grouped["all"], {"GCUSDT", "YMUSDT"})
        self.assertEqual(
            portfolio_strategy._recent_symbols_for_screening_mode(grouped, "all"),
            {"ESUSDT", "NQUSDT", "GCUSDT", "YMUSDT"},
        )
        self.assertEqual(
            portfolio_strategy._recent_symbols_for_screening_mode(grouped, "tradfi"),
            {"ESUSDT", "NQUSDT", "GCUSDT", "YMUSDT"},
        )

    def test_runtime_symbol_bans_are_normalized_and_excluded_from_active_screening(self):
        slots = portfolio_strategy._build_portfolio_slots({})
        state = portfolio_strategy._normalize_portfolio_state(
            {
                "version": portfolio_strategy.STATE_VERSION,
                "slots": {},
                "symbol_bans": ["skhynixusdt"],
            },
            slots,
        )

        self.assertIn("SKHYNIXUSDT", state["symbol_bans"])
        excluded = portfolio_strategy._active_exclusions(
            config={},
            open_positions=[],
            reserved_symbols=set(),
            banned_symbols=portfolio_strategy._runtime_banned_symbols(state),
        )
        self.assertEqual(excluded, ["SKHYNIXUSDT"])

    def test_record_symbol_ban_updates_existing_scheduler_state_entry(self):
        state = {
            "version": portfolio_strategy.STATE_VERSION,
            "slots": {},
            "symbol_bans": {
                "SKHYNIXUSDT": {
                    "symbol": "SKHYNIXUSDT",
                    "reason": "binance_region_restricted",
                    "first_seen_at": "1970-01-01T00:00:01Z",
                    "last_seen_at": "1970-01-01T00:00:01Z",
                    "event_count": 2,
                }
            },
        }

        updated = portfolio_strategy._record_symbol_ban(
            state,
            {
                "symbol": "SKHYNIXUSDT",
                "reason": "binance_region_restricted",
                "source": "entry_order",
                "error_code": -4412,
                "error_message": "not available in your region",
            },
            as_of_ms=2000,
        )

        ban = updated["symbol_bans"]["SKHYNIXUSDT"]
        self.assertEqual(ban["event_count"], 3)
        self.assertEqual(ban["first_seen_at"], "1970-01-01T00:00:01Z")
        self.assertEqual(ban["last_seen_at"], "1970-01-01T00:00:02Z")
        self.assertEqual(ban["error_code"], -4412)

    def test_stop_loss_malformed_trigger_records_symbol_ban(self):
        with patch(
            "src.strategy.portfolio_strategy.sync_existing_position_stop_loss",
            return_value={
                "success": False,
                "changed": False,
                "reason": "place_stop_loss_failed",
                "error_code": -1102,
                "error_message": "Mandatory parameter 'triggerprice' was not sent, was empty/null, or malformed.",
            },
        ):
            result = portfolio_strategy._sync_fixed_stop_loss(
                api_key="key",
                api_secret="secret",
                symbol="SPELLUSDT",
                position={
                    "symbol": "SPELLUSDT",
                    "positionAmt": "-1000",
                    "side": "Sell",
                    "entryPrice": "0.000095",
                    "markPrice": "0.000094",
                },
                stop_loss_pct=0.05,
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["symbol_ban"]["symbol"], "SPELLUSDT")
        self.assertEqual(result["symbol_ban"]["reason"], "binance_stop_loss_rejected")
        self.assertEqual(result["symbol_ban"]["source"], "stop_loss_sync")

    def test_ai_auth_failure_opens_circuit_breaker_with_backoff(self):
        updated = portfolio_strategy._record_ai_circuit_breaker_from_slot_result(
            {"version": portfolio_strategy.STATE_VERSION, "slots": {}},
            {
                "slot_id": "active_2",
                "error": 'ZAI authentication failed: Error code: 401 {"error":{"code":"1000","message":"Authentication Failed"}}',
            },
            as_of_ms=1000,
        )

        breaker = updated["ai_circuit_breaker"]
        self.assertEqual(breaker["reason"], "zai_authentication_failed")
        self.assertEqual(breaker["event_count"], 1)
        self.assertEqual(breaker["retry_after_at"], "1970-01-01T00:15:01Z")
        self.assertTrue(portfolio_strategy._ai_circuit_breaker_is_open(breaker, as_of_ms=2000))
        self.assertFalse(portfolio_strategy._ai_circuit_breaker_is_open(breaker, as_of_ms=901001))

    def test_active_screening_mode_change_is_detected_from_persisted_state(self):
        slots = portfolio_strategy._build_portfolio_slots({})
        state = portfolio_strategy._normalize_portfolio_state(
            {
                "version": portfolio_strategy.STATE_VERSION,
                "slots": {
                    "active_2": {
                        "symbol": "BTCUSDT",
                        "active_screening_mode": "crypto",
                        "last_ai_decision": "LONG",
                    },
                },
            },
            slots,
        )

        active2_state = state["slots"]["active_2"]
        self.assertEqual(active2_state["active_screening_mode"], "all")
        self.assertTrue(active2_state["active_screening_mode_changed"])
        self.assertEqual(active2_state["previous_active_screening_mode"], "crypto")


if __name__ == "__main__":
    unittest.main()
