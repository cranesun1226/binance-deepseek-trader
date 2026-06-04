import unittest
from unittest.mock import patch

from src.strategy import portfolio_strategy


class PortfolioHelperTests(unittest.TestCase):
    def test_default_config_builds_six_slots_in_priority_order(self):
        config = {
            "passive_symbols": ["CLUSDT", "XAUUSDT", "QQQUSDT", "BTCUSDT"],
            "active_targets": [4.0, 4.0],
        }

        slots = portfolio_strategy._build_portfolio_slots(config)

        self.assertEqual([slot.slot_id for slot in slots], [
            "passive_cl",
            "passive_xau",
            "passive_qqq",
            "passive_btc",
            "active_1",
            "active_2",
        ])
        self.assertEqual([slot.symbol for slot in slots[:4]], ["CLUSDT", "XAUUSDT", "QQQUSDT", "BTCUSDT"])
        self.assertEqual([slot.target_margin_ratio for slot in slots], [0.125, 0.125, 0.125, 0.125, 0.25, 0.25])
        self.assertEqual(slots[4].active_screening_mode, "standard")
        self.assertEqual(slots[5].active_screening_mode, "tradfi")

    def test_deepseek_reasoning_effort_normalization_matches_api_values(self):
        self.assertEqual(portfolio_strategy._normalize_reasoning_effort("xhigh"), "max")
        self.assertEqual(portfolio_strategy._normalize_reasoning_effort("max"), "max")
        self.assertEqual(portfolio_strategy._normalize_reasoning_effort("medium"), "high")
        self.assertEqual(portfolio_strategy._normalize_reasoning_effort(""), "max")

    def test_target_notional_uses_capital_usage_ratio_and_leverage(self):
        slot = portfolio_strategy.PortfolioSlot(
            slot_id="active_1",
            label="active1",
            kind="active",
            target_margin_ratio=0.25,
            active_target_abs_change_pct=4.0,
        )

        target = portfolio_strategy._target_notional_usdt(
            account_equity=1000.0,
            slot=slot,
            capital_usage_ratio=0.99,
            leverage=1,
        )

        self.assertEqual(target, 247.5)

    def test_fixed_stop_loss_is_entry_price_four_percent(self):
        self.assertEqual(
            portfolio_strategy._resolve_fixed_stop_loss_price(
                direction="long",
                entry_price=100.0,
                stop_loss_pct=0.04,
            ),
            96.0,
        )
        self.assertEqual(
            portfolio_strategy._resolve_fixed_stop_loss_price(
                direction="short",
                entry_price=100.0,
                stop_loss_pct=0.04,
            ),
            104.0,
        )

    def test_trigger_is_per_slot_last_llm_anchor(self):
        slot_state = {
            "last_ai_trigger_price": 100.0,
            "last_ai_decision": "LONG",
            "next_trigger_down": 99.0,
            "next_trigger_up": 101.0,
        }

        waiting = portfolio_strategy._determine_ai_trigger(
            has_position=True,
            current_price=100.5,
            slot_state=slot_state,
            trigger_pct_usdt=1.0,
        )
        triggered = portfolio_strategy._determine_ai_trigger(
            has_position=True,
            current_price=101.01,
            slot_state=slot_state,
            trigger_pct_usdt=1.0,
        )

        self.assertFalse(waiting["should_trigger"])
        self.assertTrue(triggered["should_trigger"])
        self.assertEqual(triggered["reason"], "price_distance_reached")

    def test_missing_ai_decision_for_existing_position_forces_llm_trigger(self):
        slot_state = {
            "last_ai_trigger_price": 100.0,
            "last_ai_decision": None,
            "next_trigger_down": 99.0,
            "next_trigger_up": 101.0,
        }

        triggered = portfolio_strategy._determine_ai_trigger(
            has_position=True,
            current_price=100.5,
            slot_state=slot_state,
            trigger_pct_usdt=1.0,
        )

        self.assertTrue(triggered["should_trigger"])
        self.assertEqual(triggered["reason"], "missing_ai_decision")

    def test_rebalance_rejects_invalid_decision_without_inferring_direction(self):
        result = portfolio_strategy._rebalance_existing_position(
            api_key="key",
            api_secret="secret",
            symbol="ETHUSDT",
            position={
                "symbol": "ETHUSDT",
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
            rebalance_threshold_pct=0.03,
            stop_loss_pct=0.04,
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
                rebalance_threshold_pct=0.03,
                stop_loss_pct=0.04,
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
                rebalance_threshold_pct=0.03,
                stop_loss_pct=0.04,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "kept_position_size")
        self.assertEqual(result["skipped_rebalance_reason"], "insufficient_available_balance")
        mocked_place.assert_not_called()

    def test_reverse_existing_position_closes_then_opens_opposite_immediately(self):
        with patch(
            "src.strategy.portfolio_strategy._close_existing_position",
            return_value={"success": True, "action": "closed_position", "symbol": "ETHUSDT"},
        ) as mocked_close, patch(
            "src.strategy.portfolio_strategy._place_direction_position",
            return_value={"success": True, "action": "opened_new_position"},
        ) as mocked_place, patch(
            "src.strategy.portfolio_strategy._sync_position_after_trade",
            return_value={"position": {"symbol": "ETHUSDT", "direction": "short"}, "stop_sync": {"success": True}},
        ) as mocked_sync:
            result = portfolio_strategy._reverse_existing_position(
                api_key="key",
                api_secret="secret",
                symbol="ETHUSDT",
                position={
                    "symbol": "ETHUSDT",
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
                stop_loss_pct=0.04,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "reversed_position")
        mocked_close.assert_called_once()
        mocked_place.assert_called_once()
        self.assertEqual(mocked_place.call_args.kwargs["symbol"], "ETHUSDT")
        self.assertEqual(mocked_place.call_args.kwargs["decision"], "SHORT")
        mocked_sync.assert_called_once()
        self.assertEqual(mocked_sync.call_args.kwargs["expected_decision"], "SHORT")

    def test_trigger_levels_keep_precision_for_low_priced_symbols(self):
        levels = portfolio_strategy._build_trigger_levels(0.093455, 1.0)

        self.assertEqual(levels["trigger_price"], 0.093455)
        self.assertLess(levels["next_trigger_down"], levels["trigger_price"])
        self.assertGreater(levels["next_trigger_up"], levels["trigger_price"])

    def test_duplicate_active_state_symbol_is_cleared_for_later_slot(self):
        config = {
            "passive_symbols": ["CLUSDT", "XAUUSDT", "QQQUSDT", "BTCUSDT"],
            "active_targets": [4.0, 4.0],
        }
        slots = portfolio_strategy._build_portfolio_slots(config)
        state = portfolio_strategy._normalize_portfolio_state(
            {
                "version": portfolio_strategy.STATE_VERSION,
                "slots": {
                    "active_1": {"symbol": "ETHUSDT", "last_ai_decision": "LONG"},
                    "active_2": {"symbol": "ETHUSDT", "last_ai_decision": "SHORT"},
                },
            },
            slots,
        )

        self.assertEqual(state["slots"]["active_1"]["symbol"], "ETHUSDT")
        self.assertIsNone(state["slots"]["active_2"]["symbol"])


if __name__ == "__main__":
    unittest.main()
