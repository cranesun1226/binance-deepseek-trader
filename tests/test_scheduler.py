import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from src.strategy.scheduler import TradingScheduler
from src.strategy.portfolio_strategy import STATE_VERSION


class SchedulerTests(unittest.TestCase):
    def test_run_cycle_once_uses_portfolio_cycle_and_persists_state(self):
        state_update = {"version": STATE_VERSION, "slots": {"active_1": {"symbol": "ETHUSDT"}}}
        with patch.object(TradingScheduler, "load_state", return_value={"version": STATE_VERSION, "slots": {}}), patch.object(
            TradingScheduler, "save_state"
        ) as mocked_save, patch(
            "src.strategy.scheduler.run_portfolio_cycle",
            return_value={
                "success": True,
                "action": "portfolio_cycle_completed",
                "ai_triggered": False,
                "slot_results": [],
                "state_update": state_update,
            },
        ) as mocked_run:
            scheduler = TradingScheduler()
            result = scheduler.run_cycle_once(datetime(2026, 6, 3, tzinfo=timezone.utc))

        self.assertTrue(result["success"])
        self.assertEqual(scheduler.state["slots"]["active_1"]["symbol"], "ETHUSDT")
        mocked_run.assert_called_once()
        mocked_save.assert_called_once()

    def test_ai_decision_message_uses_structured_decision_reason(self):
        scheduler = TradingScheduler()
        message = scheduler._build_ai_cycle_after_message(
            {
                "slot_id": "passive_1",
                "symbol": "BTCUSDT",
                "decision": "LONG",
                "position": None,
                "analysis": {
                    "decision": {
                        "decision": "LONG",
                        "reason": "The full close series holds higher lows and recent closes sustain above the recovery base.",
                    },
                    "reasoning": "raw internal reasoning should stay hidden",
                },
            }
        )

        self.assertIn("Decision Reason", message)
        self.assertIn("The full close series holds higher lows", message)
        self.assertNotIn("DeepSeek Reasoning", message)
        self.assertNotIn("raw internal reasoning", message)


if __name__ == "__main__":
    unittest.main()
