import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from src.strategy.scheduler import TradingScheduler
from src.strategy.portfolio_strategy import STATE_VERSION


class SchedulerTests(unittest.TestCase):
    def test_run_cycle_once_uses_portfolio_cycle_and_persists_state(self):
        state_update = {"version": STATE_VERSION, "slots": {"active_1": {"symbol": "ESUSDT"}}}
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
        self.assertEqual(scheduler.state["slots"]["active_1"]["symbol"], "ESUSDT")
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

    def test_close_price_chart_uses_prompt_candle_count(self):
        scheduler = TradingScheduler()
        close_prices = [100.0 + float(index) for index in range(168)]

        with patch("src.strategy.scheduler.build_close_price_line_chart_png", return_value=b"png") as mocked_chart, patch(
            "src.strategy.scheduler.send_telegram_photo", return_value=True
        ) as mocked_photo:
            sent = scheduler._emit_telegram_close_price_chart(
                {
                    "symbol": "BTCUSDT",
                    "decision": "LONG",
                    "ai_prompt_timeframe": "1h",
                    "ai_prompt_candle_count": 168,
                    "close_prices": close_prices,
                }
            )

        self.assertTrue(sent)
        mocked_chart.assert_called_once()
        self.assertEqual(len(mocked_chart.call_args.args[0]), 168)
        self.assertEqual(mocked_chart.call_args.kwargs["limit"], 168)
        self.assertIn("<b>Count:</b> 168", mocked_photo.call_args.kwargs["caption"])

    def test_active_screening_event_sends_message_and_chart(self):
        scheduler = TradingScheduler()
        payload = {
            "slot_id": "active_1",
            "slot_label": "active1",
            "symbol": "NQUSDT",
            "current_price": 55.0,
            "trigger_reason": "active_candidate_selected",
            "screening_decision": "LONG",
            "decision": "LONG",
            "ai_decision": "LONG",
            "decision_source": "deepseek_trader",
            "ai_analysis": {
                "decision": {
                    "decision": "LONG",
                    "reason": "DeepSeek confirms the active setup.",
                }
            },
            "action": "opened_new_position",
            "success": True,
            "execution": {"action": "opened_new_position", "side": "Buy", "qty": "3"},
            "position": {"symbol": "NQUSDT", "direction": "long", "size": 3, "entry_price": 55.0},
            "screener": {
                "metadata": {"screening_mode": "tradfi"},
                "selection": {
                    "ranking_metric": "close_range_volatility",
                    "selected": {
                        "close_range_volatility_pct": 6.2,
                        "net_return_pct": 6.1,
                    },
                },
            },
            "ai_prompt_timeframe": "1h",
            "ai_prompt_candle_count": 3,
            "close_prices": [52.0, 54.0, 55.0],
        }

        with patch("src.strategy.scheduler.send_telegram_message", return_value=True) as mocked_text, patch(
            "src.strategy.scheduler.build_close_price_line_chart_png", return_value=b"png"
        ) as mocked_chart, patch("src.strategy.scheduler.send_telegram_photo", return_value=True) as mocked_photo:
            status = scheduler._notify_telegram_event("active_screening_after", payload)

        mocked_text.assert_called_once()
        self.assertIn("Active Screening Decision", mocked_text.call_args.args[0])
        self.assertIn("NQUSDT", mocked_text.call_args.args[0])
        mocked_chart.assert_called_once()
        self.assertEqual(mocked_chart.call_args.args[0], [52.0, 54.0, 55.0])
        mocked_photo.assert_called_once()
        self.assertIn("Active Screening Close Prices Line Chart", mocked_photo.call_args.kwargs["caption"])
        self.assertEqual(status["event"], "active_screening_after")
        self.assertTrue(status["sent"])
        self.assertTrue(status["chart_sent"])
        self.assertEqual(status["symbol"], "NQUSDT")


if __name__ == "__main__":
    unittest.main()
