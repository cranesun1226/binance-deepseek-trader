import unittest
from unittest.mock import Mock, patch

from src.ai import zai_rebalancer


def _candidate(symbol, decision, start):
    prices = [start + float(index) for index in range(3)]
    return {
        "symbol": symbol,
        "decision": decision,
        "reference_price": prices[-1],
        "timeframes": {"1h": prices},
    }


class ZAIRebalancerTests(unittest.TestCase):
    def test_rebalance_payload_uses_equal_priority_candidate_shape(self):
        payload = zai_rebalancer._build_rebalance_input_payload(
            candidates=[
                {
                    **_candidate("NQUSDT", "SHORT", 200.0),
                    "metrics": {"r_squared": 0.99, "trend_score": 1.0},
                },
                {
                    **_candidate("ESUSDT", "LONG", 100.0),
                    "metrics": {"r_squared": 0.98, "trend_score": 0.9},
                },
            ],
            timeframe="1h",
            candle_count=3,
        )

        self.assertEqual(
            [candidate["symbol"] for candidate in payload["candidates"]],
            ["ESUSDT", "NQUSDT"],
        )
        for candidate in payload["candidates"]:
            self.assertEqual(set(candidate.keys()), {"symbol", "reference_price", "timeframes"})
            self.assertNotIn("current", candidate)
            self.assertNotIn("existing", candidate)
            self.assertNotIn("replacement", candidate)
            self.assertNotIn("metrics", candidate)
            self.assertNotIn("decision", candidate)

        prompt = zai_rebalancer._format_rebalance_prompt(payload)
        lowered_prompt = prompt.lower()
        self.assertEqual(zai_rebalancer.REBALANCE_REASON_MAX_CHARS, 300)
        self.assertIn("300 characters or fewer", prompt)
        self.assertIn("compare both candidates directly", lowered_prompt)
        self.assertIn("selected symbol is more rational", lowered_prompt)
        self.assertNotIn("current", lowered_prompt)
        self.assertNotIn("existing", lowered_prompt)
        self.assertNotIn("replacement", lowered_prompt)
        self.assertNotIn("metrics", lowered_prompt)
        self.assertNotIn("r_squared", lowered_prompt)
        self.assertNotIn("trend_score", lowered_prompt)
        self.assertNotIn("decision", lowered_prompt)
        self.assertNotIn("fixed-direction", lowered_prompt)
        self.assertNotIn("long/short", lowered_prompt)

    def test_structured_rebalance_call_parses_candidate_symbol_selection(self):
        response_payload = {
            "id": "rebalance-id",
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"selected_symbol":"ESUSDT","reason":"ES holds a cleaner higher-low structure while '
                            'the alternative has weaker continuation evidence."}'
                        ),
                        "reasoning_content": "private reasoning",
                    }
                }
            ],
            "usage": {"prompt_tokens": 90, "completion_tokens": 20, "total_tokens": 110},
        }
        client = Mock()
        client.chat.completions.create.return_value = response_payload

        with patch("src.ai.zai_rebalancer.get_zai_api_key", return_value="key"), patch(
            "src.ai.zai_rebalancer.ZaiClient", return_value=client
        ) as mocked_client:
            result = zai_rebalancer._call_zai_rebalance_selection(
                prompt="prompt",
                reasoning_effort="max",
                allowed_symbols=["ESUSDT", "NQUSDT"],
                max_tokens=123,
                timeout_seconds=45.0,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.decision.selected_symbol, "ESUSDT")
        self.assertEqual(result.reasoning, "private reasoning")
        mocked_client.assert_called_once_with(api_key="key")
        payload = client.chat.completions.create.call_args.kwargs
        self.assertEqual(payload["model"], "glm-5.2")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["reasoning_effort"], "max")
        self.assertEqual(payload["max_tokens"], 123)
        self.assertIn("selected_symbol", payload["messages"][0]["content"])

    def test_rebalance_response_rejects_symbols_outside_supplied_candidates(self):
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": '{"selected_symbol":"BTCUSDT","reason":"This symbol was not supplied."}'
                    }
                }
            ],
            "usage": {},
        }
        client = Mock()
        client.chat.completions.create.return_value = response_payload

        with patch("src.ai.zai_rebalancer.get_zai_api_key", return_value="key"), patch(
            "src.ai.zai_rebalancer.ZaiClient", return_value=client
        ), patch("src.ai.zai_rebalancer.time.sleep") as mocked_sleep, patch(
            "src.ai.zai_rebalancer.logger"
        ):
            result = zai_rebalancer._call_zai_rebalance_selection(
                prompt="prompt",
                reasoning_effort="high",
                allowed_symbols=["ESUSDT", "NQUSDT"],
            )

        self.assertIsNone(result)
        self.assertEqual(client.chat.completions.create.call_count, zai_rebalancer.ZAI_GENERATE_MAX_RETRIES)
        self.assertEqual(mocked_sleep.call_count, zai_rebalancer.ZAI_GENERATE_MAX_RETRIES - 1)

    def test_evaluate_rebalance_fails_closed_on_invalid_payload(self):
        with patch("src.ai.zai_rebalancer._call_zai_rebalance_selection") as mocked_call, patch(
            "src.ai.zai_rebalancer.logger"
        ):
            result = zai_rebalancer.evaluate_active_rebalance_symbol(
                cycle_dir="/tmp/zai-rebalance-test",
                candidates=[_candidate("ESUSDT", "LONG", 100.0)],
                timeframe="1h",
                candle_count=3,
            )

        self.assertIsNone(result)
        mocked_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
