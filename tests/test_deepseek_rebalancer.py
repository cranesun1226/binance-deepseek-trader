import unittest
from unittest.mock import Mock, patch

from src.ai import deepseek_rebalancer


def _candidate(symbol, decision, start):
    prices = [start + float(index) for index in range(3)]
    return {
        "symbol": symbol,
        "decision": decision,
        "reference_price": prices[-1],
        "timeframes": {"1h": prices},
    }


class DeepSeekRebalancerTests(unittest.TestCase):
    def test_rebalance_payload_uses_equal_priority_candidate_shape(self):
        payload = deepseek_rebalancer._build_rebalance_input_payload(
            candidates=[
                _candidate("NQUSDT", "SHORT", 200.0),
                _candidate("ESUSDT", "LONG", 100.0),
            ],
            timeframe="1h",
            candle_count=3,
        )

        self.assertEqual(
            [candidate["symbol"] for candidate in payload["candidates"]],
            ["ESUSDT", "NQUSDT"],
        )
        for candidate in payload["candidates"]:
            self.assertEqual(set(candidate.keys()), {"symbol", "decision", "reference_price", "timeframes"})
            self.assertNotIn("current", candidate)
            self.assertNotIn("existing", candidate)
            self.assertNotIn("replacement", candidate)

        prompt = deepseek_rebalancer._format_rebalance_prompt(payload)
        lowered_prompt = prompt.lower()
        self.assertEqual(deepseek_rebalancer.REBALANCE_REASON_MAX_WORDS, 500)
        self.assertIn("500 words or fewer", prompt)
        self.assertNotIn("current", lowered_prompt)
        self.assertNotIn("existing", lowered_prompt)
        self.assertNotIn("replacement", lowered_prompt)

    def test_structured_rebalance_call_parses_candidate_symbol_selection(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
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

        with patch("src.ai.deepseek_rebalancer.get_deepseek_api_key", return_value="key"), patch(
            "src.ai.deepseek_rebalancer.requests.post", return_value=response
        ) as mocked_post:
            result = deepseek_rebalancer._call_deepseek_rebalance_selection(
                prompt="prompt",
                reasoning_effort="high",
                allowed_symbols=["ESUSDT", "NQUSDT"],
                max_tokens=123,
                timeout_seconds=45.0,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.decision.selected_symbol, "ESUSDT")
        self.assertEqual(result.reasoning, "private reasoning")
        payload = mocked_post.call_args.kwargs["json"]
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertEqual(payload["max_tokens"], 123)
        self.assertIn("selected_symbol", payload["messages"][0]["content"])
        self.assertEqual(mocked_post.call_args.kwargs["timeout"], (10.0, 45.0))

    def test_rebalance_response_rejects_symbols_outside_supplied_candidates(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"selected_symbol":"BTCUSDT","reason":"This symbol was not supplied."}'
                    }
                }
            ],
            "usage": {},
        }

        with patch("src.ai.deepseek_rebalancer.get_deepseek_api_key", return_value="key"), patch(
            "src.ai.deepseek_rebalancer.requests.post", return_value=response
        ) as mocked_post, patch("src.ai.deepseek_rebalancer.time.sleep") as mocked_sleep, patch(
            "src.ai.deepseek_rebalancer.logger"
        ):
            result = deepseek_rebalancer._call_deepseek_rebalance_selection(
                prompt="prompt",
                reasoning_effort="high",
                allowed_symbols=["ESUSDT", "NQUSDT"],
            )

        self.assertIsNone(result)
        self.assertEqual(mocked_post.call_count, deepseek_rebalancer.DEEPSEEK_GENERATE_MAX_RETRIES)
        self.assertEqual(mocked_sleep.call_count, deepseek_rebalancer.DEEPSEEK_GENERATE_MAX_RETRIES - 1)

    def test_evaluate_rebalance_fails_closed_on_invalid_payload(self):
        with patch("src.ai.deepseek_rebalancer._call_deepseek_rebalance_selection") as mocked_call, patch(
            "src.ai.deepseek_rebalancer.logger"
        ):
            result = deepseek_rebalancer.evaluate_active_rebalance_symbol(
                cycle_dir="/tmp/deepseek-rebalance-test",
                candidates=[_candidate("ESUSDT", "LONG", 100.0)],
                timeframe="1h",
                candle_count=3,
            )

        self.assertIsNone(result)
        mocked_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
