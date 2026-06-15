import json
import unittest
from unittest.mock import Mock, patch

from src.ai import deepseek_trader


class DeepSeekTraderTests(unittest.TestCase):
    def test_direction_model_uses_current_deepseek_v4_flash_id(self):
        self.assertEqual(deepseek_trader.DEEPSEEK_DIRECTION_MODEL, "deepseek-v4-flash")
        self.assertEqual(deepseek_trader.DEEPSEEK_DEFAULT_REASONING_EFFORT, "max")
        self.assertEqual(deepseek_trader.DEEPSEEK_MAX_REASONING_EFFORT, "max")
        self.assertEqual(deepseek_trader.DEEPSEEK_DEFAULT_TIMEOUT_SECONDS, 300.0)

    def test_structured_call_sends_deepseek_payload_and_parses_decision(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "id": "completion-id",
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"decision":"LONG","reason":"The full close series shows constructive higher lows, '
                            'with the latest price holding above the prior consolidation area and favoring upside continuation."}'
                        ),
                        "reasoning_content": "full reasoning text",
                    }
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }

        with patch("src.ai.deepseek_trader.get_deepseek_api_key", return_value="key"), patch(
            "src.ai.deepseek_trader.requests.post", return_value=response
        ) as mocked_post:
            result = deepseek_trader._call_deepseek_structured_decision(
                prompt="prompt",
                reasoning_effort="high",
                response_model=deepseek_trader.TradeDirectionDecision,
                max_tokens=1234,
                timeout_seconds=345.0,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.decision.decision, "LONG")
        self.assertEqual(
            result.decision.reason,
            "The full close series shows constructive higher lows, with the latest price holding above the prior consolidation area and favoring upside continuation.",
        )
        self.assertEqual(result.reasoning, "full reasoning text")
        self.assertEqual(mocked_post.call_args.args[0], "https://api.deepseek.com/chat/completions")
        self.assertEqual(mocked_post.call_args.kwargs["headers"]["Authorization"], "Bearer key")
        payload = mocked_post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "deepseek-v4-flash")
        self.assertEqual(
            payload["messages"][0]["content"],
            "You are a world-class USDT perpetual futures crypto trader. "
            "Analyze with a trend-following market consensus view, favoring the LONG or SHORT direction that a clear supermajority of reasonable momentum traders would broadly agree has the higher expected value. "
            "Use only English to reason and respond. "
            "Return exactly one json object containing only the decision and reason. "
            "The reason must be english, reasonable, data-based, and 200 words or fewer.",
        )
        self.assertEqual(payload["messages"][1]["content"], "prompt")
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertNotIn("provider", payload)
        self.assertEqual(payload["max_tokens"], 1234)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(mocked_post.call_args.kwargs["timeout"], (10.0, 345.0))

    def test_xhigh_reasoning_alias_maps_to_deepseek_max(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"decision":"SHORT","reason":"The full close series is rolling over, with weak rebounds and lower highs favoring downside continuation."}'
                    }
                }
            ],
            "usage": {},
        }

        with patch("src.ai.deepseek_trader.get_deepseek_api_key", return_value="key"), patch(
            "src.ai.deepseek_trader.requests.post", return_value=response
        ) as mocked_post:
            result = deepseek_trader._call_deepseek_structured_decision(
                prompt="prompt",
                reasoning_effort="xhigh",
                response_model=deepseek_trader.TradeDirectionDecision,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.decision.decision, "SHORT")
        payload = mocked_post.call_args.kwargs["json"]
        self.assertEqual(payload["reasoning_effort"], "max")

    def test_default_reasoning_effort_uses_max(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"decision":"SHORT","reason":"The full close series is rolling over, with weak rebounds and lower highs favoring downside continuation."}'
                    }
                }
            ],
            "usage": {},
        }

        with patch("src.ai.deepseek_trader.get_deepseek_api_key", return_value="key"), patch(
            "src.ai.deepseek_trader.requests.post", return_value=response
        ) as mocked_post:
            result = deepseek_trader._call_deepseek_structured_decision(
                prompt="prompt",
                reasoning_effort="",
                response_model=deepseek_trader.TradeDirectionDecision,
            )

        self.assertIsNotNone(result)
        payload = mocked_post.call_args.kwargs["json"]
        self.assertEqual(payload["reasoning_effort"], "max")

    def test_empty_deepseek_content_retries_cleanly(self):
        empty_response = Mock()
        empty_response.status_code = 200
        empty_response.json.return_value = {
            "choices": [{"finish_reason": "stop", "message": {"content": ""}}],
            "usage": {},
        }
        valid_response = Mock()
        valid_response.status_code = 200
        valid_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"decision":"LONG","reason":"The full close series stabilized after the dip, and recent closes hold above the recovery base, favoring long exposure."}'
                    }
                }
            ],
            "usage": {},
        }

        with patch("src.ai.deepseek_trader.get_deepseek_api_key", return_value="key"), patch(
            "src.ai.deepseek_trader.requests.post", side_effect=[empty_response, valid_response]
        ) as mocked_post, patch("src.ai.deepseek_trader.time.sleep") as mocked_sleep:
            result = deepseek_trader._call_deepseek_structured_decision(
                prompt="prompt",
                reasoning_effort="high",
                response_model=deepseek_trader.TradeDirectionDecision,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.decision.decision, "LONG")
        self.assertEqual(mocked_post.call_count, 2)
        mocked_sleep.assert_called_once()

    def test_extra_decision_fields_are_retried_before_accepting_valid_json(self):
        extra_field_response = Mock()
        extra_field_response.status_code = 200
        extra_field_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"decision":"LONG","reason":"The full close series supports upside continuation.","confidence":0.72}'
                    }
                }
            ],
            "usage": {},
        }
        valid_response = Mock()
        valid_response.status_code = 200
        valid_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"decision":"SHORT","reason":"The full close series keeps failing at lower highs, while the latest closes remain weak and favor downside exposure."}'
                    }
                }
            ],
            "usage": {},
        }

        with patch("src.ai.deepseek_trader.get_deepseek_api_key", return_value="key"), patch(
            "src.ai.deepseek_trader.requests.post", side_effect=[extra_field_response, valid_response]
        ) as mocked_post, patch("src.ai.deepseek_trader.time.sleep") as mocked_sleep:
            result = deepseek_trader._call_deepseek_structured_decision(
                prompt="prompt",
                reasoning_effort="high",
                response_model=deepseek_trader.TradeDirectionDecision,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.decision.decision, "SHORT")
        self.assertEqual(mocked_post.call_count, 2)
        mocked_sleep.assert_called_once()

    def test_non_exact_decision_json_fails_closed_after_retries(self):
        lowercase_response = Mock()
        lowercase_response.status_code = 200
        lowercase_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"decision":"long","reason":"The full close series shows constructive higher lows, but the decision casing is invalid."}'
                    }
                }
            ],
            "usage": {},
        }

        with patch("src.ai.deepseek_trader.get_deepseek_api_key", return_value="key"), patch(
            "src.ai.deepseek_trader.requests.post", return_value=lowercase_response
        ) as mocked_post, patch("src.ai.deepseek_trader.time.sleep") as mocked_sleep, patch(
            "src.ai.deepseek_trader.logger"
        ):
            result = deepseek_trader._call_deepseek_structured_decision(
                prompt="prompt",
                reasoning_effort="high",
                response_model=deepseek_trader.TradeDirectionDecision,
            )

        self.assertIsNone(result)
        self.assertEqual(mocked_post.call_count, deepseek_trader.DEEPSEEK_GENERATE_MAX_RETRIES)
        self.assertEqual(mocked_sleep.call_count, deepseek_trader.DEEPSEEK_GENERATE_MAX_RETRIES - 1)

    def test_decision_reason_over_word_limit_is_rejected(self):
        overlong_reason = " ".join(["word"] * (deepseek_trader.DECISION_REASON_MAX_WORDS + 1))
        raw_response = json.dumps({"decision": "LONG", "reason": overlong_reason})

        with self.assertRaises(ValueError):
            deepseek_trader._parse_strict_decision_response(
                raw_response,
                deepseek_trader.TradeDirectionDecision,
            )

    def test_decision_reason_with_cjk_text_is_rejected(self):
        raw_response = json.dumps({"decision": "SHORT", "reason": "The flow is weak \u4e0a downside continuation."})

        with self.assertRaises(ValueError):
            deepseek_trader._parse_strict_decision_response(
                raw_response,
                deepseek_trader.TradeDirectionDecision,
            )

    def test_invalid_prompt_payload_fails_before_deepseek_call(self):
        with patch("src.ai.deepseek_trader._call_deepseek_structured_decision") as mocked_call, patch(
            "src.ai.deepseek_trader.logger"
        ):
            decision = deepseek_trader.evaluate_trade_direction(
                cycle_dir="/tmp/deepseek-test",
                symbol="BTCUSDT",
                reference_price=100.0,
                timeframe_ohlcv={"1h": []},
                reasoning_effort="high",
            )

        self.assertIsNone(decision)
        mocked_call.assert_not_called()

    def test_direction_prompt_contract_uses_symbol_reference_and_close_prices_only(self):
        prompt = deepseek_trader._build_direction_prompt(
            symbol="btcusdt",
            reference_price=100.0,
            timeframe_ohlcv={"1h": [98.0, 99.0, 100.0]},
        )

        self.assertEqual(
            prompt,
            'You are a world-class BTCUSDT trader.\n'
            'Return JSON only with exactly two fields: decision and reason.\n'
            'Analyze with a trend-following market consensus view, favoring the LONG or SHORT direction that a clear supermajority of reasonable trend-following traders would broadly agree has the higher expected value.\n'
            'The reason must be english, reasonable, data-based, and 200 words or fewer.\n'
            'Examples: {"decision":"LONG","reason":"..."} or {"decision":"SHORT","reason":"..."}.\n'
            'Market payload:\n{"symbol":"BTCUSDT","reference_price":100.0,"timeframes":{"1h":[98.0,99.0,100.0]}}',
        )

    def test_cost_estimate_uses_deepseek_cache_hit_and_miss_rates(self):
        estimate = deepseek_trader.estimate_deepseek_cost(
            {
                "prompt_tokens": 1000,
                "prompt_cache_hit_tokens": 200,
                "prompt_cache_miss_tokens": 800,
                "completion_tokens": 100,
                "total_tokens": 1100,
            }
        )

        self.assertEqual(estimate["prompt_cache_hit_tokens"], 200)
        self.assertEqual(estimate["prompt_cache_miss_tokens"], 800)
        self.assertEqual(estimate["input_cost_usd"], 0.00011256)
        self.assertEqual(estimate["output_cost_usd"], 0.000028)
        self.assertEqual(estimate["total_cost_usd"], 0.00014056)


if __name__ == "__main__":
    unittest.main()
