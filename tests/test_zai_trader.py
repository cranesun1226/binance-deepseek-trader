import json
import unittest
from unittest.mock import Mock, patch

from src.ai import zai_trader


class ZAITraderTests(unittest.TestCase):
    def test_direction_model_uses_glm_5_2_with_max_reasoning(self):
        self.assertEqual(zai_trader.ZAI_DIRECTION_MODEL, "glm-5.2")
        self.assertEqual(zai_trader.ZAI_DEFAULT_REASONING_EFFORT, "max")
        self.assertEqual(zai_trader.ZAI_MAX_REASONING_EFFORT, "max")
        self.assertEqual(zai_trader.ZAI_DEFAULT_TIMEOUT_SECONDS, 300.0)

    def test_structured_call_sends_zai_sdk_payload_and_parses_decision(self):
        response_payload = {
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
        client = Mock()
        client.chat.completions.create.return_value = response_payload

        with patch("src.ai.zai_trader.get_zai_api_key", return_value="key"), patch(
            "src.ai.zai_trader.ZaiClient", return_value=client
        ) as mocked_client:
            result = zai_trader._call_zai_structured_decision(
                prompt="prompt",
                reasoning_effort="max",
                response_model=zai_trader.TradeDirectionDecision,
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
        mocked_client.assert_called_once_with(api_key="key")
        payload = client.chat.completions.create.call_args.kwargs
        self.assertEqual(payload["model"], "glm-5.2")
        self.assertEqual(
            payload["messages"][0]["content"],
            "You are a world-class USDT perpetual futures crypto trader. "
            "Use your own trader judgment and the supplied market data to choose the LONG or SHORT direction that appears more likely to produce future profit. "
            "Use only English to reason and respond. "
            "Return exactly one json object containing only the decision and reason. "
            "The reason must be english, reasonable, data-based, and 300 characters or fewer.",
        )
        self.assertEqual(payload["messages"][1]["content"], "prompt")
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "max")
        self.assertNotIn("provider", payload)
        self.assertEqual(payload["max_tokens"], 1234)
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_xhigh_reasoning_alias_maps_to_zai_max(self):
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": '{"decision":"SHORT","reason":"The full close series is rolling over, with weak rebounds and lower highs favoring downside continuation."}'
                    }
                }
            ],
            "usage": {},
        }
        client = Mock()
        client.chat.completions.create.return_value = response_payload

        with patch("src.ai.zai_trader.get_zai_api_key", return_value="key"), patch(
            "src.ai.zai_trader.ZaiClient", return_value=client
        ):
            result = zai_trader._call_zai_structured_decision(
                prompt="prompt",
                reasoning_effort="xhigh",
                response_model=zai_trader.TradeDirectionDecision,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.decision.decision, "SHORT")
        payload = client.chat.completions.create.call_args.kwargs
        self.assertEqual(payload["reasoning_effort"], "max")

    def test_default_reasoning_effort_uses_max(self):
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": '{"decision":"SHORT","reason":"The full close series is rolling over, with weak rebounds and lower highs favoring downside continuation."}'
                    }
                }
            ],
            "usage": {},
        }
        client = Mock()
        client.chat.completions.create.return_value = response_payload

        with patch("src.ai.zai_trader.get_zai_api_key", return_value="key"), patch(
            "src.ai.zai_trader.ZaiClient", return_value=client
        ):
            result = zai_trader._call_zai_structured_decision(
                prompt="prompt",
                reasoning_effort="",
                response_model=zai_trader.TradeDirectionDecision,
            )

        self.assertIsNotNone(result)
        payload = client.chat.completions.create.call_args.kwargs
        self.assertEqual(payload["reasoning_effort"], "max")

    def test_empty_zai_content_retries_cleanly(self):
        empty_response = {
            "choices": [{"finish_reason": "stop", "message": {"content": ""}}],
            "usage": {},
        }
        valid_response = {
            "choices": [
                {
                    "message": {
                        "content": '{"decision":"LONG","reason":"The full close series stabilized after the dip, and recent closes hold above the recovery base, favoring long exposure."}'
                    }
                }
            ],
            "usage": {},
        }
        client = Mock()
        client.chat.completions.create.side_effect = [empty_response, valid_response]

        with patch("src.ai.zai_trader.get_zai_api_key", return_value="key"), patch(
            "src.ai.zai_trader.ZaiClient", return_value=client
        ), patch("src.ai.zai_trader.time.sleep") as mocked_sleep:
            result = zai_trader._call_zai_structured_decision(
                prompt="prompt",
                reasoning_effort="high",
                response_model=zai_trader.TradeDirectionDecision,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.decision.decision, "LONG")
        self.assertEqual(client.chat.completions.create.call_count, 2)
        mocked_sleep.assert_called_once()

    def test_authentication_error_is_raised_without_retrying(self):
        client = Mock()
        client.chat.completions.create.side_effect = Exception(
            'Error code: 401, with error text {"error":{"code":"1000","message":"Authentication Failed"}}'
        )

        with patch("src.ai.zai_trader.get_zai_api_key", return_value="key"), patch(
            "src.ai.zai_trader.ZaiClient", return_value=client
        ), patch("src.ai.zai_trader.time.sleep") as mocked_sleep, patch("src.ai.zai_trader.logger"):
            with self.assertRaises(zai_trader.ZAIAuthenticationError):
                zai_trader._call_zai_structured_decision(
                    prompt="prompt",
                    reasoning_effort="max",
                    response_model=zai_trader.TradeDirectionDecision,
                )

        self.assertEqual(client.chat.completions.create.call_count, 1)
        mocked_sleep.assert_not_called()

    def test_extra_decision_fields_are_retried_before_accepting_valid_json(self):
        extra_field_response = {
            "choices": [
                {
                    "message": {
                        "content": '{"decision":"LONG","reason":"The full close series supports upside continuation.","confidence":0.72}'
                    }
                }
            ],
            "usage": {},
        }
        valid_response = {
            "choices": [
                {
                    "message": {
                        "content": '{"decision":"SHORT","reason":"The full close series keeps failing at lower highs, while the latest closes remain weak and favor downside exposure."}'
                    }
                }
            ],
            "usage": {},
        }
        client = Mock()
        client.chat.completions.create.side_effect = [extra_field_response, valid_response]

        with patch("src.ai.zai_trader.get_zai_api_key", return_value="key"), patch(
            "src.ai.zai_trader.ZaiClient", return_value=client
        ), patch("src.ai.zai_trader.time.sleep") as mocked_sleep:
            result = zai_trader._call_zai_structured_decision(
                prompt="prompt",
                reasoning_effort="high",
                response_model=zai_trader.TradeDirectionDecision,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.decision.decision, "SHORT")
        self.assertEqual(client.chat.completions.create.call_count, 2)
        mocked_sleep.assert_called_once()

    def test_non_exact_decision_json_fails_closed_after_retries(self):
        lowercase_response = {
            "choices": [
                {
                    "message": {
                        "content": '{"decision":"long","reason":"The full close series shows constructive higher lows, but the decision casing is invalid."}'
                    }
                }
            ],
            "usage": {},
        }
        client = Mock()
        client.chat.completions.create.return_value = lowercase_response

        with patch("src.ai.zai_trader.get_zai_api_key", return_value="key"), patch(
            "src.ai.zai_trader.ZaiClient", return_value=client
        ), patch("src.ai.zai_trader.time.sleep") as mocked_sleep, patch(
            "src.ai.zai_trader.logger"
        ):
            result = zai_trader._call_zai_structured_decision(
                prompt="prompt",
                reasoning_effort="high",
                response_model=zai_trader.TradeDirectionDecision,
            )

        self.assertIsNone(result)
        self.assertEqual(client.chat.completions.create.call_count, zai_trader.ZAI_GENERATE_MAX_RETRIES)
        self.assertEqual(mocked_sleep.call_count, zai_trader.ZAI_GENERATE_MAX_RETRIES - 1)

    def test_decision_reason_over_character_limit_is_rejected(self):
        overlong_reason = "x" * (zai_trader.DECISION_REASON_MAX_CHARS + 1)
        raw_response = json.dumps({"decision": "LONG", "reason": overlong_reason})

        with self.assertRaises(ValueError):
            zai_trader._parse_strict_decision_response(
                raw_response,
                zai_trader.TradeDirectionDecision,
            )

    def test_decision_reason_with_cjk_text_is_rejected(self):
        raw_response = json.dumps({"decision": "SHORT", "reason": "The flow is weak \u4e0a downside continuation."})

        with self.assertRaises(ValueError):
            zai_trader._parse_strict_decision_response(
                raw_response,
                zai_trader.TradeDirectionDecision,
            )

    def test_invalid_prompt_payload_fails_before_zai_call(self):
        with patch("src.ai.zai_trader._call_zai_structured_decision") as mocked_call, patch(
            "src.ai.zai_trader.logger"
        ):
            decision = zai_trader.evaluate_trade_direction(
                cycle_dir="/tmp/zai-test",
                symbol="BTCUSDT",
                reference_price=100.0,
                timeframe_ohlcv={"1h": []},
                reasoning_effort="high",
            )

        self.assertIsNone(decision)
        mocked_call.assert_not_called()

    def test_direction_prompt_contract_uses_symbol_reference_and_close_prices_only(self):
        prompt = zai_trader._build_direction_prompt(
            symbol="btcusdt",
            reference_price=100.0,
            timeframe_ohlcv={"1h": [98.0, 99.0, 100.0]},
        )

        self.assertEqual(
            prompt,
            'You are a world-class BTCUSDT trader.\n'
            'Return JSON only with exactly two fields: decision and reason.\n'
            'Use your own trader judgment and the supplied market data to choose the LONG or SHORT position that appears more likely to produce future profit.\n'
            'The reason must be english, reasonable, data-based, and 300 characters or fewer.\n'
            'Examples: {"decision":"LONG","reason":"..."} or {"decision":"SHORT","reason":"..."}.\n'
            'Market payload:\n{"symbol":"BTCUSDT","reference_price":100.0,"timeframes":{"1h":[98.0,99.0,100.0]}}',
        )
        lowered_prompt = prompt.lower()
        self.assertNotIn("trend-following", lowered_prompt)
        self.assertNotIn("momentum", lowered_prompt)
        self.assertNotIn("consensus", lowered_prompt)

    def test_cost_estimate_returns_none_when_glm_5_2_pricing_is_unconfigured(self):
        estimate = zai_trader.estimate_zai_cost(
            {
                "prompt_tokens": 1000,
                "prompt_cache_hit_tokens": 200,
                "prompt_cache_miss_tokens": 800,
                "completion_tokens": 100,
                "total_tokens": 1100,
            }
        )

        self.assertIsNone(estimate)


if __name__ == "__main__":
    unittest.main()
