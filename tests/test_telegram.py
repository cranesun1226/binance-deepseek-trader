import unittest

from src.infra import telegram


class TelegramTests(unittest.TestCase):
    def test_sanitize_telegram_error_redacts_bot_token_in_url(self):
        raw = "HTTPSConnectionPool(host='api.telegram.org', url='/bot123:secret/sendMessage')"

        sanitized = telegram._sanitize_telegram_error(raw)

        self.assertIn("/bot<redacted>/sendMessage", sanitized)
        self.assertNotIn("123:secret", sanitized)


if __name__ == "__main__":
    unittest.main()
