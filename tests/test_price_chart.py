import unittest

from src.infra.price_chart import build_close_price_line_chart_png


class PriceChartTests(unittest.TestCase):
    def test_close_price_line_chart_renders_png(self):
        close_prices = [100000.0 + float(index * 10) for index in range(100)]

        chart_bytes = build_close_price_line_chart_png(close_prices)

        # The renderer returns a real PNG payload that Telegram can upload via sendPhoto.
        self.assertTrue(chart_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(chart_bytes), 1000)

    def test_close_price_line_chart_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            build_close_price_line_chart_png([100000.0, -1.0])


if __name__ == "__main__":
    unittest.main()
