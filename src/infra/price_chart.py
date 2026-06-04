"""PNG chart rendering helpers for Telegram price notifications."""

from __future__ import annotations

import math
import struct
import zlib
from typing import Any, Iterable

Color = tuple[int, int, int]

_CHART_WIDTH = 1200
_CHART_HEIGHT = 675
_BACKGROUND: Color = (255, 255, 255)
_PLOT_BACKGROUND: Color = (248, 250, 252)
_GRID_COLOR: Color = (226, 232, 240)
_AXIS_COLOR: Color = (71, 85, 105)
_LABEL_COLOR: Color = (51, 65, 85)
_UP_LINE_COLOR: Color = (0, 126, 87)
_UP_SHADOW_COLOR: Color = (187, 247, 208)
_DOWN_LINE_COLOR: Color = (200, 48, 48)
_DOWN_SHADOW_COLOR: Color = (254, 202, 202)
_START_MARKER_COLOR: Color = (100, 116, 139)

_NUMERIC_GLYPHS: dict[str, tuple[str, ...]] = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    ",": ("000", "000", "000", "010", "100"),
    ".": ("000", "000", "000", "000", "010"),
    "-": ("000", "000", "111", "000", "000"),
    " ": ("000", "000", "000", "000", "000"),
}


class _Canvas:
    """Tiny RGB canvas so the service can render charts without heavy image dependencies."""

    def __init__(self, width: int, height: int, background: Color) -> None:
        self.width = int(width)
        self.height = int(height)
        self.pixels = bytearray(background) * (self.width * self.height)

    def _set_pixel(self, x: int, y: int, color: Color) -> None:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return
        offset = ((y * self.width) + x) * 3
        self.pixels[offset : offset + 3] = bytes(color)

    def fill_rect(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
        # Clamp rectangle coordinates so labels or markers near edges stay harmless.
        left = max(0, min(int(x0), int(x1), self.width))
        right = max(0, min(max(int(x0), int(x1)), self.width))
        top = max(0, min(int(y0), int(y1), self.height))
        bottom = max(0, min(max(int(y0), int(y1)), self.height))
        for y in range(top, bottom):
            for x in range(left, right):
                self._set_pixel(x, y, color)

    def draw_circle(self, cx: int, cy: int, radius: int, color: Color) -> None:
        radius = max(0, int(radius))
        radius_sq = radius * radius
        for y in range(int(cy) - radius, int(cy) + radius + 1):
            for x in range(int(cx) - radius, int(cx) + radius + 1):
                if ((x - int(cx)) ** 2) + ((y - int(cy)) ** 2) <= radius_sq:
                    self._set_pixel(x, y, color)

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: Color, *, thickness: int = 1) -> None:
        # Interpolated points plus small circles produce a readable anti-gap thick line.
        dx = int(x1) - int(x0)
        dy = int(y1) - int(y0)
        steps = max(abs(dx), abs(dy), 1)
        radius = max(0, int(thickness) // 2)
        for step in range(steps + 1):
            ratio = step / steps
            x = round(int(x0) + (dx * ratio))
            y = round(int(y0) + (dy * ratio))
            if radius <= 0:
                self._set_pixel(x, y, color)
            else:
                self.draw_circle(x, y, radius, color)

    def draw_polyline(self, points: list[tuple[int, int]], color: Color, *, thickness: int) -> None:
        for start, end in zip(points, points[1:]):
            self.draw_line(start[0], start[1], end[0], end[1], color, thickness=thickness)

    def to_png(self) -> bytes:
        # Encode unfiltered RGB scanlines into a small standards-compliant PNG.
        raw = bytearray()
        stride = self.width * 3
        for y in range(self.height):
            raw.append(0)
            row_start = y * stride
            raw.extend(self.pixels[row_start : row_start + stride])

        ihdr = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        return b"".join(
            [
                b"\x89PNG\r\n\x1a\n",
                _png_chunk(b"IHDR", ihdr),
                _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=6)),
                _png_chunk(b"IEND", b""),
            ]
        )


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _normalize_price_series(close_prices: Iterable[Any], *, limit: int) -> list[float]:
    if isinstance(close_prices, (str, bytes)):
        raise ValueError("close_prices must be an iterable of numeric prices")

    values = list(close_prices)[-max(1, int(limit)) :]
    prices: list[float] = []
    for index, value in enumerate(values):
        try:
            price = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"close_prices[{index}] is not numeric") from None
        if not math.isfinite(price) or price <= 0.0:
            raise ValueError(f"close_prices[{index}] must be a positive finite price")
        prices.append(price)

    if not prices:
        raise ValueError("close_prices must contain at least one price")
    return prices


def _axis_label(value: float) -> str:
    absolute_value = abs(float(value))
    if absolute_value >= 1000:
        return f"{value:,.0f}"
    if absolute_value >= 10:
        return f"{value:,.2f}"
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def _measure_numeric_text(text: str, *, scale: int) -> int:
    width = 0
    spacing = max(1, int(scale))
    for char in str(text):
        glyph = _NUMERIC_GLYPHS.get(char, _NUMERIC_GLYPHS[" "])
        width += (len(glyph[0]) * scale) + spacing
    return max(0, width - spacing)


def _draw_numeric_text(
    canvas: _Canvas,
    text: str,
    x: int,
    y: int,
    color: Color,
    *,
    scale: int = 3,
    anchor: str = "left",
) -> None:
    # A compact bitmap font is enough for price ticks while keeping dependencies at zero.
    text = str(text)
    text_width = _measure_numeric_text(text, scale=scale)
    cursor_x = int(x)
    if anchor == "right":
        cursor_x -= text_width
    elif anchor == "center":
        cursor_x -= text_width // 2

    spacing = max(1, int(scale))
    for char in text:
        glyph = _NUMERIC_GLYPHS.get(char, _NUMERIC_GLYPHS[" "])
        for row_index, row in enumerate(glyph):
            for column_index, bit in enumerate(row):
                if bit != "1":
                    continue
                canvas.fill_rect(
                    cursor_x + (column_index * scale),
                    int(y) + (row_index * scale),
                    cursor_x + ((column_index + 1) * scale),
                    int(y) + ((row_index + 1) * scale),
                    color,
                )
        cursor_x += (len(glyph[0]) * scale) + spacing


def build_close_price_line_chart_png(
    close_prices: Iterable[Any],
    *,
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    limit: int = 100,
) -> bytes:
    """Render recent close prices as a high-contrast Telegram-friendly PNG line chart."""
    del symbol, timeframe
    prices = _normalize_price_series(close_prices, limit=limit)

    canvas = _Canvas(_CHART_WIDTH, _CHART_HEIGHT, _BACKGROUND)
    plot_left = 128
    plot_right = _CHART_WIDTH - 48
    plot_top = 42
    plot_bottom = _CHART_HEIGHT - 86
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    min_price = min(prices)
    max_price = max(prices)
    price_range = max_price - min_price
    if price_range <= 0.0:
        price_range = max(min_price * 0.002, 1.0)
        min_price -= price_range / 2.0
        max_price += price_range / 2.0
    else:
        padding = max(price_range * 0.08, max_price * 0.0005)
        min_price -= padding
        max_price += padding

    def _point(index: int, price: float) -> tuple[int, int]:
        if len(prices) == 1:
            x = plot_left + (plot_width // 2)
        else:
            x = plot_left + round((index / (len(prices) - 1)) * plot_width)
        y_ratio = (price - min_price) / (max_price - min_price)
        y = plot_bottom - round(y_ratio * plot_height)
        return x, y

    canvas.fill_rect(plot_left, plot_top, plot_right, plot_bottom, _PLOT_BACKGROUND)

    horizontal_ticks = 6
    for tick in range(horizontal_ticks):
        ratio = tick / (horizontal_ticks - 1)
        y = plot_bottom - round(ratio * plot_height)
        price = min_price + (ratio * (max_price - min_price))
        canvas.draw_line(plot_left, y, plot_right, y, _GRID_COLOR, thickness=1)
        _draw_numeric_text(
            canvas,
            _axis_label(price),
            plot_left - 12,
            y - 8,
            _LABEL_COLOR,
            scale=3,
            anchor="right",
        )

    vertical_ticks = 5
    for tick in range(vertical_ticks):
        ratio = tick / (vertical_ticks - 1)
        x = plot_left + round(ratio * plot_width)
        candle_number = 1 + round(ratio * (len(prices) - 1))
        canvas.draw_line(x, plot_top, x, plot_bottom, _GRID_COLOR, thickness=1)
        _draw_numeric_text(
            canvas,
            str(candle_number),
            x,
            plot_bottom + 22,
            _LABEL_COLOR,
            scale=3,
            anchor="center",
        )

    canvas.draw_line(plot_left, plot_top, plot_left, plot_bottom, _AXIS_COLOR, thickness=2)
    canvas.draw_line(plot_left, plot_bottom, plot_right, plot_bottom, _AXIS_COLOR, thickness=2)

    points = [_point(index, price) for index, price in enumerate(prices)]
    is_up = prices[-1] >= prices[0]
    shadow_color = _UP_SHADOW_COLOR if is_up else _DOWN_SHADOW_COLOR
    line_color = _UP_LINE_COLOR if is_up else _DOWN_LINE_COLOR

    canvas.draw_polyline(points, shadow_color, thickness=10)
    canvas.draw_polyline(points, line_color, thickness=4)

    # Mark the first and latest closes so the direction of the 100-candle series is obvious.
    first_x, first_y = points[0]
    last_x, last_y = points[-1]
    canvas.draw_circle(first_x, first_y, 8, _BACKGROUND)
    canvas.draw_circle(first_x, first_y, 5, _START_MARKER_COLOR)
    canvas.draw_circle(last_x, last_y, 10, _BACKGROUND)
    canvas.draw_circle(last_x, last_y, 6, line_color)

    return canvas.to_png()


__all__ = ["build_close_price_line_chart_png"]
