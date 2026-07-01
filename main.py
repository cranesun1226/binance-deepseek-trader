"""CLI entrypoint for Binance ZAI Trader."""

import argparse
from typing import Any, Dict, Optional

from src.infra.logger import get_logger
from src.strategy.scheduler import TradingScheduler
from src.strategy.portfolio_strategy import run_portfolio_cycle

logger = get_logger("main")
ONCE_RESULT_FIELDS = (
    "success",
    "action",
    "slot_id",
    "slot_label",
    "symbol",
    "candidate_symbol",
    "current_price",
    "ai_triggered",
    "ai_decision",
)


def run_portfolio_strategy_cycle(
    *,
    state: Optional[Dict[str, Any]] = None,
    as_of_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Expose a single Binance ZAI Trader cycle for external callers and tests."""
    return run_portfolio_cycle(state=state, as_of_ms=as_of_ms)


def _print_once_summary(result: Dict[str, Any]) -> None:
    for field_name in ONCE_RESULT_FIELDS:
        print(f"{field_name}={result.get(field_name)}")

    cycle_dir = result.get("cycle_dir")
    if cycle_dir:
        print(f"cycle_dir={cycle_dir}")


def main_once() -> None:
    """Run one trading cycle and print a concise CLI summary."""
    logger.info("=" * 60)
    logger.info("=== Binance ZAI Trader (once mode) ===")
    logger.info("=" * 60)

    scheduler = TradingScheduler()
    result = scheduler.run_cycle_once()
    _print_once_summary(result)


def main_scheduled() -> None:
    """Start the long-running scheduler loop."""
    logger.info("=" * 60)
    logger.info("=== Binance ZAI Trader (scheduled mode) ===")
    logger.info("=" * 60)

    scheduler = TradingScheduler()
    scheduler.run_forever()


def main() -> None:
    """Parse CLI arguments and dispatch the requested runtime mode."""
    parser = argparse.ArgumentParser(description="Binance ZAI Trader | two-slot USDT-M futures bot")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one cycle and exit",
    )
    args = parser.parse_args()

    if args.once:
        main_once()
    else:
        main_scheduled()


if __name__ == "__main__":
    main()
