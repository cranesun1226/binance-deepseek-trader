"""Run one live-data portfolio cycle while blocking all Binance order writes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict
from unittest.mock import patch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.infra.env_loader import load_env_var
from src.infra.telegram import send_telegram_message
from src.strategy.portfolio_strategy import run_portfolio_cycle
from src.strategy.scheduler import TradingScheduler


def _fake_account_overview(_api_key: str, _api_secret: str) -> Dict[str, float]:
    return {
        "equity": 1000.0,
        "available_balance": 1000.0,
        "wallet_balance": 1000.0,
        "unrealized_profit": 0.0,
    }


def _fake_positions(_api_key: str, _api_secret: str) -> list[Dict[str, Any]]:
    return []


def _fake_place_market_entry_order(
    _api_key: str,
    _api_secret: str,
    symbol: str,
    side: str,
    qty: str,
    *,
    leverage: int | None = None,
):
    return (
        {
            "dry_run": True,
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": qty,
            "leverage": leverage,
            "status": "DRY_RUN_ACCEPTED",
        },
        0,
        "dry_run_no_order_submitted",
    )


def _fake_close_position(*_args, **_kwargs) -> bool:
    return True


def _fake_cancel_all_symbol_orders(*_args, **_kwargs) -> bool:
    return True


def _fake_sync_existing_position_stop_loss(*_args, **_kwargs) -> Dict[str, Any]:
    return {
        "success": True,
        "changed": True,
        "dry_run": True,
        "reason": "dry_run_no_stop_order_submitted",
    }


def _fake_sync_position_after_trade(*, symbol: str, stop_loss_pct: float, **_kwargs) -> Dict[str, Any]:
    return {
        "position": {
            "symbol": symbol,
            "direction": "dry_run_unknown_until_exchange_fill",
            "size": None,
            "entry_price": None,
            "stop_loss_pct": stop_loss_pct,
        },
        "stop_sync": {
            "success": True,
            "changed": True,
            "dry_run": True,
            "reason": "dry_run_no_stop_order_submitted",
            "stop_loss_pct": stop_loss_pct,
        },
    }


def _summarize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success": result.get("success"),
        "action": result.get("action"),
        "ai_triggered": result.get("ai_triggered"),
        "screening_triggered": result.get("screening_triggered"),
        "cycle_dir": result.get("cycle_dir"),
        "slots": [
            {
                "slot_id": row.get("slot_id"),
                "slot_label": row.get("slot_label"),
                "symbol": row.get("symbol"),
                "action": row.get("action"),
                "ai_triggered": row.get("ai_triggered"),
                "ai_decision": row.get("ai_decision"),
                "screening_triggered": row.get("screening_triggered"),
                "screening_decision": row.get("screening_decision"),
                "decision_source": row.get("decision_source"),
                "leverage": row.get("leverage"),
                "target_notional_usdt": row.get("target_notional_usdt"),
            }
            for row in result.get("slot_results", [])
            if isinstance(row, dict)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Live-data dry-run cycle without Binance order submission")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram notifications for this dry run")
    args = parser.parse_args()

    checks = {
        "deepseek_key_configured": bool(load_env_var("DEEPSEEK_API_KEY")),
        "telegram_configured": bool(load_env_var("TELEGRAM_BOT_TOKEN")) and bool(load_env_var("TELEGRAM_CHAT_ID")),
        "binance_order_functions": "patched",
        "account_positions": "simulated_empty_account",
    }
    print(json.dumps({"preflight": checks}, indent=2, ensure_ascii=False))
    if not checks["deepseek_key_configured"]:
        print("DEEPSEEK_API_KEY is not configured.", file=sys.stderr)
        return 2

    scheduler = TradingScheduler()
    notification_callback = None if args.no_telegram else scheduler._notify_telegram_event
    if not args.no_telegram and checks["telegram_configured"]:
        send_telegram_message(
            "<b>Binance DeepSeek Trader dry-run validation started</b>\n"
            "No Binance orders will be submitted."
        )

    # Patch private/account-mutating Binance calls only. Public market data,
    # passive-slot DeepSeek calls, active screening, and Telegram still run
    # against live external services.
    with patch("src.strategy.portfolio_strategy.get_binance_credentials", return_value=("dry_run", "dry_run")), patch(
        "src.strategy.portfolio_strategy.get_account_overview", side_effect=_fake_account_overview
    ), patch(
        "src.strategy.portfolio_strategy.get_positions", side_effect=_fake_positions
    ), patch(
        "src.strategy.portfolio_strategy.place_market_entry_order", side_effect=_fake_place_market_entry_order
    ), patch(
        "src.strategy.portfolio_strategy.close_position", side_effect=_fake_close_position
    ), patch(
        "src.strategy.portfolio_strategy.cancel_all_symbol_orders", side_effect=_fake_cancel_all_symbol_orders
    ), patch(
        "src.strategy.portfolio_strategy.sync_existing_position_stop_loss",
        side_effect=_fake_sync_existing_position_stop_loss,
    ), patch(
        "src.strategy.portfolio_strategy._sync_position_after_trade", side_effect=_fake_sync_position_after_trade
    ):
        result = run_portfolio_cycle(notification_callback=notification_callback)

    summary = _summarize_result(result)
    print(json.dumps({"dry_run_summary": summary}, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({"dry_run_result_path": os.path.join(result.get("cycle_dir") or "", "portfolio_cycle_output.json")}))
    if not args.no_telegram and checks["telegram_configured"]:
        send_telegram_message(
            "<b>Binance DeepSeek Trader dry-run validation completed</b>\n"
            f"<code>{json.dumps(summary, ensure_ascii=False, default=str)[:3500]}</code>"
        )
    return 0 if result.get("slot_results") else 1


if __name__ == "__main__":
    raise SystemExit(main())
