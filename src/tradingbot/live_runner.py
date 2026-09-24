# live_runner.py
# Runs one trading session against Alpaca paper trading, then exits.
# Started each weekday by Windows Task Scheduler (see scripts/register_tasks.ps1).
# Safe to restart mid-session: all state is re-read from Alpaca.
import logging
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from tradingbot.alphas import build_alphas
from tradingbot.core.framework import Algorithm
from tradingbot.data.live import BarBuffer
from tradingbot.execution.alpaca_exec import AlpacaExecution, with_retries
from tradingbot.monitoring.trade_logger import TradeLogger
from tradingbot.portfolio import build_portfolio_model
from tradingbot.data.history import fetch_daily_closes
from tradingbot.risk import build_risk_model


log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


def sleep_until_next_minute(offset_seconds=5):
    now = time.time()
    time.sleep(60 - now % 60 + offset_seconds)


def log_account_equity(logger, timestamp, account, benchmark_price):
    values = account["market_values"].values()
    equity = account["equity"]

    logger.log_equity(
        timestamp,
        equity=round(equity, 2),
        cash=round(account["cash"], 2),
        gross_exposure=round(sum(abs(v) for v in values) / equity, 4),
        net_exposure=round(sum(values) / equity, 4),
        n_positions=len(account["positions"]),
        benchmark_price=benchmark_price,
    )


def run_live(settings):
    logger = TradeLogger(settings["logging"]["log_dir"])
    execution = AlpacaExecution(logger, settings["risk"]["min_trade_value"],
                                settings["risk"].get("rebalance_band", 0.0))

    clock = with_retries(execution.client.get_clock)

    if not clock.is_open:
        wait = (clock.next_open - clock.timestamp).total_seconds()

        if wait > 3 * 3600:
            log.info("Market closed; next open %s. Exiting.", clock.next_open)
            return

        log.info("Waiting %.0f min for the open", wait / 60)
        time.sleep(max(0, wait))

    universe = settings["universe"]
    benchmark = settings["benchmark"]
    interval = settings["strategy"]["decision_interval_min"]
    flatten_at_close = settings["risk"].get("flatten_at_close", True)
    flatten_minutes = settings["risk"]["flatten_minutes_before_close"] if flatten_at_close else 0
    first_decision = settings["strategy"].get("first_decision_min", 0)
    snapshot_every = settings["logging"]["equity_snapshot_min"]

    trend = settings["risk"].get("trend_filter") or {}
    daily_closes = (
        with_retries(fetch_daily_closes, trend["symbol"], feed=settings["data"]["history_feed"])
        if trend.get("enabled") else None
    )

    algorithm = Algorithm(
        alphas=build_alphas(settings),
        portfolio_model=build_portfolio_model(settings),
        risk_model=build_risk_model(settings, daily_closes),
        execution_model=execution,
    )

    bars = BarBuffer(universe, feed=settings["data"]["live_feed"])
    bars.warm_up()
    bars.start()
    execution.start_fill_stream()

    day_start_equity = execution.account_state()["last_equity"]
    session_day = datetime.now(ET).date()
    log.info("Session %s started; previous close equity $%.2f", session_day, day_start_equity)

    try:
        while True:
            sleep_until_next_minute()

            clock = with_retries(execution.client.get_clock)

            if not clock.is_open:
                break

            now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            now_et = now.astimezone(ET)
            minutes_open = (now_et.hour - 9) * 60 + now_et.minute - 30
            minutes_to_close = int((clock.next_close - clock.timestamp).total_seconds() // 60)

            history, prices = bars.snapshot()
            account = execution.account_state()

            if minutes_open % snapshot_every == 0:
                log_account_equity(logger, now, account, prices.get(benchmark))

            if minutes_to_close <= 2:
                if flatten_at_close and account["positions"]:
                    log.info("Closing all positions before the bell")
                    execution.close_all()
                continue

            scheduled = minutes_open % interval == 0 and minutes_open >= first_decision

            if not scheduled and minutes_to_close > flatten_minutes:
                continue

            state = {
                "equity": account["equity"],
                "cash": account["cash"],
                "positions": account["positions"],
                "day_start_equity": day_start_equity,
                "minutes_to_close": minutes_to_close,
            }

            try:
                orders = algorithm.step(now, history, state, prices)
                log.info("%s  equity $%.2f  orders %d", now_et.strftime("%H:%M"),
                         account["equity"], len(orders))
            except Exception:
                log.exception("Strategy step failed; skipping this tick")

    finally:
        bars.stop()
        time.sleep(5)   # let the last fill updates arrive
        execution.stop()

        final = execution.account_state()
        log_account_equity(logger, datetime.now(timezone.utc), final,
                           bars.snapshot()[1].get(benchmark))
        summary = logger.write_daily_summary(session_day, day_start_equity)
        log.info("Session finished: %s", summary)
