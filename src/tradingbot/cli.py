# cli.py  (run as `tradingbot <command>` or `python -m tradingbot <command>`)
#   download                  cache historical minute bars
#   backtest [--from --to]    run the strategy on cached data
#   train-alpha [--until]     walk-forward train the LightGBM alpha
#   train-rl [--until]        train + evaluate the PPO allocator
#   live                      trade one session on Alpaca paper
#   report [--from --to]      graphs from logs/ (or --logs DIR)
#   demo                      original single-trade example
import argparse
import json
import logging
import sys
from datetime import date

from tradingbot.config import ROOT, load_settings


def setup_logging(name):
    log_dir = ROOT / "runtime_logs"
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / f"{name}-{date.today()}.log"),
        ],
    )
    # Per-trade INFO lines from the simulator ledger are too chatty for backtests.
    logging.getLogger("tradingbot.backtest.ledger").setLevel(logging.WARNING)


def cmd_download(settings, args):
    from tradingbot.data.history import update_cache

    update_cache(
        settings["universe"],
        ROOT / settings["data"]["cache_dir"],
        settings["data"]["history_start"],
        settings["data"]["history_feed"],
    )


def cmd_backtest(settings, args):
    from tradingbot.alphas import build_alphas
    from tradingbot.backtest.engine import Backtest
    from tradingbot.core.framework import Algorithm
    from tradingbot.data.history import load_bars
    from tradingbot.execution.sim_exec import SimExecution
    from tradingbot.monitoring.report import build_report
    from tradingbot.portfolio import build_portfolio_model
    from tradingbot.risk.limits import RiskLimits

    bars = load_bars(settings["universe"], ROOT / settings["data"]["cache_dir"],
                     args.start, args.end)

    if not bars:
        raise SystemExit("No cached data. Run `python -m tradingbot download` first.")

    algorithm = Algorithm(
        alphas=build_alphas(settings),
        portfolio_model=build_portfolio_model(settings),
        risk_model=RiskLimits(**settings["risk"]),
        execution_model=SimExecution(settings["risk"]["min_trade_value"],
                                     settings["risk"].get("rebalance_band", 0.0)),
    )

    out_dir = ROOT / "reports" / "backtest" / args.name
    metrics = Backtest(algorithm, bars, settings, out_dir).run()

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    html, png = build_report(out_dir, out_dir)

    print(json.dumps(metrics, indent=2))
    print(f"Dashboard: {html}\nChart:     {png}")


def _load_all_bars(settings, args):
    from tradingbot.data.history import load_bars

    bars = load_bars(settings["universe"], ROOT / settings["data"]["cache_dir"],
                     getattr(args, "start", None))

    if not bars:
        raise SystemExit("No cached data. Run `python -m tradingbot download` first.")

    return bars


def cmd_train_alpha(settings, args):
    from tradingbot.ml.train_alpha import train_alpha

    meta = train_alpha(_load_all_bars(settings, args), settings,
                       ROOT / settings["strategy"]["ml_model_path"], args.until)

    print(json.dumps({k: meta[k] for k in ("trained_until", "walk_forward_mean")}, indent=2))
    print("Add \"ml\" to strategy.alphas in config/settings.yaml to use it.")


def cmd_train_rl(settings, args):
    from tradingbot.rl.train import train_rl

    if args.until:
        settings["rl"]["train_until"] = args.until
    if args.timesteps:
        settings["rl"]["total_timesteps"] = args.timesteps

    meta = train_rl(_load_all_bars(settings, args), settings,
                    ROOT / settings["strategy"]["rl_model_path"])

    print(json.dumps({k: meta[k] for k in ("trained_until", "rl_heldout", "baseline_heldout")},
                     indent=2))
    print("Set strategy.portfolio: rl in config/settings.yaml only if RL beats the baseline.")


def cmd_live(settings, args):
    from tradingbot.live_runner import run_live

    run_live(settings)


def cmd_report(settings, args):
    from tradingbot.monitoring.report import build_report

    log_dir = ROOT / (args.logs or settings["logging"]["log_dir"])
    html, png = build_report(log_dir, ROOT / "reports", args.start, args.end)

    print(f"Dashboard: {html}\nChart:     {png}")


def cmd_demo(settings, args):
    from tradingbot.backtest.ledger import Ledger
    from tradingbot.data.market_data import get_stock_price

    ledger = Ledger(10_000)
    price = get_stock_price("AAPL")
    print(f"AAPL price: ${price:.2f}")

    ledger.buy("AAPL", 10, price)
    ledger.print_status()
    print(f"Profit/Loss at +$10: ${ledger.get_profit_loss({'AAPL': price + 10}):.2f}")
    ledger.sell("AAPL", 10, price)
    ledger.print_status()


def main():
    parser = argparse.ArgumentParser(description="TradingBotV1")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download")

    backtest = sub.add_parser("backtest")
    backtest.add_argument("--from", dest="start")
    backtest.add_argument("--to", dest="end")
    backtest.add_argument("--name", default="latest")

    train_alpha = sub.add_parser("train-alpha")
    train_alpha.add_argument("--from", dest="start")
    train_alpha.add_argument("--until", help="last date used for the final model")

    train_rl = sub.add_parser("train-rl")
    train_rl.add_argument("--from", dest="start")
    train_rl.add_argument("--until", help="train on days up to here, evaluate after")
    train_rl.add_argument("--timesteps", type=int)

    sub.add_parser("live")

    report = sub.add_parser("report")
    report.add_argument("--from", dest="start")
    report.add_argument("--to", dest="end")
    report.add_argument("--logs", help="log directory (default: logs/)")

    sub.add_parser("demo")

    args = parser.parse_args()
    setup_logging(args.command)

    commands = {
        "download": cmd_download,
        "backtest": cmd_backtest,
        "train-alpha": cmd_train_alpha,
        "train-rl": cmd_train_rl,
        "live": cmd_live,
        "report": cmd_report,
        "demo": cmd_demo,
    }
    commands[args.command](load_settings(), args)


if __name__ == "__main__":
    main()
