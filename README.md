# TradingBotV1

Minute-level paper trading bot for US stocks/ETFs on Alpaca, structured like
QuantConnect's Algorithm Framework:

```
AlphaModels -> Insights -> PortfolioConstruction -> RiskManagement -> Execution
(momentum,     (symbol,    (equal weight, or       (position caps,   (Alpaca paper,
 VWAP mean-rev, direction,  PPO RL allocator)       daily loss stop,  or simulator
 LightGBM)      confidence)                         flat at close)    in backtests)
```

The backtester and the live bot run the **same** `Algorithm`; only the data
feed and execution model differ.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\pip install -e ".[dev]"
```

This installs the `tradingbot` package (editable) and a `tradingbot` command;
`python -m tradingbot <command>` works too.

`.env` needs `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` (paper account keys).

## Commands

| Command | What it does |
| --- | --- |
| `tradingbot download` | Cache SIP minute bars for the universe to `data/cache/` (incremental) |
| `tradingbot backtest --from 2026-06-01 --to 2026-08-31` | Backtest; writes `reports/backtest/latest/` (metrics, logs, dashboard) |
| `tradingbot train-alpha --until 2026-03-31` | Walk-forward train the LightGBM alpha → `models/lgbm_alpha.*` |
| `tradingbot train-rl --until 2026-03-31` | Train PPO allocator, evaluate on later days vs equal-weight → `models/ppo_allocator.*` |
| `tradingbot live` | Trade one session on Alpaca paper, then exit |
| `tradingbot report` | Graphs from `logs/` → `reports/dashboard.html`, `reports/performance.png` |
| `python -m pytest` | Tests (synthetic data, no API needed) |

Enable the ML alpha by adding `ml` to `strategy.alphas`, and the RL allocator
with `strategy.portfolio: rl`, in `config/settings.yaml` — only once they beat
the baseline out of sample. Always backtest on dates **after** a model's
`--until` date, otherwise results are in-sample.

## Project layout

```
config/settings.yaml   universe, alpha/risk/cost parameters
src/tradingbot/
  cli.py               command-line entry point
  config.py            settings loader, repo ROOT
  live_runner.py       one live paper-trading session
  core/                framework base classes + Insight
  alphas/              momentum, VWAP mean-reversion, LightGBM alpha, shared signals
  portfolio/           equal-weight and RL portfolio construction
  risk/                position caps, daily loss stop, flatten before close
  execution/           order sizing; Alpaca and simulated execution
  data/                Alpaca client, history cache, live bar stream
  backtest/            event-driven engine + cash/position ledger
  ml/                  LightGBM walk-forward training
  rl/                  Gymnasium env, features, PPO training
  monitoring/          CSV trade logger + dashboard/report
scripts/               Task Scheduler + hourly log commit (PowerShell)
tests/                 pytest suite (synthetic data, no API needed)
data/cache/            downloaded minute bars (git-ignored)
models/, reports/      trained models, backtest output (git-ignored)
logs/                  live trade/equity logs (committed)
```

## Logs (committed to GitHub)

* `logs/trades/YYYY-MM-DD.csv` — every fill: time, symbol, side, qty, price, alpha source, confidence, target weight, realized P&L
* `logs/equity/YYYY-MM-DD.csv` — equity, cash, exposure, positions, SPY price every 5 minutes
* `logs/daily_summary.csv` — daily return, cumulative return, Sharpe to date, max drawdown, trades, win rate

## Running unattended on this PC

```powershell
powershell -ExecutionPolicy Bypass -File scripts\register_tasks.ps1
```

Creates two scheduled tasks (UK times; the bot waits on Alpaca's market clock):

* `TradingBotV1-Live` — weekdays 13:15 and at logon; runs `python -m tradingbot live`
* `TradingBotV1-CommitLogs` — weekdays hourly 14:00–22:00; `scripts/commit_logs.ps1` commits and pushes **only** `logs/`

Keep the PC awake 13:15–22:00 on weekdays (Settings → Power → never sleep when
plugged in) and set Windows Update active hours to cover that window.
Runtime logs go to `runtime_logs/` (not committed).
