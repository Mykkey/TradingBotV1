# report.py
# Turn the CSV logs (live or backtest) into graphs:
#   <out_dir>/dashboard.html  interactive Plotly dashboard (works offline)
#   <out_dir>/performance.png static equity + drawdown chart
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from monitoring.trade_logger import load_equity, load_trades


def _filter(df, start, end):
    if df.empty:
        return df

    dates = df["timestamp"].dt.date

    if start:
        df = df[dates >= pd.Timestamp(start).date()]
        dates = df["timestamp"].dt.date

    if end:
        df = df[dates <= pd.Timestamp(end).date()]

    return df


def build_report(log_dir="logs", out_dir="reports", start=None, end=None):
    equity = _filter(load_equity(log_dir), start, end)
    trades = _filter(load_trades(log_dir), start, end)

    if equity.empty:
        raise SystemExit(f"No equity logs found in {log_dir}/equity")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = equity["timestamp"].dt.tz_localize(None)
    curve = equity["equity"]
    drawdown = (curve / curve.cummax() - 1) * 100
    growth = (curve / curve.iloc[0] - 1) * 100

    bench = equity["benchmark_price"]
    bench_growth = (bench / bench.dropna().iloc[0] - 1) * 100 if bench.notna().any() else None

    daily = equity.set_index("timestamp")["equity"].groupby(lambda t: t.date()).last()
    daily_returns = daily.pct_change().dropna() * 100

    fig = make_subplots(
        rows=5, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{}, {}],
            [{}, {}],
            [{"colspan": 2}, None],
        ],
        subplot_titles=(
            "Return vs benchmark (%)",
            "Drawdown (%)",
            "Daily returns (%)",
            "Avg gross exposure by time of day",
            "Realized P&L by symbol ($)",
            "Realized P&L by alpha ($)",
            "Trades per day",
        ),
        vertical_spacing=0.07,
    )

    # Rangebreaks hide nights and weekends so the curve is continuous.
    breaks = [dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")]

    fig.add_trace(go.Scatter(x=ts, y=growth, name="Bot", line=dict(width=2)), 1, 1)

    if bench_growth is not None:
        fig.add_trace(
            go.Scatter(x=ts, y=bench_growth, name="Benchmark", line=dict(width=1, dash="dot")),
            1, 1,
        )

    fig.add_trace(
        go.Scatter(x=ts, y=drawdown, name="Drawdown", fill="tozeroy", showlegend=False),
        2, 1,
    )
    fig.update_xaxes(rangebreaks=breaks, row=1, col=1)
    fig.update_xaxes(rangebreaks=breaks, row=2, col=1)

    fig.add_trace(go.Histogram(x=daily_returns, nbinsx=30, showlegend=False), 3, 1)

    minute_of_day = equity["timestamp"].dt.strftime("%H:%M")
    exposure = equity.groupby(minute_of_day)["gross_exposure"].mean()
    fig.add_trace(go.Scatter(x=exposure.index, y=exposure.values, showlegend=False), 3, 2)

    if not trades.empty:
        by_symbol = trades.groupby("symbol")["realized_pnl"].sum().sort_values()
        fig.add_trace(
            go.Bar(x=by_symbol.values, y=by_symbol.index, orientation="h", showlegend=False),
            4, 1,
        )

        by_alpha = trades.groupby(trades["sources"].fillna("none"))["realized_pnl"].sum()
        fig.add_trace(go.Bar(x=by_alpha.index, y=by_alpha.values, showlegend=False), 4, 2)

        per_day = trades.groupby(trades["timestamp"].dt.date).size()
        fig.add_trace(
            go.Bar(x=[str(d) for d in per_day.index], y=per_day.values, showlegend=False),
            5, 1,
        )

    fig.update_layout(
        height=1800,
        title="TradingBotV1 performance",
        template="plotly_white",
        legend=dict(orientation="h", y=1.02),
    )

    html_path = out / "dashboard.html"
    fig.write_html(html_path, include_plotlyjs=True)

    png, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    x = range(len(ts))
    ax1.plot(x, growth, label="Bot")

    if bench_growth is not None:
        ax1.plot(x, bench_growth, label="Benchmark", linestyle=":")

    ax1.set_ylabel("Return (%)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.fill_between(x, drawdown, 0, alpha=0.5, color="tab:red")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.3)

    day_starts = equity["timestamp"].dt.date.ne(equity["timestamp"].dt.date.shift()).to_numpy().nonzero()[0]
    step = max(1, len(day_starts) // 10)
    ax2.set_xticks(day_starts[::step])
    ax2.set_xticklabels([str(equity["timestamp"].iloc[i].date()) for i in day_starts[::step]],
                        rotation=30, ha="right")

    png.tight_layout()
    png_path = out / "performance.png"
    png.savefig(png_path, dpi=110)
    plt.close(png)

    return html_path, png_path
