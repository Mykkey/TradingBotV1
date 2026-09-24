# risk/__init__.py
from tradingbot.risk.limits import RiskLimits
from tradingbot.risk.trend_gate import TrendGate


def build_risk_model(settings, daily_closes=None):
    """daily_closes: Series of the trend-filter symbol's daily closes (ET dates);
    required when risk.trend_filter.enabled is true."""
    risk = dict(settings["risk"])
    trend = risk.pop("trend_filter", None) or {}
    gate = None

    if trend.get("enabled"):
        if daily_closes is None:
            raise ValueError("trend_filter is enabled but no daily closes were provided")
        gate = TrendGate(daily_closes, trend.get("days", 100))

    return RiskLimits(trend_gate=gate, **risk)
