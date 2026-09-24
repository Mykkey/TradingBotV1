# test_framework.py
from datetime import datetime, timedelta, timezone

import pytest

from alphas.mean_reversion import VwapMeanReversionAlpha
from alphas.momentum import MomentumAlpha
from core.insight import Direction, Insight
from execution.base import orders_for_targets
from portfolio.equal_weight import EqualWeightPortfolio
from risk.limits import RiskLimits
from tests.conftest import make_bars


NOW = datetime(2025, 3, 3, 15, 0, tzinfo=timezone.utc)


def insight(symbol, direction, confidence=1.0, source="test"):
    return Insight(symbol, direction, NOW, timedelta(minutes=15), confidence, source=source)


def state(**overrides):
    base = {
        "equity": 100_000, "cash": 100_000, "positions": {},
        "day_start_equity": 100_000, "minutes_to_close": 120,
    }
    return {**base, **overrides}


def test_insight_expiry():
    i = insight("AAA", Direction.UP)
    assert i.is_active(NOW + timedelta(minutes=14))
    assert not i.is_active(NOW + timedelta(minutes=15))


def test_equal_weight_long_only_and_exits_held():
    insights = [
        insight("AAA", Direction.UP),
        insight("BBB", Direction.UP),
        insight("CCC", Direction.DOWN),
    ]
    targets = EqualWeightPortfolio().create_targets(NOW, insights, state(positions={"DDD": 5}))

    assert targets == {"AAA": 0.5, "BBB": 0.5, "DDD": 0.0}


def test_risk_caps_and_flattens():
    risk = RiskLimits(max_position_pct=0.1, max_gross_exposure=0.25)

    capped = risk.manage_risk(NOW, {"A": 0.5, "B": 0.5, "C": 0.5}, state())
    assert all(w == pytest.approx(0.25 / 3) for w in capped.values())

    near_close = risk.manage_risk(NOW, {"A": 0.1}, state(minutes_to_close=3, positions={"B": 1}))
    assert near_close == {"A": 0.0, "B": 0.0}


def test_risk_daily_loss_kill_switch():
    risk = RiskLimits(daily_loss_limit_pct=0.02)

    halted = risk.manage_risk(NOW, {"A": 0.1}, state(equity=97_000))
    assert halted == {"A": 0.0}

    # Stays halted for the rest of the day even if equity recovers.
    assert risk.manage_risk(NOW, {"A": 0.1}, state()) == {"A": 0.0}


def test_orders_for_targets():
    orders = orders_for_targets(
        NOW, {"AAA": 0.1, "BBB": 0.0, "CCC": 0.1001},
        state(positions={"BBB": 10, "CCC": 100}),
        {"AAA": 100.0, "BBB": 50.0, "CCC": 100.0}, {}, min_trade_value=200,
    )

    # Sell first; CCC's 1-share rebalance is below min_trade_value.
    assert [(o["symbol"], o["side"], o["qty"]) for o in orders] == [
        ("BBB", "sell", 10),
        ("AAA", "buy", 100),
    ]


def test_alphas_emit_insights():
    bars = make_bars(days=1, seed=5)
    history = {"AAA": bars.iloc[:200]}
    now = bars.index[199]

    for alpha in (MomentumAlpha(threshold=1e-9), VwapMeanReversionAlpha(vwap_z_entry=1e-9)):
        insights = alpha.update(now, history)
        assert len(insights) == 1
        assert insights[0].source == alpha.name
        assert 0 <= insights[0].confidence <= 1
