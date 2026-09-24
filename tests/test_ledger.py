# test_ledger.py
import pytest

from tradingbot.backtest.ledger import Ledger


def test_buy_sell_realized_pnl():
    p = Ledger(10_000)

    assert p.buy("AAPL", 10, 100)
    assert p.buy("AAPL", 10, 110)
    assert p.avg_cost["AAPL"] == pytest.approx(105)

    assert p.sell("AAPL", 20, 120)
    assert p.realized_pnl == pytest.approx(300)
    assert p.cash == pytest.approx(10_300)
    assert p.positions == {}
    assert len(p.trades) == 3


def test_rejects_overspend_and_oversell():
    p = Ledger(1_000)

    assert not p.buy("AAPL", 20, 100)
    assert not p.sell("AAPL", 1, 100)

    p.buy("AAPL", 5, 100)
    assert not p.sell("AAPL", 6, 100)


def test_commission_and_values():
    p = Ledger(10_000, commission_per_share=0.01)
    p.buy("AAPL", 100, 50)

    assert p.cash == pytest.approx(10_000 - 5_000 - 1)
    assert p.get_value({"AAPL": 60}) == pytest.approx(10_999)
    assert p.get_unrealized_pnl({"AAPL": 60}) == pytest.approx(1_000)
    assert p.get_profit_loss({"AAPL": 60}) == pytest.approx(999)
