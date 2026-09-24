# limits.py
import logging

from core.framework import RiskManagementModel


log = logging.getLogger(__name__)


class RiskLimits(RiskManagementModel):

    def __init__(
        self,
        max_position_pct=0.10,
        max_gross_exposure=1.0,
        daily_loss_limit_pct=0.02,
        flatten_minutes_before_close=5,
        **_,
    ):
        self.max_position = max_position_pct
        self.max_gross = max_gross_exposure
        self.daily_loss_limit = daily_loss_limit_pct
        self.flatten_minutes = flatten_minutes_before_close
        self.halted_day = None

    def manage_risk(self, now, targets, state):
        today = now.date()

        loss = 1 - state["equity"] / state["day_start_equity"]

        if loss >= self.daily_loss_limit and self.halted_day != today:
            log.warning("Daily loss limit hit (%.2f%%): flattening for the day", loss * 100)
            self.halted_day = today

        if self.halted_day == today or state["minutes_to_close"] <= self.flatten_minutes:
            return {symbol: 0.0 for symbol in set(targets) | set(state["positions"])}

        capped = {
            symbol: max(-self.max_position, min(self.max_position, weight))
            for symbol, weight in targets.items()
        }

        gross = sum(abs(w) for w in capped.values())

        if gross > self.max_gross:
            capped = {s: w * self.max_gross / gross for s, w in capped.items()}

        return capped
