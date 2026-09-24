# framework.py
# QuantConnect-style Algorithm Framework:
#   AlphaModel -> PortfolioConstructionModel -> RiskManagementModel -> ExecutionModel
# The same Algorithm runs in backtests and live; only the data feed and
# execution model differ.
from abc import ABC, abstractmethod


class AlphaModel(ABC):

    name = "alpha"

    @abstractmethod
    def update(self, now, history):
        """history: dict symbol -> DataFrame of minute bars up to `now`.
        Returns a list of Insights."""


class PortfolioConstructionModel(ABC):

    @abstractmethod
    def create_targets(self, now, insights, state):
        """Returns dict symbol -> target weight of equity (negative = short)."""


class RiskManagementModel(ABC):

    @abstractmethod
    def manage_risk(self, now, targets, state):
        """Returns adjusted dict symbol -> target weight."""


class ExecutionModel(ABC):

    @abstractmethod
    def execute(self, now, targets, state, prices, context):
        """Moves holdings toward targets. Returns list of submitted orders."""


class Algorithm:

    def __init__(self, alphas, portfolio_model, risk_model, execution_model):
        self.alphas = alphas
        self.portfolio_model = portfolio_model
        self.risk_model = risk_model
        self.execution_model = execution_model
        self.active_insights = {}

    def step(self, now, history, state, prices):
        """state: dict with equity, cash, positions (symbol -> qty),
        day_start_equity, minutes_to_close."""

        for alpha in self.alphas:
            for insight in alpha.update(now, history):
                self.active_insights[(insight.source, insight.symbol)] = insight

        self.active_insights = {
            key: insight
            for key, insight in self.active_insights.items()
            if insight.is_active(now)
        }

        insights = list(self.active_insights.values())

        # Some portfolio models (e.g. the RL allocator) need raw data too.
        state = {**state, "history": history, "prices": prices}

        targets = self.portfolio_model.create_targets(now, insights, state)
        targets = self.risk_model.manage_risk(now, targets, state)

        context = {
            symbol: [i for i in insights if i.symbol == symbol]
            for symbol in set(targets) | set(state["positions"])
        }

        return self.execution_model.execute(now, targets, state, prices, context)
