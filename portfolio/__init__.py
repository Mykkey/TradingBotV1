# portfolio/__init__.py
from portfolio.equal_weight import EqualWeightPortfolio


def build_portfolio_model(settings):
    kind = settings["strategy"]["portfolio"]

    if kind == "equal_weight":
        return EqualWeightPortfolio()

    if kind == "rl":
        from portfolio.rl_allocator import RLAllocator
        return RLAllocator(settings)

    raise ValueError(f"Unknown portfolio model: {kind}")
