# alphas/__init__.py
from tradingbot.alphas.mean_reversion import VwapMeanReversionAlpha
from tradingbot.alphas.momentum import MomentumAlpha


def build_alphas(settings):

    registry = {
        "momentum": MomentumAlpha,
        "mean_reversion": VwapMeanReversionAlpha,
    }

    alphas = []

    for name in settings["strategy"]["alphas"]:
        if name == "ml":
            from tradingbot.alphas.ml_alpha import load_ml_alpha
            alphas.append(load_ml_alpha(settings, **settings["alphas"].get("ml", {})))
        else:
            alphas.append(registry[name](**settings["alphas"].get(name, {})))

    return alphas
