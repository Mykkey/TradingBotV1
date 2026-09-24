# sim_exec.py
# In backtests, orders are returned to the engine, which fills them at the
# next bar's open with slippage (see backtest/engine.py).
from execution.base import TargetExecution


class SimExecution(TargetExecution):
    pass
