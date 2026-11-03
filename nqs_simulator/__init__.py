"""神经态库量子线路模拟器初版。

波函数、损失和局部门更新沿用已复算通过的 Wang & Fu 实现（arXiv:2606.08707v1）。
默认只保存配置、中途变更和结果，不长期保存网络权重。
"""

from .config import Config, TuningConfig
from .circuits import Circuit, Gate, Update, make_hczch
from .engine import simulate, SimulationResult
from .model import Statebank

__all__ = [
    "Config",
    "TuningConfig",
    "Circuit",
    "Gate",
    "Update",
    "make_hczch",
    "simulate",
    "SimulationResult",
    "Statebank",
]
