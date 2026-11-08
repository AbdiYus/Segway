from typing import Protocol
from enum import Enum
import numpy as np


class Controller(Protocol):
    def compute(self, state_estimate: np.ndarray) -> np.ndarray: ...
    def reset(self) -> None: ...


class ControllerType(Enum):
    LINEAR_MPC = 0
    LQR = 1
    NONLINEAR_MPC = 2
