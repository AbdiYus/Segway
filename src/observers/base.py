from typing import Protocol
from enum import Enum
import numpy as np


class Observer(Protocol):
    @property
    def states(self) -> np.ndarray: ...
    def reset_to(self, states: np.ndarray) -> None: ...
    def update(self, measurement: np.ndarray, input: np.ndarray) -> None: ...
    def predict(self, input: np.ndarray) -> None: ...


class ObserverType(Enum):
    KALMAN_FILTER = 0
    UNSCENTED_KALMAN_FILTER = 1
