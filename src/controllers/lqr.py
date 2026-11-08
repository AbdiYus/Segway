import numpy as np
import scipy as sci
from scipy.signal import cont2discrete

from ..models.configuration import CONFIG
from ..models.dynamics import MODEL


class LQR:
    def __init__(self, Q: np.ndarray, R: np.ndarray, dt: float = CONFIG.mpc.T) -> None:
        # Model as the controller sees it: torque held constant over dt (zero-order hold)
        n_states, n_inputs = MODEL.B.shape
        Ad, Bd, *_ = cont2discrete(
            (MODEL.A, MODEL.B, np.eye(n_states), np.zeros((n_states, n_inputs))),
            dt,
            method="zoh",
        )

        P = sci.linalg.solve_discrete_are(Ad, Bd, Q, R)
        self.K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)

    def compute(self, state_estimate: np.ndarray) -> np.ndarray:
        return -self.K @ state_estimate

    def reset(self) -> None:
        pass
