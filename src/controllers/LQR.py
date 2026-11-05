import numpy as np
import scipy as sci
from ..models.dynamics import MODEL


class LQR:
    def __init__(self, Q: np.ndarray, R: np.ndarray) -> None:
        P = sci.linalg.solve_continuous_are(MODEL.A, MODEL.B, Q, R)
        self.K = np.linalg.inv(R) @ (MODEL.B.T @ P)

    def compute(self, state_estiamtion: np.ndarray):
        return -self.K @ state_estiamtion

    def reset(self) -> None:
        pass
