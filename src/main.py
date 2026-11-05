import numpy as np
from .simulator import Segway
from .controllers import ControllerType, LQR, LinearMPC
from .observers import ObserverType, KalmanFilter, UnscentedKalmanFilter


# ================================ Tuning =========================================

# [s, ds, theta, dtheta, psi, dpsi]

Q_LQR = np.diag([1.0, 0.1, 1.0, 0.1, 100.0, 1.0])
R_LQR = np.diag([0.01, 0.01])

Q_LINEAR_MPC = np.diag([1.0, 0.1, 1.0, 0.1, 100.0, 1.0])
R_LINEAR_MPC = np.diag([0.01, 0.01])

# ================================ Setup =========================================

controllers = {
    ControllerType.LQR: LQR(Q=Q_LQR, R=R_LQR),
    ControllerType.LINEAR_MPC: LinearMPC(Q=Q_LINEAR_MPC, R=R_LINEAR_MPC),
}


observers = {}
