"""
Unscented Kalman filter for the segway.

Uses the same model as the MPC (models/dynamics.py -> MODEL.plant),
so the filter, the MPC and the simulated plant share one description of
the robot.

    x = [s, ds, theta, dtheta, psi, dpsi]
    u = [tau_l, tau_r]
    z = [a_x, a_z, gyro_pitch, gyro_yaw, enc_l, enc_r]   (models/sensors.py)

Per sample (driven by simulator/control_loop.py):
    ukf.predict(u_prev)       # prior, with the torque applied since the last sample
    ukf.update(z, u_prev)     # correct with the sensors
    u = controller(ukf.states)
"""

import casadi as ca
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as FilterpyUKF

from ..models import sensors
from ..models.configuration import CONFIG
from ..models.dynamics import MODEL

N_STATES = 6


def make_discrete_step(dt, substeps):
    """x_{k+1} = f(x_k, u_k): RK4 on MODEL.plant with u held over dt.
    Built once as a CasADi function, so each call is fast."""
    x, u = MODEL.x, MODEL.u
    h = dt / substeps
    x_next = x
    for _ in range(substeps):
        k1 = MODEL.plant(x_next, u)
        k2 = MODEL.plant(x_next + h / 2 * k1, u)
        k3 = MODEL.plant(x_next + h / 2 * k2, u)
        k4 = MODEL.plant(x_next + h * k3, u)
        x_next = x_next + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ca.Function("f_discrete", [x, u], [x_next])


class UnscentedKalmanFilter:
    def __init__(
        self,
        dt=CONFIG.mpc.T,
        x0=None,
        settings=CONFIG.ukf,
        sensor_settings=CONFIG.sensors,
    ):
        n = N_STATES
        self.settings = settings
        self._step = make_discrete_step(dt, settings.rk4_substeps)

        points = MerweScaledSigmaPoints(
            n=n, alpha=settings.alpha, beta=settings.beta, kappa=settings.kappa
        )
        self.ukf = FilterpyUKF(
            dim_x=n,
            dim_z=sensors.N_MEAS,
            dt=dt,
            fx=self._fx,
            hx=sensors.h,
            points=points,
        )

        # Q: one block per (position, velocity) pair: (s, ds), (theta, dtheta), (psi, dpsi)
        Q = np.zeros((n, n))
        Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=settings.accel_std_s**2)
        Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=settings.accel_std_theta**2)
        Q[4:6, 4:6] = Q_discrete_white_noise(2, dt=dt, var=settings.accel_std_psi**2)
        self.ukf.Q = Q

        # R: independent sensors. Encoder error = quantisation, Delta^2 / 12
        var_enc = sensors.encoder_resolution(sensor_settings) ** 2 / 12
        self.R_full = np.diag(
            [
                sensor_settings.accel_std**2,
                sensor_settings.accel_std**2,
                sensor_settings.gyro_std**2,
                sensor_settings.gyro_std**2,
                var_enc,
                var_enc,
            ]
        )
        self.ukf.R = self.R_full

        self.ukf.x = np.zeros(n) if x0 is None else np.asarray(x0, float)
        self.ukf.P = np.diag(np.square(settings.initial_std))

        self._prior_is_fresh = False
        self.nis = np.nan  # normalised innovation squared, last update

    def _fx(self, x, dt, u):
        return np.array(self._step(x, u)).ravel()

    def predict(self, torque):
        """Prior for the next sample, given the torque applied over it."""
        self.ukf.predict(u=np.asarray(torque, float))
        self._prior_is_fresh = True

    def update(self, measurement, torque, which=("imu", "encoders")):
        """Correct with a measurement.

        torque : torque being applied when the measurement was taken (the
                 accelerometer depends on it)
        which  : sensors that reported; the measurement holds only their
                 rows, in the order imu, encoders. Lets the IMU and encoders
                 run at different rates.
        """
        rows = np.r_[tuple(sensors.SENSOR_ROWS[w] for w in which)]
        R = self.R_full[np.ix_(rows, rows)]

        def hx(x, u):
            return sensors.h(x, u)[rows]

        # FilterPy reuses the sigma points from predict(). If a second update
        # follows without a predict, redraw them around the current estimate.
        if not self._prior_is_fresh:
            self.ukf.sigmas_f = self.ukf.points_fn.sigma_points(self.ukf.x, self.ukf.P)

        self.ukf.update(
            np.asarray(measurement, float), R=R, hx=hx, u=np.asarray(torque, float)
        )
        self._prior_is_fresh = False
        self.nis = float(self.ukf.y @ self.ukf.SI @ self.ukf.y)

    def reset_to(self, states):
        """Restart the filter from this estimate, with the initial uncertainty."""
        self.ukf.x = np.asarray(states, float).copy()
        self.ukf.P = np.diag(np.square(self.settings.initial_std))
        self._prior_is_fresh = False

    @property
    def states(self):
        """State estimate, same ordering as MODEL.x."""
        return self.ukf.x.copy()

    @property
    def P(self):
        """Covariance of the estimate."""
        return self.ukf.P.copy()
