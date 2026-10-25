"""
Unscented Kalman filter for the segway.

Uses the same model as the MPC (Models/segway_dynamics.py -> MODEL.plant),
so the filter, the MPC and the simulated plant share one description of
the robot.

    x = [s, ds, theta, dtheta, psi, dpsi]
    u = [tau_l, tau_r]
    z = [a_x, a_z, gyro_pitch, gyro_yaw, enc_l, enc_r]   (Models/segway_sensors.py)

Per sample:
    ukf.update(z, u_prev)   # correct with the sensors
    u = MPC(ukf.x)          # control from the estimate
    ukf.predict(u)          # prior for the next sample
"""

import casadi as ca
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

from Control_System.Models import segway_sensors as sensors
from Control_System.Models.segway_configuration import CONFIG
from Control_System.Models.segway_dynamics import MODEL

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


class SegwayUKF:
    def __init__(
        self,
        dt=CONFIG.mpc.T,
        x0=None,
        settings=CONFIG.ukf,
        sensor_settings=CONFIG.sensors,
    ):
        n = N_STATES
        self._step = make_discrete_step(dt, settings.rk4_substeps)

        points = MerweScaledSigmaPoints(
            n=n, alpha=settings.alpha, beta=settings.beta, kappa=settings.kappa
        )
        self.ukf = UnscentedKalmanFilter(
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

    def predict(self, u):
        """Prior for the next sample, given the torque u applied over it."""
        self.ukf.predict(u=np.asarray(u, float))
        self._prior_is_fresh = True

    def update(self, z, u, which=("imu", "encoders")):
        """Correct with measurement z.

        u     : torque being applied when z was measured (the accelerometer
                depends on it)
        which : sensors that reported; z holds only their rows, in the
                order imu, encoders. Lets the IMU and encoders run at
                different rates.
        """
        rows = np.r_[tuple(sensors.SENSOR_ROWS[w] for w in which)]
        R = self.R_full[np.ix_(rows, rows)]

        def hx(x, u):
            return sensors.h(x, u)[rows]

        # FilterPy re-uses the sigma points from predict(). If a second update
        # follows without a predict, redraw them around the current estimate.
        if not self._prior_is_fresh:
            self.ukf.sigmas_f = self.ukf.points_fn.sigma_points(self.ukf.x, self.ukf.P)

        self.ukf.update(np.asarray(z, float), R=R, hx=hx, u=np.asarray(u, float))
        self._prior_is_fresh = False
        self.nis = float(self.ukf.y @ self.ukf.SI @ self.ukf.y)

    @property
    def x(self):
        """State estimate, same ordering as MODEL.x."""
        return self.ukf.x.copy()

    @property
    def P(self):
        """Covariance of the estimate."""
        return self.ukf.P.copy()

