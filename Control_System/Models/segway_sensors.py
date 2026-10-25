"""
Sensor models for the segway: IMU + wheel encoders.

    h(x, u)       what the sensors SHOULD read in state x  (used by the UKF)
    measure(...)  what the sensors DO read: h + noise + quantisation
                  (used by the simulation in place of real hardware)

Measurement vector:
    z = [a_x, a_z, gyro_pitch, gyro_yaw, enc_l, enc_r]

State and input follow Models/segway_dynamics.py:
    x = [s, ds, theta, dtheta, psi, dpsi],   u = [tau_l, tau_r]
"""

import numpy as np

from .segway_configuration import CONFIG
from .segway_dynamics import MODEL

S, DS, THETA, DTHETA, PSI, DPSI = range(6)
G = 9.81

# Which rows of z belong to which sensor (for updates with only some sensors)
SENSOR_ROWS = {"imu": slice(0, 4), "encoders": slice(4, 6)}
N_MEAS = 6


def state_derivative(x, u):
    """x_dot from the Lagrange model, as a flat numpy array."""
    return np.array(MODEL.plant(x, u)).ravel()


def h_imu(x, u, sensors=CONFIG.sensors):
    """IMU model -> [a_x, a_z, gyro_pitch, gyro_yaw].

    IMU axes, fixed to the beam:
        x  forward, perpendicular to the beam
        z  along the beam, up when upright
        y  along the axle (pitch axis)

    The accelerometer measures SPECIFIC FORCE: the IMU's own acceleration
    minus gravity, in the tilted IMU axes. The IMU's acceleration depends on
    s_ddot and psi_ddot, which come from the model and therefore depend on u.

    SIGN CHECK: at rest and leaning forward, a_x = -g sin(psi) with these
    axes. Flip a_x if the real IMU is mounted the other way.
    """
    psi, dpsi, dtheta = x[PSI], x[DPSI], x[DTHETA]
    xdot = state_derivative(x, u)
    s_dd, psi_dd = xdot[DS], xdot[DPSI]
    c, sn, h = np.cos(psi), np.sin(psi), sensors.imu_height

    # Acceleration of the IMU point, in a level frame following the heading
    a_fwd = s_dd + h * (psi_dd * c - dpsi**2 * sn) - h * sn * dtheta**2
    a_up = -h * (psi_dd * sn + dpsi**2 * c)

    # Add +g upward (specific force), rotate into the tilted IMU axes
    a_x = c * a_fwd - sn * (a_up + G)
    a_z = sn * a_fwd + c * (a_up + G)

    gyro_pitch = dpsi
    gyro_yaw = dtheta * c  # yaw gyro is tilted with the beam
    return np.array([a_x, a_z, gyro_pitch, gyro_yaw])


def h_encoders(x, params=CONFIG.segway):
    """Encoder model -> [enc_l, enc_r], wheel angle RELATIVE to the beam.

    Rolling: left wheel travels s - L1*theta/2, right wheel s + L1*theta/2.
    The encoder housing turns with the beam, so the pitch is subtracted.
    """
    r, L1 = params.wheel_radius, params.axle_length
    enc_l = (x[S] - L1 * x[THETA] / 2) / r - x[PSI]
    enc_r = (x[S] + L1 * x[THETA] / 2) / r - x[PSI]
    return np.array([enc_l, enc_r])


def h(x, u):
    """Full measurement model, z = h(x, u)."""
    return np.concatenate([h_imu(x, u), h_encoders(x)])


def encoder_resolution(sensors=CONFIG.sensors):
    return 2 * np.pi / sensors.encoder_slots


def measure(x, u, rng, sensors=CONFIG.sensors):
    """Simulated sensor readings from the TRUE state: noise + quantisation."""
    imu = h_imu(x, u) + rng.normal(
        0, [sensors.accel_std, sensors.accel_std, sensors.gyro_std, sensors.gyro_std]
    )
    delta = encoder_resolution(sensors)
    enc = np.round(h_encoders(x) / delta) * delta  # encoders count whole slots
    return np.concatenate([imu, enc])
