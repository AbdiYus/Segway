"""Configuration: every physical parameter and tuning value in one place.

Holds the segway's physical dimensions, sensor characteristics, and the
tuning of the MPC and UKF. All classes are frozen, so a config cannot be
changed mid-simulation. For tuning you simply build a new one instead
(e.g dataclasses.replace).

Role in the system
------------------
Built in main.py and passed down to the parts that need it:
    - models/dynamics.py           SegwayParameters -> equations of motion
    - models/sensors.py            SensorSettings -> measurement noise, encoder quantization
    - observers/ukf.py             UKFSettings (+ SensorSettings for its model)
    - controllers/linear_mpc.py    MPCSettings
    - renderer/                    RendererSettings (+ SegwayParameters for the 3D model)

Contains no logic, nothing here imports from the rest of the project.

Conventions
-----------
SI units throughout ([m], [kg], [s], [rad], [Nm]); angles in radians.
State order (used by UKFSettings.initial_std):
    [s, ds, theta, dtheta, psi, dpsi]
    s        position/displacement along the ground   [m]
    psi      body pitch                              [rad]
    theta    heading (yaw)                           [rad]
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class SegwayParameters:
    axle_mass: float = 0.3  # [kg]
    beam_mass: float = 0.5  # [kg]
    wheel_mass: float = 0.2  # [kg]
    axle_length: float = 0.1  # [Meters]
    beam_length: float = 0.25  # [Meters]
    wheel_radius: float = 0.05  # [Meters]


@dataclass(frozen=True)
class MPCSettings:
    T: float = 0.1  # [seconds]
    horizon: int = 10  # [MPC window]
    torque_constraint: float = 0.08  # [Nm]
    pitch_constraint: float = np.radians(5)  # [rad]
    pitch_rate_constraint: float = np.radians(60)  # [rad/s]
    displacement_rate_constraint: float = 1.0  # [m/s]
    slack_weight: float = 1e4


@dataclass(frozen=True)
class SensorSettings:
    imu_height: float = 0.10  # [Meters] axle -> IMU, measured along the beam
    encoder_slots: int = 20  # slots on the optical encoder disc
    accel_std: float = 0.3  # [m/s^2] accelerometer noise (measure at rest)
    gyro_std: float = 0.01  # [rad/s] gyro noise (measure at rest)


@dataclass(frozen=True)
class UKFSettings:
    # Van der Merwe scaled sigma points, recommended values: kappa = 3 - n, n = 6
    alpha: float = 0.1
    beta: float = 2.0
    kappa: float = -3.0
    rk4_substeps: int = 4  # integration steps per sample inside the filter
    # Process noise Q: unknown acceleration on each (position, velocity) pair
    accel_std_s: float = 1.5  # [m/s^2]
    accel_std_theta: float = 2.0  # [rad/s^2]
    accel_std_psi: float = 10.0  # [rad/s^2]
    # Initial uncertainty (std) of [s, ds, theta, dtheta, psi, dpsi]
    initial_std: tuple = (0.01, 0.1, 0.01, 0.1, 0.4, 1.0)


@dataclass(frozen=True)
class RendererSettings:
    window_size: tuple[int, int] = (900, 900)  # [pixels]
    floor_grid_size: float = 0.1  # [m]
    floor_size: int = 2  # [m]


@dataclass(frozen=True)
class Config:
    segway: SegwayParameters = field(default_factory=SegwayParameters)
    mpc: MPCSettings = field(default_factory=MPCSettings)
    sensors: SensorSettings = field(default_factory=SensorSettings)
    ukf: UKFSettings = field(default_factory=UKFSettings)
    renderer: RendererSettings = field(default_factory=RendererSettings)


CONFIG = Config()
