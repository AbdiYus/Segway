from dataclasses import dataclass, field
import numpy as np

# frozen ensures that the class cannot be changed during runtime


@dataclass(frozen=True)
class SegwayParams:
    axle_mass: float = 0.3  # [kg]
    beam_mass: float = 0.5  # [kg]
    wheel_mass: float = 0.2  # [kg]
    axle_length: float = 0.1  # [Meters]
    beam_lenght: float = 0.25  # [Meters]
    wheel_radius: float = 0.05  # [Meters]


@dataclass(frozen=True)
class MPCSettings:
    T: float = 0.1  # [seconds]
    horizon: int = 10  # [MPC window]
    torque_constraint: float = 0.5  # [Nm]
    pitch_constraint: float = np.radians(5)

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
class Config:
    segway: SegwayParams = field(default_factory=SegwayParams)
    mpc: MPCSettings = field(default_factory=MPCSettings)
    sensors: SensorSettings = field(default_factory=SensorSettings)
    ukf: UKFSettings = field(default_factory=UKFSettings)


CONFIG = Config()
