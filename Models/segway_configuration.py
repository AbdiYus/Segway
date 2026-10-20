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
class Config:
    segway: SegwayParams = field(default_factory=SegwayParams)
    mpc: MPCSettings = field(default_factory=MPCSettings)


CONFIG = Config()
