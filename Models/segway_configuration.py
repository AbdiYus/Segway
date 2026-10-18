from dataclasses import dataclass, field

# frozen ensures that the class cannot be changed during runtime


@dataclass(frozen=True)
class SegwayParams:
    axle_mass: float = 1.0  # [kg]
    beam_mass: float = 1.0  # [kg]
    wheel_mass: float = 1.0  # [kg]
    axle_length: float = 0.1  # [Meters]
    beam_lenght: float = 0.1  # [Meters]
    wheel_radius: float = 0.1  # [Meters]


@dataclass(frozen=True)
class MPCSettings:
    T: float = 0.02  # [seconds]
    horizon: int = 20  # [MPC window]
    torque_max: float = 3.0  # [Nm]


@dataclass(frozen=True)
class Config:
    segway: SegwayParams = field(default_factory=SegwayParams)
    mpc: MPCSettings = field(default_factory=MPCSettings)


CONFIG = Config()
