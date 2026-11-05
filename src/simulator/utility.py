from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np


@dataclass
class History:
    """Recording of every applied actuation and states"""

    time: list = field(default_factory=list)
    true_state: list = field(default_factory=list)
    estimated_state: list = field(default_factory=list)
    torque: list = field(default_factory=list)

    def record(self, time, true_state, estimated_state, torque) -> None:
        self.time.append(time)
        self.true_state.append(true_state.copy())
        self.estimated_state.append(estimated_state.copy())
        self.torque.append(torque.copy())

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array(self.time),
            np.array(self.true_state),
            np.array(self.estimated_state),
            np.array(self.torque),
        )


@dataclass(frozen=True)
class Pose:
    x: float  # [m]
    y: float  # [m]
    yaw: float  # [rad]
    pitch: float  # [rad]
    displacement: float  # [m]


class StateDefitions(IntEnum):
    S = 0  # distance driven along the heading   [m]
    S_DOT = 1  # forward speed                        [m/s]
    YAW = 2  # heading, positive = turning left     [rad]
    YAW_RATE = 3  #                                      [rad/s]
    PITCH = 4  # body tilt, 0 = upright, + = forward  [rad]
    PITCH_RATE = 5  #                                      [rad/s]
