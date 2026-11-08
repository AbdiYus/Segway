from time import perf_counter
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


@dataclass
class History:
    """Everything that happened at each controller update, for plotting later."""

    time: list = field(default_factory=list)
    true_state: list = field(default_factory=list)
    estimated_state: list = field(default_factory=list)
    torque: list = field(default_factory=list)
    true_pitch: list = field(default_factory=list)
    estimated_pitch: list = field(default_factory=list)

    def record(self, time, true_state, estimated_state, torque) -> None:
        self.time.append(time)
        self.true_state.append(true_state.copy())
        self.estimated_state.append(estimated_state.copy())
        self.torque.append(torque.copy())
        self.true_pitch.append(np.degrees(true_state[StateIndex.PITCH]))
        self.estimated_pitch.append(np.degrees(estimated_state[StateIndex.PITCH]))

    def as_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array(self.time),
            np.array(self.true_state),
            np.array(self.estimated_state),
            np.array(self.torque),
            np.array(self.true_pitch),
            np.array(self.estimated_pitch),
        )


@dataclass(frozen=True)
class Pose:
    """Everything the renderer needs to draw one segway."""

    x: float  # [m]
    y: float  # [m]
    yaw: float  # [rad]
    pitch: float  # [rad]
    distance: float  # [m] distance rolled by the axle centre, spins the wheels


class StateIndex(IntEnum):
    """Position of each quantity in the state vector x."""

    S = 0  # distance driven along the heading   [m]
    S_DOT = 1  # forward speed                        [m/s]
    YAW = 2  # heading, positive = turning left     [rad]
    YAW_RATE = 3  #                                      [rad/s]
    PITCH = 4  # body tilt, 0 = upright, + = forward  [rad]
    PITCH_RATE = 5  #                                      [rad/s]


class HistoryIndex(IntEnum):
    """Position of each quantity in the history"""

    TIME = 0
    TRUE_STATE = 1
    ESTIMATED_STATE = 2
    TORQUE = 3
    TRUE_PITCH = 4
    ESTIMATED_PITCH = 5


class RealTimePacer:
    """
    Turns wall-clock time into a whole number of fixed physics steps.

    The physics always uses the same dt (stable, repeatable), while the window
    redraws whenever it can. If a frame is late, the next one simply takes
    more physics steps to catch up.
    """

    MAX_CATCH_UP = (
        0.1  # [s] never catch up more than this (e.g. after dragging the window)
    )

    def __init__(self, physics_dt: float, time_scale: float = 1.0) -> None:
        self.physics_dt = physics_dt
        self.time_scale = time_scale  # 0 = paused, 0.5 = slow motion, 1 = real time
        self.last_wall_time = perf_counter()
        self.unsimulated_time = 0.0

    def steps_due(self) -> int:
        now = perf_counter()
        elapsed = min(now - self.last_wall_time, self.MAX_CATCH_UP)
        self.last_wall_time = now

        self.unsimulated_time += elapsed * self.time_scale
        steps = int(self.unsimulated_time / self.physics_dt)
        self.unsimulated_time -= steps * self.physics_dt
        return steps
