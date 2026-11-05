from dataclasses import dataclass, field
import numpy as np


@dataclass
class History:
    """Everything that happened at each controller update, for plotting later."""

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
