"""Entry point: build one segway with its controllers and observers, simulate it, plot it.

Run from the project root with:  python -m src.main
"""

import numpy as np

from .simulator import Segway, StateIndex, Simulator
from .controllers import ControllerType
from .observers import ObserverType

# ================================ Simulation =====================================

PHYSICS_DT = 0.01  # [s] plant integration step, CONFIG.mpc.T must be a multiple of it
# SIMULATION_TIME = 5.0  # [s]
INITIAL_PITCH = np.radians(3)  # must lie inside CONFIG.mpc.pitch_constraint (5 deg)


def main() -> None:
    initial_state = np.zeros(6)
    initial_state[StateIndex.PITCH] = INITIAL_PITCH

    segway = Segway(
        ControllerType.NONLINEAR_MPC,
        ObserverType.UNSCENTED_KALMAN_FILTER,
        initial_state,
        PHYSICS_DT,
        position=(0, 0),
    )

    # controller = segway.control_loop.controllers[ControllerType.NONLINEAR_MPC]
    # controller.set_target(1, 1)

    # FIXME: When adding other segwways, it generates the controllers code again.
    # segway_2 = Segway(
    #     ControllerType.LINEAR_MPC,
    #     ObserverType.UNSCENTED_KALMAN_FILTER,
    #     initial_state,
    #     PHYSICS_DT,
    #     position=(1, 1),
    # )

    simulator = Simulator(PHYSICS_DT, segway.history)
    simulator.add(segway)
    # simulator.add(segway_2)
    simulator.run()


if __name__ == "__main__":
    main()
