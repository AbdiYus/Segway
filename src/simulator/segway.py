"""
One simulated segway: the true plant, its sensors and motors.

Role in the system
------------------
Replaces the hardware. The true state is integrated forward, the sensors
read it with noise, and the ControlLoop (the brain) turns the measurement
into a torque. Between control samples the torque is held constant
(zero-order hold), like on real hardware.

Two clocks run side by side:
    physics_dt : how often the true plant is integrated. Small, so the motion
                 looks smooth and disturbances can land between controller
                 updates.
    control_dt : how often the sensors are read, the observer updates and the
                 controller runs. Must be a whole multiple of physics_dt.

Owned by a World, which owns the time. Knows nothing about other segways
or rendering.
"""

import numpy as np

from ..controllers.nonlinear_mpc import segway_integrator, segway_ocp


from ..models import measure, CONFIG
from ..controllers import (
    Controller,
    ControllerType,
    LQR,
    LinearMPC,
    NonlinearMPC,
)
from ..observers import ObserverType, Observer, UnscentedKalmanFilter
from .utility import History, Pose, StateIndex
from .control_loop import ControlLoop

# ================================ Tuning =========================================

# Weights on [s, ds, theta, dtheta, psi, dpsi] and [tau_l, tau_r]
Q_LQR = np.diag([1.0, 0.1, 1.0, 0.1, 100.0, 1.0])
R_LQR = np.diag([0.01, 0.01])

Q_LINEAR_MPC = np.diag([0.1, 0.1, 1.0, 0.1, 100.0, 1.0])
R_LINEAR_MPC = np.diag([0.01, 0.01])

Q_NONLINEAR_MPC = np.diag([0.1, 0.1, 1.0, 0.1, 100.0, 1.0, 1.0, 1.0])
R_NONLINEAR_MPC = np.diag([0.01, 0.01])


# ================================ Segway =========================================
class Segway:
    """True plant, sensors and motors, steered by a ControlLoop."""

    def __init__(
        self,
        controller_type: ControllerType,
        observer_type: ObserverType,
        initial_state: np.ndarray,
        physics_dt: float,
        position: tuple[float, float] = (0.0, 0.0),
        seed: int = 0,
    ) -> None:
        self.control_loop = self.create_controller_and_observer(
            controller_type, observer_type
        )
        self.plant = segway_integrator(segway_ocp(Q_NONLINEAR_MPC, R_NONLINEAR_MPC))
        self.plant.set("T", physics_dt)
        self.physics_dt = physics_dt

        control_dt = CONFIG.mpc.T
        self.steps_per_control = round(control_dt / physics_dt)
        if not np.isclose(self.steps_per_control * physics_dt, control_dt):
            raise ValueError("control_dt must be a whole multiple of physics_dt")

        self.rng = np.random.default_rng(seed)
        self.state: np.ndarray = np.asarray(
            initial_state, dtype=float
        ).copy()  # TRUE state
        self.position = np.array(position, dtype=float)  # [x, y] on the floor
        self.torque = np.zeros(2)  # applied torque, held between controller updates
        self.step_count = 0
        self.history = History()
        self.position = np.zeros(2)

    # ============================== Running ==============================
    def step(self, time: float) -> None:
        """Advance by one physics step. The world owns the clock."""
        if self.step_count % self.steps_per_control == 0:
            self.update_controller(time)
        self.integrate_plant()
        self.step_count += 1

    @property
    def pose(self) -> Pose:
        x, y = self.position
        return Pose(
            x,
            y,
            self.state[StateIndex.YAW],
            self.state[StateIndex.PITCH],
            self.state[StateIndex.S],
        )

    def set_goal(self, x: float, y: float) -> None:
        self.control_loop.set_goal(x, y)

    @property
    def goal(self) -> np.ndarray:
        return self.control_loop.goal

    def integrate_plant(self) -> None:
        self.state = np.asarray(self.plant.simulate(x=self.state, u=self.torque))
        s_dot, theta = self.state[StateIndex.S_DOT], self.state[StateIndex.YAW]
        self.position += (
            self.physics_dt * s_dot * np.array([np.cos(theta), np.sin(theta)])
        )

    # ============================ Disturbances ===========================
    def push(self, pitch_rate_change: float) -> None:
        """A sudden hit on the body: instantly changes how fast it is tipping."""
        self.state[StateIndex.PITCH_RATE] += pitch_rate_change

    def tilt_to(self, pitch: float) -> None:
        """Teleport the body to a given pitch angle."""
        self.state[StateIndex.PITCH] = pitch

    # ============================== Internals ============================
    def create_controller_and_observer(
        self,
        controller_type: ControllerType,
        observer_type: ObserverType,
    ) -> ControlLoop:
        controllers: dict[ControllerType, Controller] = {
            ControllerType.LQR: LQR(Q=Q_LQR, R=R_LQR),
            ControllerType.LINEAR_MPC: LinearMPC(Q=Q_LINEAR_MPC, R=R_LINEAR_MPC),
            ControllerType.NONLINEAR_MPC: NonlinearMPC(
                Q=Q_NONLINEAR_MPC, R=R_NONLINEAR_MPC
            ),
        }

        observers: dict[ObserverType, Observer] = {
            ObserverType.UNSCENTED_KALMAN_FILTER: UnscentedKalmanFilter(
                dt=CONFIG.mpc.T
            ),
        }
        return ControlLoop(
            controllers,
            observers,
            active_controller=controller_type,
            active_observer=observer_type,
        )

    def update_controller(self, time: float) -> None:
        measurement = measure(self.state, self.torque, self.rng)
        self.torque = self.control_loop.update(measurement, applied_torque=self.torque)
        self.history.record(
            time, self.state, self.control_loop.observer.states, self.torque
        )

    # def integrate_plant(self) -> None:
    #     self.state = np.asarray(self.plant.simulate(x=self.state, u=self.torque))
    #     speed, heading = self.state[StateIndex.S_DOT], self.state[StateIndex.YAW]
    #     self.position += (
    #         speed * np.array([np.cos(heading), np.sin(heading)]) * self.physics_dt
    #     )
