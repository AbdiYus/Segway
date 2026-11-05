from ..controllers.base import Controller, ControllerType
from ..observers.base import Observer, ObserverType
from .utility import History, Pose, StateDefitions
from ..models.sensors import measure
import numpy as np


class ControlLoop:
    """The segway's brain: an observer and a controller, each swappable at runtime."""

    def __init__(
        self,
        controllers: dict[ControllerType, Controller],
        observers: dict[ObserverType, Observer],
        active_controller: ControllerType,
        active_observer: ObserverType,
    ) -> None:
        self.controllers = controllers
        self.observers = observers
        self.active_controller = active_controller
        self.active_observer = active_observer
        self.first_update = True

    @property
    def controller(self) -> Controller:
        return self.controllers[self.active_controller]

    @property
    def observer(self) -> Observer:
        return self.observers[self.active_observer]

    def switch_controller_to(self, chosen_controller: ControllerType) -> None:
        self.controllers[chosen_controller].reset()
        self.active_controller = chosen_controller

    # NOTE: This switch could cause issues with the covariance matrix P...
    def switch_observer_to(self, chosen_observer: ObserverType) -> None:
        self.observers[chosen_observer].reset_to(self.observer.states.copy())
        self.active_observer = chosen_observer

    def update(self, measurement: np.ndarray, applied_torque: np.ndarray) -> np.ndarray:
        """Estimate the state from the latest measurement and return the commanded torque."""
        if not self.first_update:  # nothing to predict at t = 0
            self.observer.predict(applied_torque)
        self.first_update = False
        self.observer.update(measurement, applied_torque)
        return self.controller.compute(self.observer.states)


class Segway:
    """One simulated segway: true plant, sensors and motors, steered by a ControlLoop."""

    def __init__(
        self,
        control_loop: ControlLoop,
        plant,  # AcadosSimSolver, already set to physics_dt
        initial_state: np.ndarray,
        physics_dt: float,
        control_dt: float,
        position: tuple[float, float] = (0.0, 0.0),
        seed: int = 0,
    ) -> None:
        self.control_loop = control_loop
        self.plant = plant
        self.physics_dt = physics_dt
        self.states_defitions = StateDefitions

        self.steps_per_control = round(control_dt / physics_dt)
        if not np.isclose(self.steps_per_control * physics_dt, control_dt):
            raise ValueError("control_dt must be a whole multiple of physics_dt")

        self.rng = np.random.default_rng(seed)
        self.state: np.ndarray = np.asarray(initial_state, dtype=float).copy()
        self.position = np.array(position, dtype=float)  # [x, y] on the floor
        self.torque = np.zeros(2)  # applied torque, held between controller updates
        self.step_count = 0
        self.history = History()

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
            self.state[StateDefitions.YAW],
            self.state[StateDefitions.PITCH],
            self.state[StateDefitions.S],
        )

    # ============================ Disturbances ===========================
    def push(self, pitch_rate_change: float) -> None:
        self.state[StateDefitions.PITCH_RATE] += pitch_rate_change

    def tilt_to(self, pitch: float) -> None:
        self.state[StateDefitions.PITCH] = pitch

    # ============================== Internals ============================
    def update_controller(self, time: float) -> None:
        measurement = measure(self.state, self.torque, self.rng)
        self.torque = self.control_loop.update(measurement, applied_torque=self.torque)
        self.history.record(
            time, self.state, self.control_loop.observer.states, self.torque
        )

    def integrate_plant(self) -> None:
        self.state = np.asarray(self.plant.simulate(x=self.state, u=self.torque))
        speed, heading = (
            self.state[StateDefitions.S_DOT],
            self.state[StateDefitions.YAW],
        )
        self.position += (
            speed * np.array([np.cos(heading), np.sin(heading)]) * self.physics_dt
        )
