"""
The segway's brain: an observer and a controller, each swappable at runtime.

Role in the system
------------------
Owned by a Segway, which calls update() once per control sample with the
latest measurement. Holds every controller and observer the segway can use,
but only one of each is active. Knows nothing about the true state, the
plant or rendering.
"""

import numpy as np

from ..controllers import Controller, ControllerType
from ..observers import Observer, ObserverType


class ControlLoop:
    """Observer + controller, run once per control sample."""

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

    # ============================== Switching ============================
    def switch_controller_to(self, chosen_controller: ControllerType) -> None:
        self.controllers[chosen_controller].reset()
        self.active_controller = chosen_controller

    def switch_observer_to(self, chosen_observer: ObserverType) -> None:
        """The new observer starts from the old estimate, with its initial uncertainty."""
        self.observers[chosen_observer].reset_to(self.observer.states.copy())
        self.active_observer = chosen_observer

    # ============================== Running ==============================
    def update(self, measurement: np.ndarray, applied_torque: np.ndarray) -> np.ndarray:
        """Estimate the state from the latest measurement and return the commanded torque."""
        if not self.first_update:  # nothing to predict at t = 0
            self.observer.predict(applied_torque)
        self.first_update = False
        self.observer.update(measurement, applied_torque)
        return self.controller.compute(self.observer.states)

    # ============================== Utiltiy ==============================
    def get_names(self) -> list[str]:
        names: list[str] = []
        for controller in self.controllers:
            names.append(controller.name)
        return names
