"""
The simulated segway with its control system in the loop.

This replaces the hardware: a "true" plant is integrated forward, the
sensors read it, the UKF estimates the state and the MPC chooses the torque.
Nothing in here knows about rendering, so the same loop can be driven by the
3D viewer, a GUI or a plain for-loop.

Two clocks run side by side:
    physics_dt : how often the true plant is integrated. Small, so the motion
                 looks smooth and disturbances can land between controller
                 updates.
    control_dt : how often the sensors are read, the UKF updates and the MPC
                 solves (CONFIG.mpc.T). Between updates the torque is held
                 constant (zero-order hold), exactly like on real hardware.
"""

from dataclasses import dataclass, field

import numpy as np

from Control_System.MPC import segway_integrator, segway_ocp, segway_solver
from Control_System.Models.segway_configuration import CONFIG
from Control_System.Models.segway_sensors import measure
from Simulator.UKF import SegwayUKF

# Index of each entry in the state vector [s, s_dot, yaw, yaw_rate, pitch, pitch_rate]
S, S_DOT, YAW, YAW_RATE, PITCH, PITCH_RATE = range(6)


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


class ClosedLoop:
    """True plant + sensors + UKF + MPC, advanced one physics step at a time."""

    def __init__(
        self,
        initial_state: np.ndarray,
        physics_dt: float = 0.01,
        use_ukf: bool = True,
        seed: int = 0,
    ) -> None:
        self.control_dt = CONFIG.mpc.T
        self.physics_dt = physics_dt
        self.physics_steps_per_control = round(self.control_dt / physics_dt)
        if not np.isclose(self.physics_steps_per_control * physics_dt, self.control_dt):
            raise ValueError("control_dt must be a whole multiple of physics_dt")

        ocp = segway_ocp()
        self.mpc = segway_solver(ocp)
        self.plant = segway_integrator(ocp)
        self.plant.set(
            "T", physics_dt
        )  # the plant takes small steps, not a full MPC step

        self.ukf = SegwayUKF(
            dt=self.control_dt
        )  # starts at x_hat = 0, it must find the true state
        self.use_ukf = use_ukf
        self.rng = np.random.default_rng(seed)

        self.state = np.asarray(initial_state, dtype=float).copy()  # the TRUE state
        self.position = np.zeros(
            2
        )  # [x, y] on the floor, integrated from s_dot and yaw
        self.torque = np.zeros(2)  # [tau_l, tau_r], held between controller updates
        self.time = 0.0
        self.physics_step_count = 0

        self.motors_enabled = True
        self.solver_status = 0
        self.history = History()

    # ============================== Running ==============================
    def step(self) -> None:
        """Advance the world by one physics step."""
        if self.physics_step_count % self.physics_steps_per_control == 0:
            self.update_controller()

        self.integrate_plant()
        self.physics_step_count += 1
        self.time += self.physics_dt

    # ============================ Disturbances ===========================
    # Called from the keyboard now, from the GUI later.
    def push(self, pitch_rate_change: float) -> None:
        """A sudden hit on the body: instantly changes how fast it is tipping."""
        self.state[PITCH_RATE] += pitch_rate_change

    def tilt_to(self, pitch: float) -> None:
        """Teleport the body to a given pitch angle."""
        self.state[PITCH] = pitch

    def set_motors_enabled(self, enabled: bool) -> None:
        """With the motors off the controller still estimates, but applies no torque."""
        self.motors_enabled = enabled

    # ============================== Internals ============================
    def update_controller(self) -> None:
        """One control cycle: read sensors -> UKF correct -> MPC solve -> UKF predict."""
        measurement = measure(self.state, self.torque, self.rng)
        self.ukf.update(measurement, self.torque)

        state_for_mpc = self.ukf.x if self.use_ukf else self.state

        if self.motors_enabled:
            self.mpc.set(0, "lbx", state_for_mpc)
            self.mpc.set(0, "ubx", state_for_mpc)
            self.solver_status = self.mpc.solve()
            new_torque = self.mpc.get(0, "u")
        else:
            new_torque = np.zeros(2)

        self.history.record(self.time, self.state, self.ukf.x, new_torque)

        self.ukf.predict(new_torque)
        self.torque = new_torque

    def integrate_plant(self) -> None:
        """Move the true segway forward by physics_dt with the current torque."""
        self.state = self.plant.simulate(x=self.state, u=self.torque)

        speed, heading = self.state[S_DOT], self.state[YAW]
        self.position += (
            speed * np.array([np.cos(heading), np.sin(heading)]) * self.physics_dt
        )
