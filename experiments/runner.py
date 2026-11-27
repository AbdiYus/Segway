"""Headless runner for the robustness experiments.

Model error
-----------
The UKF and the controllers are built once, on the nominal parameters in
configuration.py. The simulated robot ("true plant") gets its own parameters
p_i = p_nominal_i * (1 + delta_i), used by the plant integrator AND by the
sensor model. The true plant takes the parameters as acados runtime
parameters, so its C code is generated once and every configuration reuses it.

Test scenario (same for every run)
----------------------------------
Start at rest with a pitch of 3 deg, push the pitch rate by 0.5 rad/s at
t = 2 s, simulate 20 s. Sensor noise comes from a fixed seed per run, so every
configuration sees the same noise for the same seed.
"""

from dataclasses import dataclass, fields, replace
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosSim, AcadosSimSolver

from src.controllers import ControllerType
from src.models import CONFIG, SegwayParameters, dynamics as dyn
from src.models.sensors import measure
from src.observers import ObserverType
from src.simulator import Segway, StateIndex

BUILD_DIR = Path(__file__).parent / "build"

PHYSICS_DT = 0.01  # [s]
INITIAL_PITCH = np.radians(3)  # [rad]
PUSH_TIME = 2.0  # [s]
PUSH = 0.5  # [rad/s] sudden change in pitch rate
T_END = 20.0  # [s]

# Physical parameters, in the order they are passed to the true plant
PARAMETERS = [f.name for f in fields(SegwayParameters)]
_SYMBOLS = {
    "axle_mass": dyn.m_axle,
    "beam_mass": dyn.m_beam,
    "wheel_mass": dyn.m_wheel,
    "axle_length": dyn.AXLE_LENGTH,
    "beam_length": dyn.BEAM_LENGTH,
    "wheel_radius": dyn.WHEEL_RADIUS,
}
LABELS = {  # for plots, the report's naming
    "axle_mass": r"$m_{\mathrm{axle}}$",
    "beam_mass": r"$m_{\mathrm{beam}}$",
    "wheel_mass": r"$m_{\mathrm{wheel}}$",
    "axle_length": r"$L_1$",
    "beam_length": r"$L_2$",
    "wheel_radius": r"$h$",
}


def perturbed(errors: dict[str, float]) -> SegwayParameters:
    """True parameters for relative errors, e.g. {"beam_mass": 0.1} = +10 %."""
    nominal = CONFIG.segway
    return replace(
        nominal, **{k: getattr(nominal, k) * (1 + d) for k, d in errors.items()}
    )


def as_vector(params: SegwayParameters) -> np.ndarray:
    return np.array([getattr(params, k) for k in PARAMETERS])


# ============================ True plant ======================================
def _parametric_dynamics():
    """x_dot = f(x, u, p) from the same symbolic M, b as the nominal model."""
    p = ca.SX.sym("p", len(PARAMETERS))
    syms = dyn.syms + [_SYMBOLS[k] for k in PARAMETERS]
    xup = ca.vertcat(dyn.x, dyn.u, p)
    M = dyn.sympy2casadi(dyn.M.subs(dyn.state_subs), syms, xup)
    b = dyn.sympy2casadi(dyn.b.subs(dyn.state_subs), syms, xup)
    q_ddot = ca.solve(M, b)
    x = dyn.x
    f = ca.vertcat(x[1], q_ddot[0], x[3], q_ddot[1], x[5], q_ddot[2])
    return p, f


P_SYM, F_EXPL = _parametric_dynamics()
F_TRUE = ca.Function("f_true", [dyn.x, dyn.u, P_SYM], [F_EXPL])


def true_plant_integrator() -> AcadosSimSolver:
    model = AcadosModel()
    model.x, model.u, model.p, model.xdot = dyn.x, dyn.u, P_SYM, dyn.xdot
    model.f_expl_expr = F_EXPL
    model.f_impl_expr = dyn.xdot - F_EXPL
    model.name = "SegwayTruePlant"

    sim = AcadosSim()
    sim.model = model
    sim.parameter_values = as_vector(CONFIG.segway)
    sim.solver_options.T = PHYSICS_DT
    sim.solver_options.integrator_type = "IRK"  # same as the app's plant
    sim.solver_options.num_steps = 2
    sim.code_export_directory = str(BUILD_DIR / "c_generated_code_true_plant")
    return AcadosSimSolver(sim, json_file=str(BUILD_DIR / "acados_sim_true_plant.json"))


# ============================ Segway with model error =========================
class ExperimentSegway(Segway):
    """A Segway whose plant and sensors use true parameters, not the nominal ones.

    true_state_feedback=True bypasses the UKF: the controller gets the true
    state, which isolates the controller from the estimator.
    """

    def __init__(self, controller_type: ControllerType) -> None:
        initial_state = np.zeros(6)
        initial_state[StateIndex.PITCH] = INITIAL_PITCH
        super().__init__(
            controller_type,
            ObserverType.UNSCENTED_KALMAN_FILTER,
            initial_state,
            PHYSICS_DT,
        )
        self.plant = true_plant_integrator()
        self.true_state_feedback = False
        self.set_true_parameters(CONFIG.segway)
        self.fallbacks = 0  # NMPC solves that failed and fell back to LQR

    def set_true_parameters(self, params: SegwayParameters) -> None:
        self.true_params = params
        p = as_vector(params)
        self.plant.set("p", p)
        self._f_true = lambda x, u: F_TRUE(x, u, p)

    def update_controller(self, time: float) -> None:
        controller = self.control_loop.controller
        if self.true_state_feedback:
            self.torque = controller.compute(self.state.copy())
        else:
            z = measure(
                self.state, self.torque, self.rng, f=self._f_true, params=self.true_params
            )
            self.torque = self.control_loop.update(z, applied_torque=self.torque)
        if getattr(controller, "cold", False):  # NonlinearMPC: last solve failed
            self.fallbacks += 1


@dataclass
class Run:
    errors: dict
    seed: int
    true_state_feedback: bool
    time: np.ndarray  # [s]
    pitch: np.ndarray  # [rad], true pitch after each physics step
    fallen: bool
    fallbacks: int


def run_scenario(
    segway: ExperimentSegway,
    errors: dict[str, float],
    seed: int,
    true_state_feedback: bool = False,
) -> Run:
    segway.set_true_parameters(perturbed(errors))
    segway.true_state_feedback = true_state_feedback
    segway.rng = np.random.default_rng(seed)
    segway.reset()
    segway.fallbacks = 0

    n = round(T_END / PHYSICS_DT)
    k_push = round(PUSH_TIME / PHYSICS_DT)
    pitch = np.empty(n)
    for k in range(n):
        if k == k_push:
            segway.push(PUSH)
        segway.step(k * PHYSICS_DT)
        pitch[k] = segway.state[StateIndex.PITCH]
    time = (np.arange(n) + 1) * PHYSICS_DT
    return Run(errors, seed, true_state_feedback, time, pitch, segway.fallen, segway.fallbacks)


# ============================ Scoring =========================================
def convergence_time(run: Run, band: float) -> float:
    """Time from the push until |pitch| enters the band and stays in it.

    NaN if the robot fell or has not settled inside the band by the end.
    """
    if run.fallen:
        return np.nan
    after = run.time > PUSH_TIME
    outside = np.flatnonzero(after & (np.abs(run.pitch) >= band))
    if outside.size == 0:
        return 0.0
    last = outside[-1]
    if last == run.time.size - 1:
        return np.nan
    return run.time[last + 1] - PUSH_TIME
