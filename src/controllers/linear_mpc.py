"""
Linear MPC: predicts with the model linearized around upright.

The continuous linearization (MODEL.A, MODEL.B) is discretized with a
zero-order hold over one control sample, so the prediction is

    x[k+1] = Ad x[k] + Bd u[k]

With linear dynamics, a quadratic cost and box constraints the optimal
control problem is a single QP, so one SQP iteration (SQP_RTI) solves it
exactly. The terminal cost is the discrete LQR P of the same model.

Compare with nonlinear_mpc.py: same weights and constraints, but there the
prediction uses the full nonlinear model.

Positioning
-----------
The linear model has no x or y, so the goal is handed to the MPC as
references on distance s and heading theta, computed from the
dead-reckoned position by polar_reference() below. The MPC never sees x or y, so it cannot avoid
obstacles on its own; that has to happen in whatever chooses the goal.
"""

import casadi as ca
import numpy as np
import scipy as sci
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from pathlib import Path

from ..models.configuration import CONFIG
from ..models.dynamics import MODEL

BUILD_DIR = Path(__file__).parent / "acados_build"

# Indices in the state [s, ds, theta, dtheta, psi, dpsi]
S_IDX = 0
YAW_IDX = 2

MAX_STEP = 5  # [m] largest distance reference ahead of the robot, so a far goal is not one big step

# ============================== Polar guidance ==============================
GOAL_RADIUS = (
    0.03  # [m] inside this, stop steering towards the goal (atan2 is erratic at d -> 0)
)


def wrap_angle(angle: float) -> float:
    """Wrap to [-pi, pi), so the robot turns the short way round."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def polar_reference(
    target: np.ndarray,
    position: np.ndarray,
    s: float,
    theta: float,
    max_step: float,
) -> tuple[float, float]:
    """References (s_ref, theta_ref) that bring the robot from position to target."""
    diff = target - position
    heading = np.array([np.cos(theta), np.sin(theta)])

    # Heading: face the goal, or face away from it if it lies behind
    # (then the robot reverses instead of turning around).
    if np.linalg.norm(diff) < GOAL_RADIUS:
        heading_error = 0.0  # close enough: keep the current heading
    else:
        heading_error = wrap_angle(np.arctan2(diff[1], diff[0]) - theta)
        if abs(heading_error) > np.pi / 2:
            heading_error = wrap_angle(heading_error + np.pi)

    # Distance: only the part of the goal along the current heading,
    # d*cos(heading error). Goal to the side -> turn on the spot first;
    # goal behind -> negative, so the robot reverses.
    distance = np.clip(diff @ heading, -max_step, max_step)

    return s + distance, theta + heading_error


# ============================== MPC =========================================


def discretize(dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Ad, Bd of the linearized model, with the torque held constant over dt.

    Zero-order hold: treat u as an extra state that does not change, and
    integrate both exactly with one matrix exponential.
    """
    n_states, n_inputs = MODEL.B.shape
    augmented = np.zeros((n_states + n_inputs, n_states + n_inputs))
    augmented[:n_states, :n_states] = MODEL.A
    augmented[:n_states, n_states:] = MODEL.B

    transition = sci.linalg.expm(augmented * dt)
    return transition[:n_states, :n_states], transition[:n_states, n_states:]


def linear_segway_model(Ad: np.ndarray, Bd: np.ndarray) -> AcadosModel:
    model = AcadosModel()

    model.x = MODEL.x
    model.u = MODEL.u
    model.disc_dyn_expr = ca.mtimes(ca.DM(Ad), MODEL.x) + ca.mtimes(ca.DM(Bd), MODEL.u)
    model.name = "SegwayLinear"  # differs from the nonlinear model, so the generated code does not clash

    return model


def linear_segway_ocp(Q: np.ndarray, R: np.ndarray) -> AcadosOcp:
    Ad, Bd = discretize(CONFIG.mpc.T)

    ocp = AcadosOcp()
    ocp.model = linear_segway_model(Ad, Bd)
    ocp.solver_options.N_horizon = CONFIG.mpc.horizon
    ocp.solver_options.tf = CONFIG.mpc.horizon * CONFIG.mpc.T

    nx, nu = Bd.shape

    P = sci.linalg.solve_discrete_are(Ad, Bd, Q, R)

    # Stage cost
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.Vx = np.vstack([np.eye(nx), np.zeros((nu, nx))])
    ocp.cost.Vu = np.vstack([np.zeros((nx, nu)), np.eye(nu)])
    ocp.cost.W = sci.linalg.block_diag(Q, R)
    ocp.cost.yref = np.zeros(nx + nu)

    # Terminal cost
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.W_e = P
    ocp.cost.yref_e = np.zeros(nx)

    # State constraints
    ocp.constraints.lbx = np.array([-CONFIG.mpc.pitch_constraint])
    ocp.constraints.ubx = np.array([CONFIG.mpc.pitch_constraint])
    ocp.constraints.idxbx = np.array([4])  # [psi / pitch]
    ocp.constraints.x0 = np.zeros(nx)

    # Input constraints
    ocp.constraints.lbu = -CONFIG.mpc.torque_constraint * np.ones(nu)
    ocp.constraints.ubu = CONFIG.mpc.torque_constraint * np.ones(nu)
    ocp.constraints.idxbu = np.array([0, 1])  # [torque_left, torque_right]

    # Solver options
    ocp.solver_options.integrator_type = (
        "DISCRETE"  # the model is already x[k+1] = f(x[k], u[k])
    )
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # one QP is the exact solution here

    ocp.code_export_directory = str(BUILD_DIR / "c_generated_code")

    return ocp


class LinearMPC:
    def __init__(self, Q: np.ndarray, R: np.ndarray, target=(0.0, 0.0)) -> None:
        self.solver = AcadosOcpSolver(
            linear_segway_ocp(Q, R),
            json_file=str(BUILD_DIR / "acados_ocp_linear.json"),
        )
        self.N = CONFIG.mpc.horizon
        self.dt = CONFIG.mpc.T  # same dt as the UKF / control loop
        self.nx, self.nu = MODEL.B.shape
        self.position = np.zeros(2)  # dead-reckoned [x, y]
        self.set_target(*target)

    def set_target(self, x_goal: float, y_goal: float) -> None:
        self.target = np.array([x_goal, y_goal], dtype=float)

    def get_position(self) -> np.ndarray:
        return self.position.copy()

    def compute(self, state_estimate: np.ndarray) -> np.ndarray:
        s, s_dot, theta = (
            state_estimate[S_IDX],
            state_estimate[1],
            state_estimate[YAW_IDX],
        )
        self.position += self.dt * s_dot * np.array([np.cos(theta), np.sin(theta)])

        s_ref, theta_ref = polar_reference(
            self.target, self.position, s, theta, MAX_STEP
        )
        yref = np.zeros(self.nx + self.nu)
        yref[S_IDX] = s_ref
        yref[YAW_IDX] = theta_ref
        for k in range(self.N):
            self.solver.set(k, "yref", yref)
        self.solver.set(self.N, "yref", yref[: self.nx])

        self.solver.set(0, "lbx", state_estimate)
        self.solver.set(0, "ubx", state_estimate)
        self.solver.solve()
        return self.solver.get(0, "u")

    def reset(self) -> None:
        self.solver.reset()
        self.position[:] = 0.0
