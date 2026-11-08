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
"""

import casadi as ca
import numpy as np
import scipy as sci
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from pathlib import Path

from ..models.configuration import CONFIG
from ..models.dynamics import MODEL

BUILD_DIR = Path(__file__).parent / "acados_build"


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
    def __init__(self, Q: np.ndarray, R: np.ndarray) -> None:
        self.solver = AcadosOcpSolver(
            linear_segway_ocp(Q, R),
            json_file=str(BUILD_DIR / "acados_ocp_linear.json"),
        )

    def compute(self, state_estimate: np.ndarray) -> np.ndarray:
        self.solver.set(0, "lbx", state_estimate)
        self.solver.set(0, "ubx", state_estimate)
        self.solver.solve()
        return self.solver.get(0, "u")

    def reset(self) -> None:
        self.solver.reset()
