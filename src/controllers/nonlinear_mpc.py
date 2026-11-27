"""
Nonlinear MPC: predicts with the full nonlinear model, extended with the
planar position (x, y).

Steering towards the goal uses polar guidance (polar_reference() below, the
same as in linear_mpc.py): the goal becomes references on distance s and
heading theta.
x and y are kept in the prediction model but have no weight, so they are
available for constraints later (obstacle avoidance) without driving the
solution now.

Speed choices
-------------
- SQP_RTI: one QP per sample, warm-started from the shifted previous
  solution. The solution converges over consecutive samples instead of
  within one, which keeps the solve time short and predictable.
- ERK: the dynamics are not stiff, so an explicit Runge-Kutta integrator is
  accurate enough and much cheaper than an implicit one.
- cost_scaling = 1: the stage cost is a plain sum over the horizon, which
  matches the discrete Riccati terminal cost.
"""

import numpy as np
import scipy as sci
from pathlib import Path
from ..models import MODEL, MODEL_XY, CONFIG
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

BUILD_DIR = Path(__file__).parent / "acados_build"


def segway_ode() -> AcadosModel:
    model = AcadosModel()

    model.f_expl_expr = MODEL_XY.plant_explicit_acados
    model.f_impl_expr = MODEL_XY.plant_implicit_acados
    model.x = MODEL_XY.x
    model.u = MODEL_XY.u
    model.xdot = MODEL_XY.dx
    model.name = "Segway"

    # information for plotting
    model.x_labels = [
        "$s$ [m]",
        r"$\dot{s}$ [m/s]",
        r"yaw/$\theta$ [rad]",
        r"$\dot{\theta}$ [rad/s]",
        r"pitch/$\psi$ [rad]",
        r"$\dot{\psi}$ [rad/s]",
        r"$x$ [m]",
        r"$y$ [m]",
    ]

    model.u_labels = [
        r"$\tau_l$ [Nm]",
        r"$\tau_r$ [Nm]",
    ]

    model.t_label = "$t$ [s]"

    return model


# Balance subsystem: ds, dtheta, psi, dpsi (used by the fallback LQR).
# Their dynamics don't depend on s, theta, x or y (flat ground), so this is a
# closed, controllable subsystem.
BAL_IDX = [1, 3, 4, 5]
XY_IDX = [6, 7]

S_IDX = 0
YAW_IDX = 2
N_MODEL = 6  # states of the linearized model: s, ds, theta, dtheta, psi, dpsi
MAX_STEP = 5  # [m] largest distance reference ahead of the robot

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


# ============================== Model and OCP ===============================
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


def segway_ocp(Q: np.ndarray, R: np.ndarray) -> AcadosOcp:
    ocp = AcadosOcp()
    ode = segway_ode()

    ocp.model = ode
    ocp.solver_options.N_horizon = CONFIG.mpc.horizon
    ocp.solver_options.tf = CONFIG.mpc.horizon * CONFIG.mpc.T

    nx = ode.x.rows()
    nu = ode.u.rows()

    # ---------------- Terminal cost: discrete Riccati on the 6-state model ----------------
    # s and theta are weighted again (they carry the polar references), so the
    # full 6-state Riccati equation is well posed. x and y get no terminal weight.
    Ad, Bd = discretize(CONFIG.mpc.T)
    P = np.zeros((nx, nx))
    P[:N_MODEL, :N_MODEL] = sci.linalg.solve_discrete_are(
        Ad, Bd, Q[:N_MODEL, :N_MODEL], R
    )

    # ---------------- Stage cost ----------------
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.Vx = np.vstack([np.eye(nx), np.zeros((nu, nx))])
    ocp.cost.Vu = np.vstack([np.zeros((nx, nu)), np.eye(nu)])
    ocp.cost.W = sci.linalg.block_diag(Q, R)
    ocp.cost.yref = np.zeros(nx + nu)  # s_ref, theta_ref are written in every sample

    # ---------------- Terminal cost ----------------
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.W_e = P
    ocp.cost.yref_e = np.zeros(nx)

    # ---------------- State constraints (soft) ----------------
    # 1 = ds, 4 = psi, 5 = dpsi, 3=dtheta
    ocp.constraints.idxbx = np.array([1, 4, 5, 3])
    ocp.constraints.ubx = np.array(
        [
            CONFIG.mpc.displacement_rate_constraint,
            CONFIG.mpc.pitch_constraint,
            CONFIG.mpc.pitch_rate_constraint,
            CONFIG.mpc.yaw_rate_constraint,
        ]
    )
    ocp.constraints.lbx = -ocp.constraints.ubx
    ocp.constraints.idxsbx = np.arange(4)
    ocp.constraints.x0 = np.zeros(nx)

    # ---------------- Input constraints (hard) ----------------
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.ubu = CONFIG.mpc.torque_constraint * np.ones(nu)
    ocp.constraints.lbu = -ocp.constraints.ubu

    # ---------------- Slack penalties ----------------
    ns = 4
    w = CONFIG.mpc.slack_weight
    ocp.cost.Zl = ocp.cost.Zu = w * np.ones(ns)
    ocp.cost.zl = ocp.cost.zu = 10 * w * np.ones(ns)

    # Stage cost as a plain sum (acados would otherwise scale it by dt),
    # consistent with the discrete Riccati terminal cost above.
    ocp.solver_options.cost_scaling = np.ones(CONFIG.mpc.horizon + 1)

    ocp.code_export_directory = str(BUILD_DIR / "c_generated_code")

    return ocp


def segway_solver(ocp) -> AcadosOcpSolver:
    # Solver options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = (
        "ERK"  # explicit RK4: the dynamics are not stiff
    )
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # one QP per sample, warm-started
    ocp.solver_options.levenberg_marquardt = (
        1e-3  # small regularization: keeps the QPs well posed
    )

    return AcadosOcpSolver(
        ocp,
        json_file=str(BUILD_DIR / "acados_ocp_nonlinear.json"),
    )


class NonlinearMPC:
    def __init__(self, Q, R, target=(0.0, 0.0)) -> None:
        self.solver = segway_solver(segway_ocp(Q, R))
        self.N = CONFIG.mpc.horizon
        self.dt = CONFIG.mpc.T  # same dt as the UKF / control loop
        self.nx, self.nu = 8, 2
        self.position = np.zeros(2)  # dead-reckoned [x, y]
        self.plan = np.empty((0, 2))
        self.set_target(*target)

        # Backup controller: a balancing LQR, used when the solver fails, so a
        # failed solve never reaches the motors. Discrete-time, because the
        # torque is held constant for a whole control step (dt).
        A_bal = MODEL.A[np.ix_(BAL_IDX, BAL_IDX)]
        B_bal = MODEL.B[BAL_IDX, :]
        n, m = B_bal.shape
        M = sci.linalg.expm(
            np.block([[A_bal, B_bal], [np.zeros((m, n + m))]]) * self.dt
        )
        Ad, Bd = M[:n, :n], M[:n, n:]
        Q_bal = Q[np.ix_(BAL_IDX, BAL_IDX)]
        P_d = sci.linalg.solve_discrete_are(Ad, Bd, Q_bal, R)
        self.K_bal = np.linalg.solve(R + Bd.T @ P_d @ Bd, Bd.T @ P_d @ Ad)
        self.u_max = CONFIG.mpc.torque_constraint
        self.cold = True  # no usable previous solution to warm-start from

    def set_target(self, x_goal: float, y_goal: float) -> None:
        self.target = np.array([x_goal, y_goal], dtype=float)

    def get_position(self) -> np.ndarray:
        return self.position.copy()

    def get_planned_path(self) -> np.ndarray:
        return self.plan

    def set_references(self, s: float, theta: float) -> None:
        """Polar guidance: the goal as references on s and theta (x, y get no weight)."""
        s_ref, theta_ref = polar_reference(
            self.target, self.position, s, theta, MAX_STEP
        )

        diff = self.target - self.position
        dist = np.linalg.norm(diff)
        xy_ref = (
            self.target if dist <= MAX_STEP else self.position + diff / dist * MAX_STEP
        )

        yref = np.zeros(self.nx + self.nu)
        yref[S_IDX] = s_ref
        yref[YAW_IDX] = theta_ref
        yref[XY_IDX] = xy_ref
        for k in range(self.N):
            self.solver.set(k, "yref", yref)
        self.solver.set(self.N, "yref", yref[: self.nx])

    def initial_guess(self, x_full: np.ndarray) -> None:
        """Start from the previous solution shifted one step, or from standing still."""
        if self.cold:
            xs = [x_full] * (self.N + 1)
            us = [np.zeros(self.nu)] * self.N
        else:
            xs = [self.solver.get(k, "x") for k in range(1, self.N + 1)]
            xs.append(xs[-1])  # repeat the last state
            us = [self.solver.get(k, "u") for k in range(1, self.N)]
            us.append(us[-1])  # repeat the last input
            xs[0] = x_full  # first node is the measured state
        for k in range(self.N + 1):
            self.solver.set(k, "x", xs[k])
        for k in range(self.N):
            self.solver.set(k, "u", us[k])

    def compute(self, state_estimate: np.ndarray) -> np.ndarray:
        s, s_dot, theta = (
            state_estimate[S_IDX],
            state_estimate[1],
            state_estimate[YAW_IDX],
        )
        self.position += self.dt * s_dot * np.array([np.cos(theta), np.sin(theta)])

        x_full = np.concatenate([state_estimate, self.position])  # 6 -> 8
        self.initial_guess(x_full)
        self.solver.set(0, "lbx", x_full)
        self.solver.set(0, "ubx", x_full)
        self.set_references(s, theta)

        status = self.solver.solve()

        if status == 0:
            self.cold = False
            u = self.solver.get(0, "u")
            self.plan = np.array(
                [self.solver.get(k, "x")[6:8] for k in range(self.N + 1)]
            )
        else:  # QP failed: just balance, and start the next solve from standing still
            self.cold = True
            self.solver.reset()
            self.plan = np.empty((0, 2))  # no valid plan to show
            u = -self.K_bal @ state_estimate[BAL_IDX]

        return np.clip(u, -self.u_max, self.u_max)

    def reset(self) -> None:
        self.solver.reset()
        self.position[:] = 0.0
        self.plan = np.empty((0, 2))
        self.cold = True
