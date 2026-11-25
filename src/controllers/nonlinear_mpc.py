import numpy as np
import scipy as sci
from pathlib import Path
from ..models import MODEL, MODEL_XY, CONFIG
from acados_template.acados_sim import AcadosSim
from acados_template.acados_sim_solver import AcadosSimSolver
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


# Balance subsystem: ds, dtheta, psi, dpsi.
# Their dynamics don't depend on s, theta, x or y (flat ground), so this is a
# closed, controllable subsystem and the Riccati solve is well posed.
BAL_IDX = [1, 3, 4, 5]
POS_IDX = [6, 7]  # x, y
TERMINAL_SLACK_WEIGHT = 1e4  # penalty for missing the goal at the end of the horizon


def segway_ocp(Q: np.ndarray, R: np.ndarray) -> AcadosOcp:
    ocp = AcadosOcp()
    ode = segway_ode()

    ocp.model = ode
    ocp.solver_options.N_horizon = CONFIG.mpc.horizon
    ocp.solver_options.tf = CONFIG.mpc.horizon * CONFIG.mpc.T

    nx = ode.x.rows()
    nu = ode.u.rows()

    # ---------------- Terminal cost: Riccati on the balance states only ----------------
    A, B = MODEL.A, MODEL.B  # 6x6 model: s, ds, theta, dtheta, psi, dpsi
    Q_bal = Q[np.ix_(BAL_IDX, BAL_IDX)]
    P = np.zeros((nx, nx))
    P[np.ix_(BAL_IDX, BAL_IDX)] = sci.linalg.solve_continuous_are(
        A[np.ix_(BAL_IDX, BAL_IDX)], B[BAL_IDX, :], Q_bal, R
    )
    # No terminal weight on x, y: the terminal constraint handles them.

    # ---------------- Stage cost ----------------
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.Vx = np.vstack([np.eye(nx), np.zeros((nu, nx))])
    ocp.cost.Vu = np.vstack([np.zeros((nx, nu)), np.eye(nu)])
    ocp.cost.W = sci.linalg.block_diag(Q, R)
    ocp.cost.yref = np.zeros(nx + nu)  # goal is written in by set_target

    # ---------------- Terminal cost ----------------
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.W_e = P
    ocp.cost.yref_e = np.zeros(nx)

    # ---------------- State constraints (soft) ----------------
    # 1 = ds, 4 = psi, 5 = dpsi
    ocp.constraints.idxbx = np.array([1, 4, 5])
    ocp.constraints.ubx = np.array(
        [
            CONFIG.mpc.displacement_rate_constraint,
            CONFIG.mpc.pitch_constraint,
            CONFIG.mpc.pitch_rate_constraint,
        ]
    )
    ocp.constraints.lbx = -ocp.constraints.ubx
    ocp.constraints.idxsbx = np.arange(3)
    ocp.constraints.x0 = np.zeros(nx)

    # ---------------- Terminal constraint: x_N = x_g, y_N = y_g (soft) ----------------
    ocp.constraints.idxbx_e = np.array(POS_IDX)
    ocp.constraints.lbx_e = np.zeros(2)  # goal is written in by set_target
    ocp.constraints.ubx_e = np.zeros(2)
    ocp.constraints.idxsbx_e = np.arange(
        2
    )  # remove this line (and the _e penalties) for a hard constraint

    # ---------------- Input constraints (hard) ----------------
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.ubu = CONFIG.mpc.torque_constraint * np.ones(nu)
    ocp.constraints.lbu = -ocp.constraints.ubu

    # ---------------- Slack penalties ----------------
    ns = 3
    w = CONFIG.mpc.slack_weight
    ocp.cost.Zl = ocp.cost.Zu = w * np.ones(ns)
    ocp.cost.zl = ocp.cost.zu = 10 * w * np.ones(ns)

    ns_e = 2
    ocp.cost.Zl_e = ocp.cost.Zu_e = TERMINAL_SLACK_WEIGHT * np.ones(ns_e)
    ocp.cost.zl_e = ocp.cost.zu_e = TERMINAL_SLACK_WEIGHT * np.ones(ns_e)

    ocp.code_export_directory = str(BUILD_DIR / "c_generated_code")

    return ocp


def segway_solver(ocp) -> AcadosOcpSolver:
    # Solver options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.qp_solver_iter_max = 100  # default 50 is sometimes too few
    ocp.solver_options.levenberg_marquardt = (
        1e-3  # small regularization: keeps the QPs well posed
    )

    return AcadosOcpSolver(
        ocp,
        json_file=str(BUILD_DIR / "acados_ocp_nonlinear.json"),
    )


def segway_plant_model() -> AcadosModel:
    model = AcadosModel()
    model.f_expl_expr = MODEL.plant_explicit_acados
    model.f_impl_expr = MODEL.plant_implicit_acados
    model.x = MODEL.x
    model.u = MODEL.u
    model.xdot = MODEL.dx
    model.name = "SegwayPlant"  # different name, so the generated C code doesn't clash with the OCP
    return model


def segway_integrator(ocp: AcadosOcp) -> AcadosSimSolver:
    sim = AcadosSim()
    sim.model = segway_plant_model()
    sim.solver_options.T = (
        ocp.solver_options.tf / ocp.solver_options.N_horizon
    )  # same step as before
    sim.solver_options.integrator_type = "IRK"
    sim.solver_options.num_steps = 2
    sim.code_export_directory = str(BUILD_DIR / "c_generated_code_plant")
    return AcadosSimSolver(sim, json_file=str(BUILD_DIR / "acados_sim_plant.json"))


WAYPOINT_DISTANCE = 0.2  # [m] how far ahead the terminal target may be placed


class NonlinearMPC:
    def __init__(self, Q, R, target=(0.0, 0.0)) -> None:
        self.solver = segway_solver(segway_ocp(Q, R))
        self.N = CONFIG.mpc.horizon
        self.dt = CONFIG.mpc.T  # same dt as the UKF / control loop
        self.position = np.zeros(2)  # dead-reckoned [x, y]
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

    def update_waypoint(self) -> None:
        """Terminal target: the goal, or a point WAYPOINT_DISTANCE towards it if it is farther."""
        diff = self.target - self.position
        dist = np.linalg.norm(diff)
        waypoint = (
            self.target
            if dist <= WAYPOINT_DISTANCE
            else self.position + diff / dist * WAYPOINT_DISTANCE
        )

        yref = np.zeros(8 + 2)
        yref[POS_IDX] = waypoint
        for k in range(self.N):
            self.solver.set(k, "yref", yref)
        self.solver.set(self.N, "yref", yref[:8])
        self.solver.constraints_set(self.N, "lbx", waypoint)
        self.solver.constraints_set(self.N, "ubx", waypoint)

    def compute(self, state_estimate: np.ndarray) -> np.ndarray:
        s_dot, theta = state_estimate[1], state_estimate[2]
        self.position += self.dt * s_dot * np.array([np.cos(theta), np.sin(theta)])

        x_full = np.concatenate([state_estimate, self.position])  # 6 -> 8
        self.solver.set(0, "lbx", x_full)
        self.solver.set(0, "ubx", x_full)
        self.update_waypoint()
        if self.cold:  # initial guess: stay where we are (better than all zeros)
            for k in range(self.N + 1):
                self.solver.set(k, "x", x_full)
            for k in range(self.N):
                self.solver.set(k, "u", np.zeros(2))
        status = self.solver.solve()
        self.cold = status not in (0, 2)

        if status in (0, 2):  # 0 = converged, 2 = max iterations (still usable)
            u = self.solver.get(0, "u")
        else:  # QP failed: just balance, and start the next solve fresh
            u = -self.K_bal @ state_estimate[BAL_IDX]
            self.solver.reset()
        return np.clip(u, -self.u_max, self.u_max)

    def reset(self) -> None:
        self.solver.reset()
        self.position[:] = 0.0
        self.cold = True
