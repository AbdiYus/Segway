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


def segway_ocp(Q: np.ndarray, R: np.ndarray) -> AcadosOcp:
    ocp = AcadosOcp()
    ode = segway_ode()

    ocp.model = ode
    ocp.solver_options.N_horizon = CONFIG.mpc.horizon
    ocp.solver_options.tf = CONFIG.mpc.horizon * CONFIG.mpc.T

    nx = ode.x.rows()
    nu = ode.u.rows()

    # Terminal cost: Riccati on the balance part, hand-picked weight on x/y
    A, B = MODEL_XY.A, MODEL_XY.B
    idx = [1, 2, 3, 4, 5]  # ds, theta, dtheta, psi, dpsi
    Q_red = Q[np.ix_(idx, idx)].copy()
    Q_red[1, 1] = max(
        Q_red[1, 1], 1e-3
    )  # theta needs a tiny weight, or the Riccati solve can fail

    P = np.zeros((nx, nx))
    P[np.ix_(idx, idx)] = sci.linalg.solve_continuous_are(
        A[np.ix_(idx, idx)], B[idx, :], Q_red, R
    )
    P[6, 6] = P[7, 7] = 10 * Q[6, 6]  # terminal x/y weight, tune this

    # Stage cost
    ocp.cost.cost_type = "LINEAR_LS"  # NOTE: is default, but to be explicit
    ocp.cost.Vx = np.vstack([np.eye(nx), np.zeros((nu, nx))])
    ocp.cost.Vu = np.vstack([np.zeros((nx, nu)), np.eye(nu)])
    ocp.cost.W = sci.linalg.block_diag(Q, R)
    ocp.cost.yref = np.zeros(nx + nu)

    # Terminal cost
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

    # ---------------- Input constraints (hard) ----------------
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.ubu = CONFIG.mpc.torque_constraint * np.ones(nu)
    ocp.constraints.lbu = -ocp.constraints.ubu

    # ---------------- Slack penalties ----------------
    ns = 3
    w = CONFIG.mpc.slack_weight
    ocp.cost.Zl = ocp.cost.Zu = w * np.ones(ns)
    ocp.cost.zl = ocp.cost.zu = 10 * w * np.ones(ns)

    ocp.code_export_directory = str(BUILD_DIR / "c_generated_code")

    return ocp


def segway_solver(ocp) -> AcadosOcpSolver:
    # Solver options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"

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


class NonlinearMPC:
    def __init__(self, Q, R, target=(0.0, 0.0)) -> None:
        self.solver = segway_solver(segway_ocp(Q, R))
        self.N = CONFIG.mpc.horizon
        self.dt = CONFIG.mpc.T  # same dt as the UKF / control loop
        self.position = np.zeros(2)  # dead-reckoned [x, y]
        self.set_target(*target)

    def set_target(self, x_ref: float, y_ref: float) -> None:
        self.target = (x_ref, y_ref)
        yref = np.zeros(8 + 2)
        yref[6], yref[7] = x_ref, y_ref
        for k in range(self.N):
            self.solver.set(k, "yref", yref)
        self.solver.set(self.N, "yref", yref[:8])

    def compute(self, state_estimate: np.ndarray) -> np.ndarray:
        s_dot, theta = state_estimate[1], state_estimate[2]
        self.position += self.dt * s_dot * np.array([np.cos(theta), np.sin(theta)])

        x_full = np.concatenate([state_estimate, self.position])  # 6 -> 8
        self.solver.set(0, "lbx", x_full)
        self.solver.set(0, "ubx", x_full)
        self.solver.solve()
        return self.solver.get(0, "u")

    def reset(self) -> None:
        self.solver.reset()
        self.position[:] = 0.0
        self.set_target(*self.target)
