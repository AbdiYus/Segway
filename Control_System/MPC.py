from Models.segway_dynamics import MODEL
from Models.segway_configuration import CONFIG
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import numpy as np
import casadi as ca
import scipy as sci

from acados_template.acados_sim import AcadosSim
from acados_template.acados_sim_solver import AcadosSimSolver

MODEL.plant


def segway_ode() -> AcadosModel:
    model = AcadosModel()

    model.f_expl_expr = MODEL.plant_explicit_acados
    model.f_impl_expr = MODEL.plant_implicit_acados
    model.x = MODEL.x
    model.u = MODEL.u
    model.xdot = MODEL.dx
    model.name = "Segway"

    # information for plotting
    model.x_labels = [
        "$s$ [m]",
        r"$\dot{s}$ [m/s]",
        r"yaw/$\theta$ [rad]",
        r"$\dot{\theta}$ [rad/s]",
        r"pitch/$\psi$ [rad]",
        r"$\dot{\psi}$ [rad/s]",
    ]

    model.u_labels = [
        r"$\tau_l$ [Nm]",
        r"$\tau_r$ [Nm]",
    ]

    model.t_label = "$t$ [s]"

    return model


def segway_ocp() -> AcadosOcp:
    ocp = AcadosOcp()
    ode = segway_ode()

    ocp.model = ode
    ocp.solver_options.N_horizon = CONFIG.mpc.horizon
    ocp.solver_options.tf = CONFIG.mpc.horizon * CONFIG.mpc.T

    nx = ode.x.rows()
    nu = ode.u.rows()

    Q = np.diag([1.0, 0.1, 1.0, 0.1, 100.0, 1.0])  # [s, ds, theta, dtheta, psi, dpsi]
    R = np.diag([0.01, 0.01])
    P = sci.linalg.solve_continuous_are(MODEL.A, MODEL.B, Q, R)

    # Stage cost
    ocp.cost.cost_type = "LINEAR_LS"  # NOTE: is default, but to be explicit
    ocp.cost.Vx = np.vstack([np.eye(nx), np.zeros((nu, nx))])
    ocp.cost.Vu = np.vstack([np.zeros((nx, nu)), np.eye(nu)])
    ocp.cost.W = ca.diagcat(Q, R).full()
    ocp.cost.yref = np.zeros(nx + nu)

    # Terminal cost
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.W_e = P
    ocp.cost.yref_e = np.zeros(nx)

    # Set constraints

    # State constraints
    ocp.constraints.lbx = np.array([-CONFIG.mpc.pitch_constraint])
    ocp.constraints.ubx = np.array([CONFIG.mpc.pitch_constraint])
    ocp.constraints.idxbx = np.array([4])  # [psi / pitch]
    ocp.constraints.x0 = np.zeros(nx)

    # Input constriants
    ocp.constraints.lbu = -CONFIG.mpc.torque_constraint * np.ones(nu)
    ocp.constraints.ubu = CONFIG.mpc.torque_constraint * np.ones(nu)
    ocp.constraints.idxbu = np.array([0, 1])  # [torque_left, torque_right]

    return ocp


def segway_solver(ocp) -> AcadosOcpSolver:
    # Solver options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"

    return AcadosOcpSolver(ocp)


def segway_integrator(ocp: AcadosOcp) -> AcadosSimSolver:
    simulator = AcadosSim.from_ocp(ocp)

    simulator.solver_options.num_steps = 2
    integrator = AcadosSimSolver(simulator)
    return integrator


def plant(x, u):
    return MODEL.plant(x, u)
