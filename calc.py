from dataclasses import dataclass
import sympy as sp

sp.init_printing(use_unicode=True)


@dataclass
class Energy:
    kinetic: sp.Expr
    potential: sp.Expr


#############################################################
#
#                SYMBOLS AND DEFINITIONS
#
#############################################################
t = sp.symbols("t")

s = sp.Function("s")(t)
yaw = sp.Function("theta")(t)
pitch = sp.Function("psi")(t)
AXLE_LENGTH, BEAM_LENGTH = sp.symbols("L_1 L_2")
WHEEL_RADIUS = sp.symbols("h")
torque_left, torque_right = sp.symbols(r"tau_l tau_r ")
m_axle, m_beam, m_wheel = sp.symbols("m_axle m_beam m_wheel")

yaw_rate = yaw.diff(t)
pitch_rate = pitch.diff(t)
s_dot = s.diff(t)

g = sp.Rational(981, 100)

#############################################################
#
#                     GENERIC FUNCTIONS
#
#############################################################


def pretty_print_eq(name, expr, t, s=None, yaw=None, pitch=None):
    """
    Pretty-print a SymPy expression in terminal form.

    Example:
        pretty_print_eq("v_l²", v_left_squared, t, s, yaw, pitch)
    """

    substitutions = {}

    if s is not None:
        substitutions.update(
            {
                s.diff(t): sp.Symbol("s_dot"),
                s.diff(t, 2): sp.Symbol("s_ddot"),
            }
        )

    if yaw is not None:
        substitutions.update(
            {
                yaw.diff(t): sp.Symbol("theta_dot"),
                yaw.diff(t, 2): sp.Symbol("theta_ddot"),
            }
        )

    if pitch is not None:
        substitutions.update(
            {
                pitch.diff(t): sp.Symbol("psi_dot"),
                pitch.diff(t, 2): sp.Symbol("psi_ddot"),
            }
        )

    expr_display = sp.simplify(expr.subs(substitutions))

    lhs = sp.Symbol(name)

    sp.pprint(sp.Eq(lhs, expr_display), use_unicode=True)


def pretty_print_matrix(name, matrix, t, s=None, yaw=None, pitch=None):
    substitutions = {}

    if s is not None:
        substitutions.update(
            {
                s.diff(t): sp.Symbol("s_dot"),
                s.diff(t, 2): sp.Symbol("s_ddot"),
            }
        )

    if yaw is not None:
        substitutions.update(
            {
                yaw.diff(t): sp.Symbol("theta_dot"),
                yaw.diff(t, 2): sp.Symbol("theta_ddot"),
            }
        )

    if pitch is not None:
        substitutions.update(
            {
                pitch.diff(t): sp.Symbol("psi_dot"),
                pitch.diff(t, 2): sp.Symbol("psi_ddot"),
            }
        )

    matrix_display = matrix.subs(substitutions)

    print(f"\n{name} =")
    sp.pprint(matrix_display, use_unicode=True)


def pretty_print_second_order_ode(M, b, t, s, yaw, pitch):
    s_ddot = sp.Symbol("s_ddot")
    theta_ddot = sp.Symbol("theta_ddot")
    psi_ddot = sp.Symbol("psi_ddot")

    q_ddot = sp.Matrix(
        [
            s_ddot,
            theta_ddot,
            psi_ddot,
        ]
    )

    substitutions = {
        s.diff(t): sp.Symbol("s_dot"),
        s.diff(t, 2): s_ddot,
        yaw.diff(t): sp.Symbol("theta_dot"),
        yaw.diff(t, 2): theta_ddot,
        pitch.diff(t): sp.Symbol("psi_dot"),
        pitch.diff(t, 2): psi_ddot,
    }

    M_display = M.subs(substitutions)
    b_display = b.subs(substitutions)

    print("\nM =")
    sp.pprint(M_display, use_unicode=True)

    print("\nq_ddot =")
    sp.pprint(q_ddot, use_unicode=True)

    print("\nb =")
    sp.pprint(b_display, use_unicode=True)

    print("\nM * q_ddot = b")


#############################################################
#
#         GENERALIZED FORCES AND VELOCITES VECTORS
#
#############################################################

# beam_position_world = sp.Matrix(
#     [
#         [x + (BEAM_LENGTH / 2) * sp.sin(pitch) * sp.cos(yaw)],
#         [y + (BEAM_LENGTH / 2) * sp.sin(pitch) * sp.sin(yaw)],
#         [WHEEL_RADIUS + (BEAM_LENGTH / 2) * sp.cos(pitch)],
#     ]
# )
#
# axle_position_world = sp.Matrix(
#     [
#         [x],
#         [y],
#         [WHEEL_RADIUS],
#     ]
# )
#
# right_wheel_position_world = sp.Matrix(
#     [
#         [x + (AXLE_LENGTH / 2) * sp.sin(yaw)],
#         [y - (AXLE_LENGTH / 2) * sp.cos(yaw)],
#         [WHEEL_RADIUS],
#     ]
# )
#
# left_wheel_position_world = sp.Matrix(
#     [
#         [x - (AXLE_LENGTH / 2) * sp.sin(yaw)],
#         [y + (AXLE_LENGTH / 2) * sp.cos(yaw)],
#         [WHEEL_RADIUS],
#     ]
# )


v_axle_world = sp.Matrix(
    [
        [s_dot * sp.cos(yaw)],
        [s_dot * sp.sin(yaw)],
        [0],
    ]
)
v_axle_squared = sp.simplify((v_axle_world.T * v_axle_world)[0, 0])

r_beam_axle = sp.Matrix(
    [
        [(BEAM_LENGTH / 2) * sp.sin(pitch) * sp.cos(yaw)],
        [(BEAM_LENGTH / 2) * sp.sin(pitch) * sp.sin(yaw)],
        [(BEAM_LENGTH / 2) * sp.cos(pitch)],
    ]
)
v_beam_world = v_axle_world + r_beam_axle.diff(t)
v_beam_squared = sp.simplify((v_beam_world.T * v_beam_world)[0, 0])

forward_vector = sp.Matrix(
    [
        [sp.cos(yaw)],
        [sp.sin(yaw)],
        [0],
    ]
)

r_right_axle = sp.Matrix(
    [
        [(AXLE_LENGTH / 2) * sp.sin(yaw)],
        [-(AXLE_LENGTH / 2) * sp.cos(yaw)],
        [0],
    ]
)
v_right_world = v_axle_world + r_right_axle.diff(t)
v_right_squared = sp.simplify((v_right_world.T * v_right_world)[0, 0])
v_right_forward = sp.simplify((v_right_world.T * forward_vector)[0, 0])


r_left_axle = sp.Matrix(
    [
        [-(AXLE_LENGTH / 2) * sp.sin(yaw)],
        [(AXLE_LENGTH / 2) * sp.cos(yaw)],
        [0],
    ]
)
v_left_world = v_axle_world + r_left_axle.diff(t)
v_left_squared = sp.simplify((v_left_world.T * v_left_world)[0, 0])
v_left_forward = sp.simplify((v_left_world.T * forward_vector)[0, 0])

print("\nSquared Velocites:")
pretty_print_eq("v_l", v_left_squared, t, s, yaw, pitch)
pretty_print_eq("v_r", v_right_squared, t, s, yaw, pitch)
pretty_print_eq("v_lf", v_left_forward, t, s, yaw, pitch)
pretty_print_eq("v_rf", v_right_forward, t, s, yaw, pitch)
pretty_print_eq("v_a^2", v_beam_squared, t, s, yaw, pitch)
pretty_print_eq("v_b^2", v_axle_squared, t, s, yaw, pitch)


Q_pitch = -(torque_right + torque_left)

Q_yaw = (AXLE_LENGTH / (2 * WHEEL_RADIUS)) * (torque_right - torque_left)

Q_s = (torque_left + torque_right) / WHEEL_RADIUS

print("\nGeneralized Forces:")
pretty_print_eq("Q_psi", Q_pitch, t, s, yaw, pitch)
pretty_print_eq("Q_theta", Q_yaw, t, s, yaw, pitch)
pretty_print_eq("Q_s", Q_s, t, s, yaw, pitch)

#############################################################
#
#               KINETIC AND POTENTIAL ENERGY
#
#############################################################

I_axle = sp.Rational(1, 12) * m_axle * (AXLE_LENGTH**2)

T_axle = (
    sp.Rational(1, 2) * m_axle * v_axle_squared
    + sp.Rational(1, 2) * I_axle * yaw_rate**2
)

V_axle = m_axle * g * WHEEL_RADIUS

energy_axle = Energy(T_axle, V_axle)

##############################################################

I_beam = sp.Rational(1, 12) * m_beam * (BEAM_LENGTH**2)


omega_beam_squared = pitch_rate**2 + (yaw_rate * sp.sin(pitch)) ** 2

T_beam = (
    sp.Rational(1, 2) * m_beam * v_beam_squared
    + sp.Rational(1, 2) * I_beam * omega_beam_squared
)

V_beam = m_beam * g * (WHEEL_RADIUS + (BEAM_LENGTH / 2) * sp.cos(pitch))

energy_beam = Energy(T_beam, V_beam)

##############################################################

I_wheel = sp.Rational(1, 2) * m_wheel * (WHEEL_RADIUS**2)


omega_right = v_right_forward / WHEEL_RADIUS

T_right = (
    sp.Rational(1, 2) * m_wheel * v_right_squared
    + sp.Rational(1, 2) * I_wheel * omega_right**2
)

V_right = m_wheel * g * WHEEL_RADIUS

energy_right_wheel = Energy(T_right, V_right)

##############################################################

omega_left = v_left_forward / WHEEL_RADIUS

T_left = (
    sp.Rational(1, 2) * m_wheel * v_left_squared
    + sp.Rational(1, 2) * I_wheel * omega_left**2
)

V_left = m_wheel * g * WHEEL_RADIUS

energy_left_wheel = Energy(T_left, V_left)

print("\nKinetic and Potential Energy:")
pretty_print_eq("T_a", T_axle, t, s, yaw, pitch)
pretty_print_eq("V_a", V_axle, t, s, yaw, pitch)

pretty_print_eq("T_b", T_beam, t, s, yaw, pitch)
pretty_print_eq("V_b", V_beam, t, s, yaw, pitch)

pretty_print_eq("T_l", T_left, t, s, yaw, pitch)
pretty_print_eq("V_l", V_left, t, s, yaw, pitch)

pretty_print_eq("T_r", T_right, t, s, yaw, pitch)
pretty_print_eq("V_r", V_right, t, s, yaw, pitch)

#############################################################
#
#               EULER-LAGRANGE EQUATIONS
#
#############################################################

T_total = (
    energy_beam.kinetic
    + energy_axle.kinetic
    + energy_right_wheel.kinetic
    + energy_left_wheel.kinetic
)

V_total = (
    energy_beam.potential
    + energy_axle.potential
    + energy_right_wheel.potential
    + energy_left_wheel.potential
)

L = T_total - V_total


def euler_lagrange(L, q, Q):
    q_dot = q.diff(t)
    L_q_dot = L.diff(q_dot)
    L_q = L.diff(q)
    return sp.simplify(L_q_dot.diff(t) - L_q - Q)


equation_s = euler_lagrange(L, s, Q_s)
equation_pitch = euler_lagrange(L, pitch, Q_pitch)
equation_yaw = euler_lagrange(L, yaw, Q_yaw)

print("\nEuler-Lagrange Equations:")
pretty_print_eq("Eq_s", equation_s, t, s, yaw, pitch)
pretty_print_eq("Eq_psi", equation_pitch, t, s, yaw, pitch)
pretty_print_eq("Eq_theta", equation_yaw, t, s, yaw, pitch)

#############################################################
#
#                   SECOND ORDER ODE
#
#############################################################

accelerations = [
    s.diff(t, 2),
    yaw.diff(t, 2),
    pitch.diff(t, 2),
]

equations = [
    equation_s,
    equation_yaw,
    equation_pitch,
]

M, b = sp.linear_eq_to_matrix(equations, *accelerations)

print("\nSecond order ODE's:")
pretty_print_second_order_ode(M, b, t, s, yaw, pitch)


#############################################################
#
#               FINDING STATE SPACE EXPRESSION
#
#############################################################


q_ddot = sp.simplify(M.LUsolve(b))  # solves Mq_ddot = b

print("\nSolved Accelerations Equations:")
pretty_print_matrix("q_ddot", q_ddot, t, s, yaw, pitch)

x1, x2, x3, x4, x5, x6 = sp.symbols("x_1 x_2 x_3 x_4 x_5 x_6")

state_subs = {
    s: x1,
    s.diff(t): x2,
    yaw: x3,
    yaw.diff(t): x4,
    pitch: x5,
    pitch.diff(t): x6,
}

f_s = sp.simplify(q_ddot[0].subs(state_subs))
f_yaw = sp.simplify(q_ddot[1].subs(state_subs))
f_pitch = sp.simplify(q_ddot[2].subs(state_subs))

f = sp.Matrix([x2, f_s, x4, f_yaw, x6, f_pitch])

print("\nNonlinear state space:")
pretty_print_matrix("f(x,u)", f, t, s, yaw, pitch)

constants = {
    m_axle: 2,
    m_beam: 4,
    m_wheel: 1,
    AXLE_LENGTH: 3,
    BEAM_LENGTH: 6,
    WHEEL_RADIUS: sp.Rational(1, 2),
}

x_vec = sp.Matrix([x1, x2, x3, x4, x5, x6])
u_vec = sp.Matrix([torque_left, torque_right])

A = f.jacobian(x_vec)
B = f.jacobian(u_vec)

print("\nLinear State Space:")
pretty_print_matrix("A", A.subs(constants), t, s, yaw, pitch)
pretty_print_matrix("B", B.subs(constants), t, s, yaw, pitch)
