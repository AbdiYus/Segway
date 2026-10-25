import time

import matplotlib.pyplot as plt
import numpy as np
from acados_template import latexify_plot

from Control_System.Models.segway_configuration import CONFIG
from Simulator.control_loop import PITCH, S, YAW, ClosedLoop
from Simulator.Renderer.renderer import Renderer


def plot_segway(
    t,
    u_max,
    U,
    X_true,
    latexify=False,
    plt_show=True,
    time_label="$t$ [s]",
    x_labels=None,
    u_labels=None,
    angles_in_deg=True,
    X_est=None,
):
    """
    Params:
        t: time values, shape (N_sim+1,)
        u_max: maximum absolute torque (None to skip limit lines)
        U: array with shape (N_sim, nu)  -> [tau_l, tau_r]
        X_true: array with shape (N_sim+1, nx) -> [s, s_dot, theta, theta_dot, psi, psi_dot]
        x_labels / u_labels: e.g. ocp.model.x_labels / ocp.model.u_labels
        angles_in_deg: convert theta/psi (and rates) to degrees, labels follow
        X_est: optional UKF estimates, shape (N_sim, nx), plotted dashed
    """
    if latexify:
        latexify_plot()

    X = np.array(X_true, dtype=float)
    U = np.atleast_2d(U)

    if x_labels is None:
        x_labels = [
            r"$s$ [m]",
            r"$\dot{s}$ [m/s]",
            r"$\theta$ [rad]",
            r"$\dot{\theta}$ [rad/s]",
            r"$\psi$ [rad]",
            r"$\dot{\psi}$ [rad/s]",
        ]
    if u_labels is None:
        u_labels = [r"$\tau_l$ [Nm]", r"$\tau_r$ [Nm]"]

    x_labels = list(x_labels)
    if angles_in_deg:
        X[:, 2:] = np.degrees(X[:, 2:])
        x_labels = x_labels[:2] + [lbl.replace("[rad", "[deg") for lbl in x_labels[2:]]

    nx = X.shape[1]
    nu = U.shape[1]
    fig, axes = plt.subplots(nx + 1, 1, sharex=True, figsize=(8, 12))

    for i in range(nx):
        axes[i].plot(t, X[:, i])
        axes[i].set_ylabel(x_labels[i])
        axes[i].grid()

    # all torques on the last axis
    for j in range(nu):
        axes[-1].step(t, np.append([U[0, j]], U[:, j]), label=u_labels[j])
    axes[-1].set_ylabel(r"$\tau$ [Nm]")
    axes[-1].legend(loc="upper right")

    if u_max is not None:
        axes[-1].hlines([u_max, -u_max], t[0], t[-1], linestyles="dashed", alpha=0.7)
        axes[-1].set_ylim([-1.2 * u_max, 1.2 * u_max])

    axes[-1].set_xlim(t[0], t[-1])
    axes[-1].set_xlabel(time_label)
    axes[-1].grid()

    plt.subplots_adjust(hspace=0.4)
    fig.align_ylabels()

    if plt_show:
        plt.show()


class RealTimePacer:
    """
    Turns wall-clock time into a whole number of fixed physics steps.

    The physics always uses the same dt (stable, repeatable), while the window
    redraws whenever it can. If a frame is late, the next one simply takes
    more physics steps to catch up.
    """

    MAX_CATCH_UP = (
        0.1  # [s] never try to catch up more than this (e.g. after dragging the window)
    )

    def __init__(self, physics_dt: float, time_scale: float = 1.0) -> None:
        self.physics_dt = physics_dt
        self.time_scale = time_scale  # 0.5 = half speed, nice for watching the recovery
        self.last_wall_time = time.perf_counter()
        self.unsimulated_time = 0.0

    def steps_due(self) -> int:
        now = time.perf_counter()
        elapsed = min(now - self.last_wall_time, self.MAX_CATCH_UP)
        self.last_wall_time = now

        self.unsimulated_time += elapsed * self.time_scale
        steps = int(self.unsimulated_time / self.physics_dt)
        self.unsimulated_time -= steps * self.physics_dt
        return steps


def status_text(loop: ClosedLoop) -> str:
    true_pitch = np.degrees(loop.state[PITCH])
    estimated_pitch = np.degrees(loop.ukf.x[PITCH])
    tau_l, tau_r = loop.torque
    motors = "ON" if loop.motors_enabled else "OFF"
    return (
        f"t = {loop.time:6.2f} s\n"
        f"pitch      {true_pitch:6.1f} deg\n"
        f"pitch (UKF) {estimated_pitch:6.1f} deg\n"
        f"tau_l = {tau_l:5.2f} Nm   tau_r = {tau_r:5.2f} Nm\n"
        f"motors {motors}   acados status {loop.solver_status}\n"
        "\n"
        "Right/Left: push   t: tilt 10 deg   m: motors on/off"
    )


def run_live(
    initial_pitch_deg: float = 20.0,
    use_ukf: bool = True,
    time_scale: float = 1.0,
    plot_afterwards: bool = True,
) -> None:
    initial_state = np.array([0, 0, 0, 0, np.radians(initial_pitch_deg), 0])
    loop = ClosedLoop(initial_state, physics_dt=0.01, use_ukf=use_ukf)
    renderer = Renderer(CONFIG.segway)
    pacer = RealTimePacer(loop.physics_dt, time_scale)

    def on_frame() -> None:
        for _ in range(pacer.steps_due()):
            loop.step()

        x, y = loop.position
        renderer.update_pose(
            x, y, yaw=loop.state[YAW], pitch=loop.state[PITCH], distance=loop.state[S]
        )
        renderer.follow(x, y)
        renderer.show_info(status_text(loop))

    renderer.add_key("Right", lambda: loop.push(pitch_rate_change=+1.0))
    renderer.add_key("Left", lambda: loop.push(pitch_rate_change=-1.0))
    renderer.add_key("t", lambda: loop.tilt_to(np.radians(10)))
    renderer.add_key("m", lambda: loop.set_motors_enabled(not loop.motors_enabled))

    renderer.run(on_frame, frame_dt=1 / 60)  # blocks until the window is closed

    if plot_afterwards:
        t, true_states, _, torques = loop.history.as_arrays()
        plot_segway(
            t,
            u_max=CONFIG.mpc.torque_constraint,
            U=torques[:-1],
            X_true=true_states,
        )


if __name__ == "__main__":
    run_live()


# import numpy as np
# from Simulator.UKF import SegwayUKF
# import matplotlib.pyplot as plt
# from acados_template import latexify_plot
# from Control_System.Models.segway_sensors import measure
# from Control_System.Models.segway_configuration import CONFIG
# from Control_System.MPC import segway_integrator, segway_ocp, segway_solver
#
#
# def plot_segway(
#     t,
#     u_max,
#     U,
#     X_true,
#     latexify=False,
#     plt_show=True,
#     time_label="$t$ [s]",
#     x_labels=None,
#     u_labels=None,
#     angles_in_deg=True,
#     X_est=None,
# ):
#     """
#     Params:
#         t: time values, shape (N_sim+1,)
#         u_max: maximum absolute torque (None to skip limit lines)
#         U: array with shape (N_sim, nu)  -> [tau_l, tau_r]
#         X_true: array with shape (N_sim+1, nx) -> [s, s_dot, theta, theta_dot, psi, psi_dot]
#         x_labels / u_labels: e.g. ocp.model.x_labels / ocp.model.u_labels
#         angles_in_deg: convert theta/psi (and rates) to degrees, labels follow
#         X_est: optional UKF estimates, shape (N_sim, nx), plotted dashed
#     """
#     if latexify:
#         latexify_plot()
#
#     X = np.array(X_true, dtype=float)
#     U = np.atleast_2d(U)
#
#     if x_labels is None:
#         x_labels = [
#             r"$s$ [m]",
#             r"$\dot{s}$ [m/s]",
#             r"$\theta$ [rad]",
#             r"$\dot{\theta}$ [rad/s]",
#             r"$\psi$ [rad]",
#             r"$\dot{\psi}$ [rad/s]",
#         ]
#     if u_labels is None:
#         u_labels = [r"$\tau_l$ [Nm]", r"$\tau_r$ [Nm]"]
#
#     x_labels = list(x_labels)
#     if angles_in_deg:
#         X[:, 2:] = np.degrees(X[:, 2:])
#         x_labels = x_labels[:2] + [lbl.replace("[rad", "[deg") for lbl in x_labels[2:]]
#
#     nx = X.shape[1]
#     nu = U.shape[1]
#     fig, axes = plt.subplots(nx + 1, 1, sharex=True, figsize=(8, 12))
#
#     for i in range(nx):
#         axes[i].plot(t, X[:, i])
#         axes[i].set_ylabel(x_labels[i])
#         axes[i].grid()
#
#     # all torques on the last axis
#     for j in range(nu):
#         axes[-1].step(t, np.append([U[0, j]], U[:, j]), label=u_labels[j])
#     axes[-1].set_ylabel(r"$\tau$ [Nm]")
#     axes[-1].legend(loc="upper right")
#
#     if u_max is not None:
#         axes[-1].hlines([u_max, -u_max], t[0], t[-1], linestyles="dashed", alpha=0.7)
#         axes[-1].set_ylim([-1.2 * u_max, 1.2 * u_max])
#
#     axes[-1].set_xlim(t[0], t[-1])
#     axes[-1].set_xlabel(time_label)
#     axes[-1].grid()
#
#     plt.subplots_adjust(hspace=0.4)
#     fig.align_ylabels()
#
#     if plt_show:
#         plt.show()
#
#
# def sim(T_total=8.0, use_ukf=True, seed=0):
#     x0 = np.array([0, 0, 0, 0, np.radians(20), 0])
#     rng = np.random.default_rng(seed)
#
#     ocp = segway_ocp()
#     solver = segway_solver(ocp)
#     integrator = segway_integrator(ocp)
#
#     nx = ocp.dims.nx
#     nu = ocp.dims.nu
#
#     Ts = CONFIG.mpc.T  # one MPC step = one integrator step (tf / N)
#     N_sim = int(round(T_total / Ts))
#
#     simX = np.zeros((N_sim + 1, nx))
#     simU = np.zeros((N_sim, nu))
#     simX[0, :] = x0
#     estX = np.zeros((N_sim, nx))
#     nis = np.zeros(N_sim)  # Normalised Innovation Squared
#     simX[0, :] = x0
#
#     ukf = SegwayUKF(dt=Ts)  # starts at x_hat = 0: it does NOT know about the 20 deg
#     u_prev = np.zeros(nu)
#
#     for i in range(N_sim):
#         # 1. sensors read the true state 2. UKF corrects
#         z = measure(simX[i, :], u_prev, rng)
#
#         # 2. UKF correction
#         ukf.update(z, u_prev)
#         estX[i, :] = ukf.x
#         nis[i] = ukf.nis
#
#         # 3. MPC solves from the estimate (or the true state)
#         x_mpc = ukf.x if use_ukf else simX[i, :]
#
#         solver.set(0, "lbx", x_mpc)
#         solver.set(0, "ubx", x_mpc)
#
#         # solver.set(0, "lbx", simX[i, :])
#         # solver.set(0, "ubx", simX[i, :])
#         status = solver.solve()
#         simU[i, :] = solver.get(0, "u")
#         if status != 0:
#             print(f"step {i}: acados status {status}")
#
#         # 4. plant
#         simX[i + 1, :] = integrator.simulate(x=simX[i, :], u=simU[i, :])
#
#         # 5. UKF predicts with the torque that was applied
#         ukf.predict(simU[i, :])
#         u_prev = simU[i, :]
#
#     err = estX - simX[:-1]
#     print("UKF RMS error per state [s, ds, theta, dtheta, psi, dpsi]:")
#     print("  ", np.array2string(np.sqrt(np.mean(err**2, axis=0)), precision=4))
#     print(f"Mean NIS: {nis.mean():.2f} (about 6 if consistent)")
#
#     t = np.linspace(0, N_sim * Ts, N_sim + 1)
#     plot_segway(
#         t,
#         u_max=CONFIG.mpc.torque_constraint,
#         U=simU,
#         X_true=simX,
#         x_labels=ocp.model.x_labels,
#         u_labels=ocp.model.u_labels,
#         time_label=ocp.model.t_label,
#         X_est=estX,
#     )
#
#
# if __name__ == "__main__":
#     sim()
