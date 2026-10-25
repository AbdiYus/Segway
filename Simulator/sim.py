import matplotlib.pyplot as plt
import numpy as np
from acados_template import latexify_plot

from Control_System.Models.segway_configuration import CONFIG
from Simulator.control_loop import PITCH, S, YAW, ClosedLoop, RealTimePacer
from Simulator.Renderer.control_panel import ControlPanel
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


def run_live(
    initial_pitch_deg: float = 20.0,
    use_ukf: bool = True,
    plot_afterwards: bool = True,
) -> None:
    initial_state = np.array([0, 0, 0, 0, np.radians(initial_pitch_deg), 0])

    loop = ClosedLoop(initial_state, physics_dt=0.01, use_ukf=use_ukf)  # the "hardware"
    pacer = RealTimePacer(
        loop.physics_dt
    )  # keeps simulated time in step with the clock
    renderer = Renderer(CONFIG.segway)  # 3D view + the app window
    panel = ControlPanel(loop, pacer)  # readout and buttons

    def update() -> None:
        """Called by the renderer at the start of every frame."""
        for _ in range(pacer.steps_due()):
            loop.step()

        x, y = loop.position
        renderer.update_pose(
            x, y, yaw=loop.state[YAW], pitch=loop.state[PITCH], distance=loop.state[S]
        )
        renderer.follow(x, y)

    renderer.run(update, draw_ui=panel.draw)  # blocks until the window is closed

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
