import numpy as np
import matplotlib.pyplot as plt
from acados_template import latexify_plot
from Control_System.Models.segway_sensors import measure
from UKF import SegwayUKF
from Control_System.MPC import segway_integrator, segway_ocp, segway_solver
from Control_System.Models.segway_configuration import CONFIG


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


def sim(T_total=8.0, use_ukt=True, seed=0):
    x0 = np.array([0, 0, 0, 0, np.radians(20), 0])
    rng = np.random.default_rng(seed)

    ocp = segway_ocp()
    solver = segway_solver(ocp)
    integrator = segway_integrator(ocp)

    nx = ocp.dims.nx
    nu = ocp.dims.nu

    Ts = CONFIG.mpc.T  # one MPC step = one integrator step (tf / N)
    N_sim = int(round(T_total / Ts))

    simX = np.zeros((N_sim + 1, nx))
    simU = np.zeros((N_sim, nu))
    simX[0, :] = x0
    estX = np.zeros((N_sim, nx))
    nis = np.zeros(N_sim)  # Normalised Innovation Squared
    simX[0, :] = x0

    ukf = SegwayUKF(dt=Ts)  # starts at x_hat = 0: it does NOT know about the 20 deg
    u_prev = np.zeros(nu)

    for i in range(N_sim):
        # 1. sensors read the true state 2. UKF corrects
        z = measure(simX[i, :], u_prev, rng)

        # 2. UKF correction
        ukf.update(z, u_prev)
        estX[i, :] = ukf.x
        nis[i] = ukf.nis

        # 3. MPC solves from the estimate (or the true state)
        x_mpc = ukf.x if use_ukf else simX[i, :]

        solver.set(0, "lbx", x_mpc)
        solver.set(0, "ubx", x_mpc)

        # solver.set(0, "lbx", simX[i, :])
        # solver.set(0, "ubx", simX[i, :])
        status = solver.solve()
        simU[i, :] = solver.get(0, "u")
        if status != 0:
            print(f"step {i}: acados status {status}")

        # 4. plant
        simX[i + 1, :] = integrator.simulate(x=simX[i, :], u=simU[i, :])

        # 5. UKF predicts with the torque that was applied
        ukf.predict(simU[i, :])
        u_prev = simU[i, :]

    err = estX - simX[:-1]
    print("UKF RMS error per state [s, ds, theta, dtheta, psi, dpsi]:")
    print("  ", np.array2string(np.sqrt(np.mean(err**2, axis=0)), precision=4))
    print(f"Mean NIS: {nis.mean():.2f} (about 6 if consistent)")

    t = np.linspace(0, N_sim * Ts, N_sim + 1)
    plot_segway(
        t,
        u_max=CONFIG.mpc.torque_constraint,
        U=simU,
        X_true=simX,
        x_labels=ocp.model.x_labels,
        u_labels=ocp.model.u_labels,
        time_label=ocp.model.t_label,
        X_est=estX,
    )


if __name__ == "__main__":
    sim()
