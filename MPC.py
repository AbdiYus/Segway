import numpy as np
import matplotlib.pyplot as plt
from sympy import pprint
from scipy.signal import cont2discrete, lsim

A = np.array(
    [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, -981 / 200, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 2943 / 800, 0],
    ]
)

B = np.array(
    [
        [0, 0],
        [3 / 8, 3 / 8],
        [0, 0],
        [-4 / 11, 4 / 11],
        [0, 0],
        [-11 / 96, -11 / 96],
    ]
)

# These are wrong. They get retrived from kalman
C = np.eye(6)
D = np.zeros((6, 2))

Ts = 0.02
Ad, Bd, *_ = cont2discrete((A, B, C, D), Ts, method="zoh")

np.set_printoptions(precision=4, suppress=True)
print("Ad:")
print(Ad)
print("Bd:")
print(Bd)

# # continuous reference on a fine grid
# T = np.linspace(0, 2, 1000)
# u_const = np.array([0.1, 0.1])
# U = np.tile(u_const, (len(T), 1))
# t, _, x_c = lsim((A, B, C, D), U=U, T=T)
#
# # discrete simulation: x_{k+1} = Ad x_k + Bd u_k
# K = int(2 / Ts)
# x_d = np.zeros((K + 1, 6))
# for k in range(K):
#     x_d[k + 1] = Ad @ x_d[k] + Bd @ u_const
# t_d = np.arange(K + 1) * Ts
#
# labels = ["s", "s_dot", "theta", "theta_dot", "psi", "psi_dot"]
# fig, axes = plt.subplots(3, 2, sharex=True, figsize=(10, 7), constrained_layout=True)
# for i, ax in enumerate(axes.T.flat):
#     ax.plot(t, x_c[:, i], label="continuous")
#     ax.plot(t_d, x_d[:, i], "o", markersize=3, label="discrete (ZOH)")
#     ax.set_title(labels[i])
#     ax.grid(True)
#     if np.max(np.abs(x_d[:, i])) < 1e-9:
#         ax.set_ylim(-1, 1)
#
# axes[0, 0].legend()
# for ax in axes[-1, :]:
#     ax.set_xlabel("time [s]")
# plt.show()
#
# # numeric check: max error at the sample points
# x_c_at_samples = np.array([np.interp(t_d, t, x_c[:, i]) for i in range(6)]).T
# print("max abs error:", np.max(np.abs(x_c_at_samples - x_d)))
