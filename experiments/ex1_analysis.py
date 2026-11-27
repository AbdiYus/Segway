"""EX1 scoring and tornado chart, from the trajectories saved by ex1_sensitivity.py.

Score: relative convergence time r = t_c / t_c_nominal, where t_c is the time
from the push until |pitch| stays inside +-band. For the UKF, r is computed per
noise seed (against the nominal run with the same seed) and the median over the
seeds is reported, with the interquartile range as its spread.

    python -m experiments.ex1_analysis [--controller NONLINEAR_MPC] [--band 1.0] [--T 0.02 --horizon 20]
"""

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

from .ex1_sensitivity import LEVEL, RESULTS_DIR
from .runner import LABELS, PARAMETERS, PUSH_TIME, Run, convergence_time, override_mpc, settings_tag

# Reference palette (dataviz skill), slots 1-2, and text inks
COLOR_MINUS, COLOR_PLUS = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(controller: str):
    data = np.load(RESULTS_DIR / f"ex1_{controller.lower()}_{settings_tag()}.npz")
    time = data["time"]
    runs = {}
    for key in data.files:
        parts = key.split("|")
        if len(parts) != 3:
            continue
        config, feedback, seed = parts
        runs[(config, feedback, int(seed))] = Run(
            {}, int(seed), feedback == "true", time, data[key],
            bool(data[key + "|fallen"]), int(data[key + "|fallbacks"]),
        )
    return runs


def score(runs, band_deg: float):
    """{(config, feedback): (median r, q25, q75, n_converged, n_runs)}."""
    band = np.radians(band_deg)
    tc = {k: convergence_time(r, band) for k, r in runs.items()}
    out = {}
    for config, feedback in {(c, f) for c, f, _ in runs}:
        seeds = sorted(s for c, f, s in runs if (c, f) == (config, feedback))
        base = np.array([tc[("nominal", feedback, s)] for s in seeds])
        num = np.array([tc[(config, feedback, s)] for s in seeds])
        with np.errstate(divide="ignore", invalid="ignore"):  # baseline t_c = 0 -> undefined
            r = np.where(base > 0, num / np.where(base > 0, base, 1), np.nan)
        ok = r[np.isfinite(r)]
        q = np.percentile(ok, [50, 25, 75]) if ok.size else [np.nan] * 3
        out[(config, feedback)] = (*q, ok.size, r.size)
    return out, tc


def tornado(scores, controller: str, band_deg: float, only: str | None = None) -> None:
    present = {fb for _, fb in scores} if only is None else {only}
    feedbacks = [(fb, t) for fb, t in (("true", "True state fed back"), ("ukf", "UKF in the loop"))
                 if fb in present]
    keys = {p: (f"{p}{-LEVEL:+.0%}", f"{p}{+LEVEL:+.0%}") for p in PARAMETERS}

    def effect(p):  # sort by the largest change in either feedback case
        return max(abs(scores[(k, fb)][0] - 1) for k in keys[p] for fb, _ in feedbacks
                   if np.isfinite(scores[(k, fb)][0]))

    order = sorted(PARAMETERS, key=effect)  # largest at the top
    y = np.arange(len(order))
    h = 0.36

    fig, axes = plt.subplots(1, len(feedbacks), figsize=(4.8 * len(feedbacks), 3.8),
                             sharey=True, facecolor=SURFACE, squeeze=False)
    axes = axes[0]
    lim = 1.15 * max(abs(scores[(k, fb)][i] - 1) * 100
                     for p in order for k in keys[p] for fb, _ in feedbacks for i in (0, 1, 2)
                     if np.isfinite(scores[(k, fb)][i]))
    for ax, (fb, title) in zip(axes, feedbacks):
        ax.set_facecolor(SURFACE)
        for offset, idx, color, name in ((+h / 2 + 0.01, 0, COLOR_MINUS, f"{-LEVEL:+.0%}"),
                                         (-h / 2 - 0.01, 1, COLOR_PLUS, f"{+LEVEL:+.0%}")):
            med = np.array([scores[(keys[p][idx], fb)][0] for p in order])
            lo = np.array([scores[(keys[p][idx], fb)][1] for p in order])
            hi = np.array([scores[(keys[p][idx], fb)][2] for p in order])
            val = (med - 1) * 100
            ax.barh(y + offset, val, height=h, color=color, label=f"parameter {name}", zorder=2)
            if fb == "ukf":  # spread over the noise seeds
                ax.errorbar(val, y + offset, xerr=[(med - lo) * 100, (hi - med) * 100],
                            fmt="none", ecolor=INK_2, elinewidth=1, capsize=2, zorder=3)
            for yi, v, m in zip(y + offset, val, med):
                if not np.isfinite(m):
                    ax.text(0, yi, " did not converge", va="center", fontsize=7, color=INK_2)
                elif abs(v) >= 0.05:
                    pad = 0.012 * lim * (1 if v > 0 else -1)
                    ax.text(v + pad, yi, f"{v:+.1f}", va="center", ha="left" if v > 0 else "right",
                            fontsize=7, color=INK_2)
        for yi, p in zip(y, order):
            if all(abs(scores[(k, fb)][0] - 1) < 5e-4 for k in keys[p]):
                ax.text(0.012 * lim, yi, "no effect", va="center", fontsize=7, color=INK_2)
        ax.axvline(0, color=INK, lw=1, zorder=3)
        ax.set_xlim(-lim, lim)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("change in convergence time [%]", fontsize=9, color=INK_2)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_yticks(y, [LABELS[p] for p in order], fontsize=10, color=INK)
    axes[-1].legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS_DIR / f"ex1_tornado_{controller.lower()}_{settings_tag()}.{ext}", dpi=200, facecolor=SURFACE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", default="NONLINEAR_MPC")
    parser.add_argument("--band", type=float, default=1.0, help="convergence band [deg]")
    parser.add_argument("--feedback", choices=["true", "ukf"], help="plot only this case (default: all run)")
    parser.add_argument("--T", type=float, help="sample time the runs used [s]")
    parser.add_argument("--horizon", type=int, help="horizon the runs used [steps]")
    args = parser.parse_args()
    override_mpc(args.T, args.horizon)  # only selects the result file

    runs = load(args.controller)
    scores, tc = score(runs, args.band)

    rows = []
    for config in ["nominal"] + [f"{p}{d:+.0%}" for p in PARAMETERS for d in (-LEVEL, LEVEL)]:
        for fb in sorted({f for _, f in scores}, key=["true", "ukf"].index):
            med, q25, q75, n_ok, n = scores[(config, fb)]
            seeds = sorted(s for c, f, s in runs if (c, f) == (config, fb))
            tcs = np.array([tc[(config, fb, s)] for s in seeds])
            rows.append([config, fb, f"{np.nanmedian(tcs):.2f}" if np.isfinite(tcs).any() else "nan",
                         f"{med:.3f}", f"{q25:.3f}", f"{q75:.3f}", f"{n_ok}/{n}",
                         sum(runs[(config, fb, s)].fallen for s in seeds),
                         sum(runs[(config, fb, s)].fallbacks for s in seeds)])
    header = ["config", "feedback", "t_c median [s]", "r median", "r q25", "r q75",
              "converged", "fallen", "nmpc fallbacks"]
    with open(RESULTS_DIR / f"ex1_scores_{args.controller.lower()}_{settings_tag()}.csv", "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    widths = [22, 9, 15, 9, 7, 7, 10, 7, 15]
    for row in [header] + rows:
        print("".join(str(v).ljust(w) for v, w in zip(row, widths)))

    tornado(scores, args.controller, args.band, args.feedback)
    print(f"band = +-{args.band} deg, push at {PUSH_TIME} s; figure and CSV in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
