"""EX1: each parameter alone at +-10 %, the others nominal.

Runs the nominal configuration and the 12 one-at-a-time configurations, with
the true state fed back (deterministic, one run each) and with the UKF (one
run per noise seed). Every pitch trajectory is saved, so the scoring
(ex1_analysis.py) can be redone with another convergence band without
rerunning the simulations.

    python -m experiments.ex1_sensitivity [--controller NONLINEAR_MPC] [--seeds 10]
"""

import argparse
import time
from pathlib import Path

import numpy as np

from src.controllers import ControllerType

from .runner import PARAMETERS, ExperimentSegway, run_scenario

RESULTS_DIR = Path(__file__).parent / "results"
LEVEL = 0.10  # relative error


def configurations() -> list[dict[str, float]]:
    configs = [{}]  # nominal first: the baseline
    for name in PARAMETERS:
        configs += [{name: -LEVEL}, {name: +LEVEL}]
    return configs


def config_key(errors: dict[str, float]) -> str:
    if not errors:
        return "nominal"
    (name, d), = errors.items()
    return f"{name}{d:+.0%}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", default="NONLINEAR_MPC", choices=[c.name for c in ControllerType])
    parser.add_argument("--seeds", type=int, default=10, help="noise seeds for the UKF runs (0 = true state only)")
    args = parser.parse_args()

    segway = ExperimentSegway(ControllerType[args.controller])
    out = RESULTS_DIR / f"ex1_{args.controller.lower()}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    data, t0 = {}, time.time()
    for errors in configurations():
        key = config_key(errors)
        for feedback, seeds in (("true", [0]), ("ukf", range(args.seeds))):
            for seed in seeds:
                run = run_scenario(segway, errors, seed, true_state_feedback=feedback == "true")
                data[f"{key}|{feedback}|{seed}"] = run.pitch
                data[f"{key}|{feedback}|{seed}|fallen"] = run.fallen
                data[f"{key}|{feedback}|{seed}|fallbacks"] = run.fallbacks
                data["time"] = run.time
        print(f"{key:>20s} done  ({time.time() - t0:.0f} s)", flush=True)

    np.savez_compressed(out, **data)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
