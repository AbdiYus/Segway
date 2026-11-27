"""EX1: each parameter alone at +-10 %, the others nominal.

Runs the nominal configuration and the 12 one-at-a-time configurations, with
the true state fed back (deterministic, one run each) and with the UKF (one
run per noise seed). Every pitch trajectory is saved, so the scoring
(ex1_analysis.py) can be redone with another convergence band without
rerunning the simulations.

    python -m experiments.ex1_sensitivity [--controller NONLINEAR_MPC] [--seeds 3] [--T 0.02 --horizon 20] [--workers 8]
"""

import argparse
import time
from pathlib import Path

import numpy as np

from src.controllers import ControllerType

from .parallel import Job, Result, default_workers, run_parallel
from .runner import (
    PARAMETERS,
    PHYSICS_DT,
    T_END,
    ExperimentSegway,
    override_mpc,
    run_scenario,
    settings_tag,
)

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
    parser.add_argument("--seeds", type=int, default=3, help="noise seeds for the UKF runs (0 = true state only)")
    parser.add_argument("--T", type=float, help="MPC/UKF sample time [s] (default: configuration.py)")
    parser.add_argument("--horizon", type=int, help="MPC horizon in steps (default: configuration.py)")
    parser.add_argument("--workers", type=int, default=default_workers(),
                        help="parallel processes (1 = run in this process)")
    args = parser.parse_args()

    override_mpc(args.T, args.horizon)
    out = RESULTS_DIR / f"ex1_{args.controller.lower()}_{settings_tag()}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    jobs = [
        Job(f"{config_key(errors)}|{feedback}|{seed}", errors, seed, feedback == "true")
        for errors in configurations()
        for feedback, seeds in (("true", [0]), ("ukf", range(args.seeds)))
        for seed in seeds
    ]
    t0 = time.time()
    if args.workers > 1:
        results = run_parallel(jobs, args.controller, args.T, args.horizon, args.workers)
    else:
        segway = ExperimentSegway(ControllerType[args.controller])
        results = []
        for i, job in enumerate(jobs, 1):
            run = run_scenario(segway, job.errors, job.seed, job.true_state)
            results.append(Result(job.key, run.pitch, run.fallen, run.fallbacks))
            print(f"{i:4d}/{len(jobs)}  {job.key:30s} ({time.time() - t0:.0f} s)", flush=True)

    data = {"time": (np.arange(round(T_END / PHYSICS_DT)) + 1) * PHYSICS_DT}
    for r in results:
        data[r.key] = r.pitch
        data[f"{r.key}|fallen"] = r.fallen
        data[f"{r.key}|fallbacks"] = r.fallbacks
    np.savez_compressed(out, **data)
    print(f"{len(jobs)} runs in {time.time() - t0:.0f} s")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
