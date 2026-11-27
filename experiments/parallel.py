"""Run many scenarios in parallel worker processes.

Every worker builds its own controllers, UKF and true plant once, in its own
build folder (experiments/build/workers/<n>), and then runs scenarios from a
shared queue. Results come back as plain arrays, in the order the jobs were
given, so the outcome is the same as running them one by one.

    jobs = [Job("nominal|ukf|0", {}, seed=0, true_state=False), ...]
    results = run_parallel(jobs, "NONLINEAR_MPC", T=0.02, horizon=20)
"""

import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

WORKERS_DIR = Path(__file__).parent / "build" / "workers"


@dataclass
class Job:
    key: str
    errors: dict
    seed: int
    true_state: bool


@dataclass
class Result:
    key: str
    pitch: np.ndarray
    fallen: bool
    fallbacks: int


_segway = None  # one per worker process


def _init_worker(counter, controller: str, T, horizon) -> None:
    global _segway
    import contextlib
    import io

    with counter.get_lock():
        index = counter.value
        counter.value += 1
    with contextlib.redirect_stdout(io.StringIO()):  # the model prints its derivation
        from src.controllers import ControllerType

        from .runner import ExperimentSegway, override_mpc, use_build_dir

        override_mpc(T, horizon)
        use_build_dir(WORKERS_DIR / str(index))
        _segway = ExperimentSegway(ControllerType[controller])


def _run(job: Job) -> Result:
    from .runner import run_scenario

    run = run_scenario(_segway, job.errors, job.seed, job.true_state)
    return Result(job.key, run.pitch, run.fallen, run.fallbacks)


def default_workers() -> int:
    """Leave two cores free for the rest of the machine."""
    return max(1, (os.cpu_count() or 2) - 2)


def run_parallel(
    jobs: list[Job],
    controller: str,
    T: float | None = None,
    horizon: int | None = None,
    workers: int | None = None,
) -> list[Result]:
    workers = min(workers or default_workers(), len(jobs))
    shutil.rmtree(WORKERS_DIR, ignore_errors=True)  # stale code from a previous sweep
    ctx = get_context("spawn")
    counter = ctx.Value("i", 0)

    results: dict[str, Result] = {}
    t0 = time.time()
    with ProcessPoolExecutor(
        workers, mp_context=ctx, initializer=_init_worker, initargs=(counter, controller, T, horizon)
    ) as pool:
        futures = [pool.submit(_run, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results[result.key] = result
            print(f"{done:4d}/{len(jobs)}  {result.key:30s} ({time.time() - t0:.0f} s)", flush=True)
    return [results[job.key] for job in jobs]
