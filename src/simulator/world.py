"""
All segways sharing one clock.

Role in the system
------------------
Built in main.py, which adds the segways and calls step() until the
simulation is done (later: as often as real time demands, from the window).
Owns the segways and the simulation time. Anything that involves more than
one segway (collisions, communication, formations) belongs here, never
inside a Segway. Knows nothing about rendering, so it can be stepped in a
plain for-loop.
"""

from .segway import Segway


class World:
    """The segways and the clock they share."""

    def __init__(self, physics_dt: float) -> None:
        self.physics_dt = physics_dt
        self.time = 0.0
        self.step_count = 0
        self.segways: list[Segway] = []

    # ============================== Setup ================================
    def add(self, segway: Segway) -> None:
        if segway.physics_dt != self.physics_dt:
            raise ValueError("a segway must use the same physics_dt as the world")
        self.segways.append(segway)

    # ============================== Running ==============================
    def step(self) -> None:
        """Advance every segway by one physics step."""
        for segway in self.segways:
            segway.step(self.time)
        self.step_count += 1
        self.time = self.step_count * self.physics_dt  # no drift from adding floats
