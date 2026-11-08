"""
The running application: the World shown live in the 3D window.

Role in the system
------------------
Built by main.py, which adds the segways and calls run(). Owns everything
that happens per frame: advancing the world in real time, moving the 3D
model, following it with the camera, and drawing the control panel.
This is the only module that knows about both the simulation and the window.

For now the renderer draws one segway: the one selected in the control panel.
"""

from ..models import CONFIG
from ..simulator import Segway, RealTimePacer, World, History
from ..renderer import Renderer, ControlPanel


class Simulator:
    """World + window, advanced in real time every frame."""

    def __init__(
        self, physics_dt: float, history: History, time_scale: float = 1.0
    ) -> None:
        self.world = World(physics_dt)
        self.pacer = RealTimePacer(physics_dt, time_scale)
        self.renderer = Renderer(CONFIG.segway)
        self.panel = ControlPanel(self.world, self.pacer, history)

    # ============================== Setup ================================
    def add(self, segway: Segway) -> None:
        self.world.add(segway)

    # ============================== Running ==============================
    def run(self) -> None:
        """Open the window and run until it is closed."""
        if not self.world.segways:
            raise ValueError("add at least one segway before run()")
        self.renderer.run(self.update, draw_ui=self.panel.draw)  # blocks until closed

    def update(self) -> None:
        """Called by the renderer at the start of every frame."""
        for _ in range(self.pacer.steps_due()):
            self.world.step()

        pose = self.panel.segway.pose
        self.renderer.update_pose(pose.x, pose.y, pose.yaw, pose.pitch, pose.distance)
        self.renderer.follow(pose.x, pose.y)
