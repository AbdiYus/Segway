import os
import sys
from typing import Any
import pyvista as pv


class Renderer:
    """
    3D renderer of the segway.

    It has the goal of showcasing how the control sytem is applied onto the
    segway.
    """

    def __init__(self) -> None:
        body, wheel = self.load_models()

        self.body = body
        self.wheel = wheel

    def load_models(self) -> tuple[pv.ImageData, Any]:
        PATH = os.path.dirname(os.path.abspath(sys.argv[0]))

        body_reader = pv.get_reader(f"{PATH}/3d_models/body.obj")
        wheel_reader = pv.get_reader(f"{PATH}/3d_models/wheel.obj")

        body = body_reader.read()
        wheel = wheel_reader.read()

        if body.n_points == 0:
            raise ValueError(f"{PATH}/3d_models/body.obj is empty")
        if body.n_points == 0:
            raise ValueError(f"{PATH}/3d_models/wheel.obj is empty")

        return body, wheel


# plotter = pv.Plotter(title="Segway Sim")
#
# floor = pv.Plane(
#     center=(0, 0, 0),
#     direction=(0, 0, 1),
#     i_size=15,
#     j_size=15,
#     i_resolution=15,
#     j_resolution=15,
# )
#
# plotter.add_mesh(floor, style="wireframe", color="gray")
#
# plotter.add_mesh(body, color="steel_blue")
# plotter.add_mesh(wheel, color="gray")
# plotter.add_mesh(wheel, color="gray")
#
# plotter.show()
