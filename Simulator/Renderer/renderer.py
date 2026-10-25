from pathlib import Path
from typing import Callable

import numpy as np
import pyvista as pv
from imgui_bundle import hello_imgui, imgui, immapp

from Control_System.Models.segway_configuration import CONFIG, SegwayParams
from Simulator.Renderer.imgui_compat import CompatibleImguiPlotter

MODEL_DIRECTORY = Path(__file__).parent / "3d_models"


class Renderer:
    """
    3D renderer of the segway, shown inside an imgui application.

    It only draws: it is told where the segway is and shows it there. The
    imgui app calls back every frame, first to update the world, then to let
    others (the control panel) draw their own windows on top of the 3D view.

    Frames (x forward, y left, z up):
        world
         └─ base     (x, y, yaw)              where the segway stands on the floor
             ├─ body     (axle height, pitch)
             ├─ wheel_l  (+axle/2 in y, axle height, wheel spin)
             └─ wheel_r  (-axle/2 in y, axle height, wheel spin)
    """

    FLOOR_SIZE = 2.0  # [m]
    FLOOR_CELL_SIZE = 0.1  # [m]
    WINDOW_SIZE = (1400, 900)  # [px]

    def __init__(self, params: SegwayParams = CONFIG.segway) -> None:
        self.axle_length = params.axle_length
        self.wheel_radius = params.wheel_radius

        body_mesh, wheel_mesh = self.load_models()

        # Same API as pv.Plotter, but it renders into an imgui image instead of its own window
        self.plotter = CompatibleImguiPlotter()
        self.plotter.add_mesh(self.make_floor(), style="wireframe", color="gray")

        self.body_actor = self.plotter.add_mesh(body_mesh, color="steelblue")
        self.wheel_l_actor = self.plotter.add_mesh(wheel_mesh.copy(), color="dimgray")
        self.wheel_r_actor = self.plotter.add_mesh(wheel_mesh.copy(), color="dimgray")

        self.set_side_view()

    # ================================ Pose ================================
    def update_pose(
        self, x: float, y: float, yaw: float, pitch: float, distance: float
    ) -> None:
        """
        Place every part of the segway in the world.

        distance is s, the distance rolled by the axle centre. It is only used
        to spin the wheels.
        """
        world_from_base = self.translation(x, y, 0) @ self.rotate_yaw(yaw)
        axle_height = self.translation(0, 0, self.wheel_radius)

        # In a turn the outer wheel rolls further than the inner one
        half_axle = self.axle_length / 2
        spin_l = (distance - yaw * half_axle) / self.wheel_radius
        spin_r = (distance + yaw * half_axle) / self.wheel_radius

        body_local = axle_height @ self.rotate_pitch(pitch)
        wheel_l_local = self.translation(
            0, +half_axle, self.wheel_radius
        ) @ self.rotate_pitch(spin_l)
        wheel_r_local = self.translation(
            0, -half_axle, self.wheel_radius
        ) @ self.rotate_pitch(spin_r)

        self.body_actor.user_matrix = world_from_base @ body_local
        self.wheel_l_actor.user_matrix = world_from_base @ wheel_l_local
        self.wheel_r_actor.user_matrix = world_from_base @ wheel_r_local

    # ============================= Transforms =============================
    def rotate_yaw(self, angle: float) -> np.ndarray:
        """Homogeneous rotation about the z-axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array(
            [
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    def rotate_pitch(self, angle: float) -> np.ndarray:
        """Homogeneous rotation about the y-axis. Positive tips +z towards +x."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array(
            [
                [c, 0, s, 0],
                [0, 1, 0, 0],
                [-s, 0, c, 0],
                [0, 0, 0, 1],
            ]
        )

    def translation(self, x: float, y: float, z: float) -> np.ndarray:
        """Homogeneous translation."""
        return np.array(
            [
                [1, 0, 0, x],
                [0, 1, 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )

    # =============================== Camera ===============================
    def follow(self, x: float, y: float) -> None:
        """Slide the camera along with the segway, keeping the user's view angle."""
        camera = self.plotter.camera
        focal_x, focal_y, focal_z = camera.focal_point
        position_x, position_y, position_z = camera.position
        dx, dy = x - focal_x, y - focal_y

        camera.focal_point = (x, y, focal_z)
        camera.position = (position_x + dx, position_y + dy, position_z)

    def set_side_view(self) -> None:
        """Look at the segway from its right side, so pitch is easy to see."""
        look_at = (0.0, 0.0, 0.12)
        self.plotter.camera_position = [(0.0, -1.0, 0.35), look_at, (0.0, 0.0, 1.0)]

    # ============================== Running ===============================
    def run(self, update: Callable[[], None], draw_ui: Callable[[], None]) -> None:
        """
        Open the window and run until it is closed.

        Every frame, in this order:
            update()   advance the world and move the segway (update_pose)
            3D view    drawn as the background, filling the whole window
            draw_ui()  extra imgui windows drawn on top (the control panel)
        """

        def draw_frame() -> None:
            update()
            self.draw_viewport()
            draw_ui()

        runner_params = hello_imgui.RunnerParams()
        runner_params.app_window_params.window_title = "Segway Sim"
        runner_params.app_window_params.window_geometry.size = self.WINDOW_SIZE
        runner_params.callbacks.show_gui = draw_frame
        runner_params.callbacks.setup_imgui_style = lambda: hello_imgui.apply_theme(
            hello_imgui.ImGuiTheme_.imgui_colors_dark
        )
        # hello_imgui slows to a few FPS when the mouse is still. We animate, so always redraw.
        runner_params.fps_idling.enable_idling = False
        # The default full-window dock blocks mouse input to the 3D view (see pyvista-imgui example)
        runner_params.imgui_window_params.default_imgui_window_type = (
            hello_imgui.DefaultImGuiWindowType.no_default_window
        )
        # VTK must free its OpenGL objects while imgui's OpenGL context still exists,
        # otherwise Python crashes (segfault) on exit
        runner_params.callbacks.before_exit = self.plotter.render_window.Finalize

        immapp.run(runner_params=runner_params)  # blocks until the window is closed

    def draw_viewport(self) -> None:
        """The 3D scene as a borderless imgui window covering the whole app window."""
        viewport = imgui.get_main_viewport()
        imgui.set_next_window_pos(viewport.pos)
        imgui.set_next_window_size(viewport.size)

        flags = (
            imgui.WindowFlags_.no_decoration
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_bring_to_front_on_focus  # keeps the panel on top
        )
        imgui.begin("3D view", flags=flags)
        self.plotter.render_imgui()  # renders VTK into a texture and shows it as an image
        imgui.end()

    # ============================== Utilities =============================
    def make_floor(self) -> pv.PolyData:
        cells = int(self.FLOOR_SIZE / self.FLOOR_CELL_SIZE)
        return pv.Plane(
            center=(0, 0, 0),
            direction=(0, 0, 1),
            i_size=self.FLOOR_SIZE,
            j_size=self.FLOOR_SIZE,
            i_resolution=cells,
            j_resolution=cells,
        )

    def load_models(self) -> tuple[pv.PolyData, pv.PolyData]:
        body_path = MODEL_DIRECTORY / "body.obj"
        wheel_path = MODEL_DIRECTORY / "wheel.obj"

        body = pv.get_reader(str(body_path)).read()
        wheel = pv.get_reader(str(wheel_path)).read()

        if body.n_points == 0:
            raise ValueError(f"{body_path} is empty")
        if wheel.n_points == 0:
            raise ValueError(f"{wheel_path} is empty")

        return body, wheel
