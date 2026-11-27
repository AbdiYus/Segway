"""
The imgui window on top of the 3D view: live readout and disturbance buttons.

It reads from and pokes at one segway in the World, nothing else. Immediate-mode GUI:
draw() is called every frame and rebuilds the whole window from the current
values, so there is no widget state to keep in sync.
"""

import numpy as np
from ..models import CONFIG
from imgui_bundle import imgui, implot
from ..simulator import RealTimePacer, Segway, StateIndex, World, History, HistoryIndex


class ControlPanel:
    def __init__(self, world: World, pacer: RealTimePacer, history: History) -> None:
        self.world = world
        self.pacer = pacer
        self.history = history
        self.selected = 0  # index of the segway shown and disturbed by the panel
        self.push_strength = 1.0  # [rad/s] instant change of pitch rate per push
        self.tilt_angle_deg = 10.0

    def draw(self) -> None:
        imgui.set_next_window_pos((20, 20), imgui.Cond_.first_use_ever)
        imgui.begin("Segway", flags=imgui.WindowFlags_.always_auto_resize)

        if imgui.begin_tab_bar("tabs"):
            if imgui.begin_tab_item("State")[0]:
                self.draw_state()
                imgui.separator()
                self.draw_disturbances()
                imgui.separator()
                self.draw_settings()
                imgui.end_tab_item()

            if imgui.begin_tab_item("Plots")[0]:
                self.draw_plots()
                imgui.end_tab_item()

            if imgui.begin_tab_item("Set Goal")[0]:
                self.draw_set_goal()
                imgui.end_tab_item()

            # if imgui.begin_tab_item("Tuning")[0]:
            #     self.draw_tuning()
            #     imgui.end_tab_item()

            imgui.end_tab_bar()

        imgui.end()

    @property
    def segway(self) -> Segway:
        return self.world.segways[self.selected]

    # ============================== Sections ==============================
    def draw_state(self) -> None:
        segway = self.segway
        loop = segway.control_loop
        true_state, estimate = segway.state, loop.observer.states
        tau_l, tau_r = segway.torque

        rows = [
            ("time", f"{self.world.time:.2f} s"),
            ("status", "FALLEN, press Reset" if segway.fallen else "balancing"),
            ("controller", loop.active_controller.name),
            ("observer", loop.active_observer.name),
            ("pitch (true)", f"{np.degrees(true_state[StateIndex.PITCH]):+.1f} deg"),
            ("pitch (estimate)", f"{np.degrees(estimate[StateIndex.PITCH]):+.1f} deg"),
            (
                "pitch rate",
                f"{np.degrees(true_state[StateIndex.PITCH_RATE]):+.1f} deg/s",
            ),
            ("distance s", f"{true_state[StateIndex.S]:+.3f} m"),
            ("speed", f"{true_state[StateIndex.S_DOT]:+.3f} m/s"),
            ("torque left", f"{tau_l:+.3f} Nm"),
            ("torque right", f"{tau_r:+.3f} Nm"),
            ("torque limit", f"{CONFIG.mpc.torque_constraint:.3f} Nm"),
        ]

        if imgui.begin_table("state", 2):
            for label, value in rows:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(label)
                imgui.table_next_column()
                imgui.text(value)
            imgui.end_table()

    def draw_plots(self) -> None:
        """Every constrained quantity over the last seconds, with its limit."""
        history = self.history.as_arrays()
        time = history[HistoryIndex.TIME]
        true_state = history[HistoryIndex.TRUE_STATE]
        estimated_state = history[HistoryIndex.ESTIMATED_STATE]
        torque = history[HistoryIndex.TORQUE]
        if len(time) == 0:
            imgui.text("no data yet")
            return

        mpc = CONFIG.mpc
        deg = np.degrees

        def true_and_estimated(index: StateIndex, scale=lambda v: v) -> list:
            return [
                ("true", scale(true_state[:, index])),
                ("estimated", scale(estimated_state[:, index])),
            ]

        plot_with_limit(
            "Pitch",
            "pitch [deg]",
            time,
            true_and_estimated(StateIndex.PITCH, deg),
            deg(mpc.pitch_constraint),
        )
        plot_with_limit(
            "Pitch rate",
            "pitch rate [deg/s]",
            time,
            true_and_estimated(StateIndex.PITCH_RATE, deg),
            deg(mpc.pitch_rate_constraint),
        )
        plot_with_limit(
            "Speed",
            "speed [m/s]",
            time,
            true_and_estimated(StateIndex.S_DOT),
            mpc.displacement_rate_constraint,
        )
        plot_with_limit(
            "Yaw rate",
            "yaw rate [deg/s]",
            time,
            true_and_estimated(StateIndex.YAW_RATE, deg),
            deg(mpc.yaw_rate_constraint),
        )
        plot_with_limit(
            "Torque",
            "torque [Nm]",
            time,
            [("left", torque[:, 0]), ("right", torque[:, 1])],
            mpc.torque_constraint,
        )

    # def draw_tuning(self) -> None:
    #     loop = self.segway.control_loop
    #     names = loop.get_names()
    #     current_controller_name = names.index(loop.active_controller.name)
    #     changed, index = imgui.combo("controller", current_controller_name, names)
    #
    #     if changed:
    #         loop.switch_controller_to()
    def draw_set_goal(self) -> None:
        segway = self.segway
        x, y = segway.position
        theta = segway.state[StateIndex.YAW]
        goal_x, goal_y = segway.goal

        if implot.begin_plot("Map", size=(500, 500), flags=implot.Flags_.equal):
            implot.setup_axes("x [m]", "y [m]")
            implot.setup_axes(
                "x [m]",
                "y [m]",
                implot.AxisFlags_.lock,
                implot.AxisFlags_.lock,  # no pan/zoom
            )
            implot.setup_axes_limits(-5, 5, -5, 5, imgui.Cond_.always)

            implot.plot_scatter(
                "segway",
                np.array([x]),
                np.array([y]),
                spec=implot.Spec(marker=implot.Marker_.circle, marker_size=8),
            )
            implot.plot_line(
                "heading",
                np.array([x, x + 0.5 * np.cos(theta)]),
                np.array([y, y + 0.5 * np.sin(theta)]),
            )
            implot.plot_scatter(
                "goal",
                np.array([goal_x]),
                np.array([goal_y]),
                spec=implot.Spec(marker=implot.Marker_.cross, marker_size=10),
            )

            self.draw_controller_plan()

            if implot.is_plot_hovered() and imgui.is_mouse_double_clicked(0):
                p = implot.get_plot_mouse_pos()
                segway.set_goal(p.x, p.y)

            implot.end_plot()

        imgui.text(f"goal: ({goal_x:+.2f}, {goal_y:+.2f}) m")
        imgui.text("double-click on the map to set a new goal")

    def draw_controller_plan(self) -> None:
        controller = self.segway.control_loop.controller

        if hasattr(controller, "get_planned_path"):
            plan = controller.get_planned_path()
            if len(plan) > 0:
                plan_x = np.ascontiguousarray(plan[:, 0])
                plan_y = np.ascontiguousarray(plan[:, 1])
                implot.plot_line(
                    "MPC plan",
                    plan_x,
                    plan_y,
                    spec=implot.Spec(marker=implot.Marker_.circle, marker_size=3),
                )

        if hasattr(controller, "get_position"):
            believed_x, believed_y = controller.get_position()
            implot.plot_scatter(
                "believed",
                np.array([believed_x]),
                np.array([believed_y]),
                spec=implot.Spec(marker=implot.Marker_.square, marker_size=6),
            )

    def draw_disturbances(self) -> None:
        # Seen from the default camera, +x (forward) is to the right on screen
        if imgui.button("<- Push from right"):
            self.segway.push(pitch_rate_change=-self.push_strength)
        imgui.same_line()
        if imgui.button("Push from left ->"):
            self.segway.push(pitch_rate_change=+self.push_strength)

        _, self.push_strength = imgui.slider_float(
            "push [rad/s]", self.push_strength, 0.1, 5.0
        )

        if imgui.button("Tilt"):
            self.segway.tilt_to(np.radians(self.tilt_angle_deg))
        imgui.same_line()
        _, self.tilt_angle_deg = imgui.slider_float(
            "tilt [deg]", self.tilt_angle_deg, -30.0, 30.0
        )

        if imgui.button("Reset"):
            self.segway.reset()

    def draw_settings(self) -> None:
        _, self.pacer.time_scale = imgui.slider_float(
            "speed", self.pacer.time_scale, 0.0, 2.0
        )
        imgui.text("speed 0 = paused, 0.5 = slow motion")


# ================================ Plot helpers ===================================
PLOT_SIZE = (500, 180)  # [px] each time plot
TIME_WINDOW = 10.0  # [s] how much history the time plots show
LIMIT_COLOR = (1.0, 0.3, 0.3, 1.0)  # red limit lines


def symmetric_y_range(values: list[np.ndarray], limit: float) -> float:
    """Half-height of the y axis: the limit with a margin, or more if the data goes past it.

    Soft constraints can be exceeded, so the axis grows instead of hiding a violation.
    """
    largest = max((np.max(np.abs(v)) for v in values if len(v) > 0), default=0.0)
    return 1.1 * max(limit, largest)


def plot_with_limit(
    title: str,
    y_label: str,
    time: np.ndarray,
    series: list[tuple[str, np.ndarray]],
    limit: float,
) -> None:
    """One time plot of the last TIME_WINDOW seconds, with red lines at +-limit."""
    visible = time >= time[-1] - TIME_WINDOW
    half_height = symmetric_y_range([values[visible] for _, values in series], limit)

    if not implot.begin_plot(title, size=PLOT_SIZE):
        return
    implot.setup_axes("time [s]", y_label)
    implot.setup_axis_limits(
        implot.ImAxis_.x1, time[-1] - TIME_WINDOW, time[-1], imgui.Cond_.always
    )
    implot.setup_axis_limits(
        implot.ImAxis_.y1, -half_height, half_height, imgui.Cond_.always
    )

    for label, values in series:
        # implot needs contiguous 1D arrays; column slices of a 2D array are not
        implot.plot_line(label, time, np.ascontiguousarray(values))
    implot.plot_inf_lines(
        "limit",
        np.array([-limit, limit]),
        spec=implot.Spec(
            line_color=imgui.ImVec4(*LIMIT_COLOR),
            line_weight=1.5,
            flags=implot.InfLinesFlags_.horizontal,
        ),
    )
    implot.end_plot()
