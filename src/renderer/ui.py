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
        history = self.history.as_arrays()
        time = history[HistoryIndex.TIME]
        true_pitch = history[HistoryIndex.TRUE_PITCH]
        estimated_pitch = history[HistoryIndex.ESTIMATED_PITCH]
        torque = history[HistoryIndex.TORQUE]
        torque_limit = CONFIG.mpc.torque_constraint

        if len(time) == 0:
            print("ERR: there does not exists any history")
            return

        implot.set_next_axis_to_fit(implot.ImAxis_.y1)

        if implot.begin_plot("Pitch", size=(500, 250)):
            implot.setup_axes("time [s]", "pitch [deg]")

            implot.setup_axis_limits(
                implot.ImAxis_.x1,
                time[-1] - 10,
                time[-1],
                imgui.Cond_.always,
            )

            implot.plot_line("true", time, np.array(true_pitch))
            implot.plot_line("estimated", time, np.array(estimated_pitch))
            implot.end_plot()

        imgui.separator()

        if implot.begin_plot("Torque", size=(500, 250)):
            implot.setup_axes("time [s]", "torque [Nm]")

            implot.setup_axis_limits(
                implot.ImAxis_.x1,
                time[-1] - 10,
                time[-1],
                imgui.Cond_.always,
            )

            implot.setup_axis_limits(
                implot.ImAxis_.y1,
                -1.1 * torque_limit,
                1.1 * torque_limit,
                imgui.Cond_.always,
            )

            implot.plot_line("left", time, np.ascontiguousarray(torque[:, 0]))
            implot.plot_line("right", time, np.ascontiguousarray(torque[:, 1]))

            # Red limit lines
            implot.plot_inf_lines(
                "limit",
                np.array([-torque_limit, torque_limit]),
                spec=implot.Spec(
                    line_color=imgui.ImVec4(1.0, 0.3, 0.3, 1.0),
                    line_weight=1.5,
                    flags=implot.InfLinesFlags_.horizontal,
                ),
            )
            implot.end_plot()

    # def draw_tuning(self) -> None:
    #     loop = self.segway.control_loop
    #     names = loop.get_names()
    #     current_controller_name = names.index(loop.active_controller.name)
    #     changed, index = imgui.combo("controller", current_controller_name, names)
    #
    #     if changed:
    #         loop.switch_controller_to()

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

    def draw_settings(self) -> None:
        _, self.pacer.time_scale = imgui.slider_float(
            "speed", self.pacer.time_scale, 0.0, 2.0
        )
        imgui.text("speed 0 = paused, 0.5 = slow motion")
