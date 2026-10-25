"""
The imgui window on top of the 3D view: live readout and disturbance buttons.

It reads from and pokes at the ClosedLoop, nothing else. Immediate-mode GUI:
draw() is called every frame and rebuilds the whole window from the current
values, so there is no widget state to keep in sync.
"""

import numpy as np
from imgui_bundle import imgui

from Control_System.Models.segway_configuration import CONFIG
from Simulator.control_loop import (
    PITCH,
    PITCH_RATE,
    S,
    S_DOT,
    ClosedLoop,
    RealTimePacer,
)


class ControlPanel:
    def __init__(self, loop: ClosedLoop, pacer: RealTimePacer) -> None:
        self.loop = loop
        self.pacer = pacer
        self.push_strength = 1.0  # [rad/s] instant change of pitch rate per push
        self.tilt_angle_deg = 10.0

    def draw(self) -> None:
        imgui.set_next_window_pos((20, 20), imgui.Cond_.first_use_ever)
        imgui.begin("Segway", flags=imgui.WindowFlags_.always_auto_resize)

        self.draw_state()
        imgui.separator()
        self.draw_disturbances()
        imgui.separator()
        self.draw_settings()

        imgui.end()

    # ============================== Sections ==============================
    def draw_state(self) -> None:
        loop = self.loop
        true_state, estimate = loop.state, loop.ukf.x
        tau_l, tau_r = loop.torque

        status = loop.solver_status
        rows = [
            ("time", f"{loop.time:.2f} s"),
            ("pitch (true)", f"{np.degrees(true_state[PITCH]):+.1f} deg"),
            ("pitch (UKF)", f"{np.degrees(estimate[PITCH]):+.1f} deg"),
            ("pitch rate", f"{np.degrees(true_state[PITCH_RATE]):+.1f} deg/s"),
            ("distance s", f"{true_state[S]:+.3f} m"),
            ("speed", f"{true_state[S_DOT]:+.3f} m/s"),
            ("torque left", f"{tau_l:+.3f} Nm"),
            ("torque right", f"{tau_r:+.3f} Nm"),
            ("torque limit", f"{CONFIG.mpc.torque_constraint:.3f} Nm"),
            ("acados status", f"{status}" if status == 0 else f"{status} (failed)"),
        ]

        if imgui.begin_table("state", 2):
            for label, value in rows:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(label)
                imgui.table_next_column()
                imgui.text(value)
            imgui.end_table()

    def draw_disturbances(self) -> None:
        # Seen from the default camera, +x (forward) is to the right on screen
        if imgui.button("<- Push from right"):
            self.loop.push(pitch_rate_change=-self.push_strength)
        imgui.same_line()
        if imgui.button("Push from left ->"):
            self.loop.push(pitch_rate_change=+self.push_strength)

        _, self.push_strength = imgui.slider_float(
            "push [rad/s]", self.push_strength, 0.1, 5.0
        )

        if imgui.button("Tilt"):
            self.loop.tilt_to(np.radians(self.tilt_angle_deg))
        imgui.same_line()
        _, self.tilt_angle_deg = imgui.slider_float(
            "tilt [deg]", self.tilt_angle_deg, -30.0, 30.0
        )

    def draw_settings(self) -> None:
        changed, motors_enabled = imgui.checkbox("Motors on", self.loop.motors_enabled)
        if changed:
            self.loop.set_motors_enabled(motors_enabled)

        _, self.pacer.time_scale = imgui.slider_float(
            "speed", self.pacer.time_scale, 0.0, 2.0
        )
        imgui.text("speed 0 = paused, 0.5 = slow motion")
