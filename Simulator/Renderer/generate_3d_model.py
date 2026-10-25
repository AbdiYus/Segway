"""
Generate simple segway OBJ files (numpy only).

Frame convention (matches the transforms in the renderer):
    x = forward, y = left, z = up
    body.obj  : origin at axle centre, beam points up +z (pitch rotates about y)
    wheel.obj : origin at hub, rolling axis along y (spin rotates about y)

Sizes default to CONFIG.segway values; change and re-run to regenerate.
"""

from os import path
import os
import sys
import numpy as np
from Control_System.Models.segway_configuration import CONFIG

AXLE_LENGTH = CONFIG.segway.axle_length
BEAM_LENGTH = CONFIG.segway.beam_lenght
WHEEL_RADIUS = CONFIG.segway.wheel_radius

WHEEL_WIDTH = 0.25 * WHEEL_RADIUS
BEAM_THICK = 0.15 * BEAM_LENGTH
AXLE_THICK = 0.15 * WHEEL_RADIUS


class Mesh:
    def __init__(self):
        self.v, self.f = [], []

    def add(self, verts, faces):
        off = len(self.v)
        self.v += [tuple(p) for p in verts]
        self.f += [[i + off for i in face] for face in faces]

    def box(self, center, size):
        c, h = np.array(center), np.array(size) / 2
        verts = [
            c + h * np.array(s)
            for s in [
                (-1, -1, -1),
                (1, -1, -1),
                (1, 1, -1),
                (-1, 1, -1),
                (-1, -1, 1),
                (1, -1, 1),
                (1, 1, 1),
                (-1, 1, 1),
            ]
        ]
        faces = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ]
        self.add(verts, faces)

    def cylinder_y(self, radius, width, n=32, y0=0.0):
        """Cylinder with axis along y, centred at y0."""
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ring = lambda y: [(radius * np.cos(t), y, radius * np.sin(t)) for t in a]
        verts = ring(y0 - width / 2) + ring(y0 + width / 2)
        verts += [(0, y0 - width / 2, 0), (0, y0 + width / 2, 0)]  # cap centres
        c0, c1 = 2 * n, 2 * n + 1
        faces = []
        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, n + j, n + i])  # side
            faces.append([c0, j, i])  # cap at -y
            faces.append([c1, n + i, n + j])  # cap at +y
        self.add(verts, faces)

    def write(self, path, name):
        with open(path, "w") as fh:
            fh.write(f"# simple segway part: {name}\no {name}\n")
            for p in self.v:
                fh.write("v {:.6f} {:.6f} {:.6f}\n".format(*p))
            for face in self.f:
                fh.write("f " + " ".join(str(i + 1) for i in face) + "\n")


def make_body():
    m = Mesh()
    # axle between the wheels
    m.box((0, 0, 0), (AXLE_THICK, AXLE_LENGTH, AXLE_THICK))
    # beam (stem) from axle upwards
    m.box((0, 0, BEAM_LENGTH / 2), (BEAM_THICK, BEAM_THICK, BEAM_LENGTH))
    # handlebar on top
    m.box((0, 0, BEAM_LENGTH), (BEAM_THICK, 0.6 * AXLE_LENGTH, BEAM_THICK * 0.6))
    return m


def make_wheel():
    m = Mesh()
    m.cylinder_y(WHEEL_RADIUS, WHEEL_WIDTH)
    # spoke sticking out of both faces so the spin is visible
    w = WHEEL_WIDTH * 1.3
    m.box((0, 0, WHEEL_RADIUS * 0.45), (WHEEL_RADIUS * 0.15, w, WHEEL_RADIUS * 0.9))
    return m


if __name__ == "__main__":
    PATH = os.path.dirname(os.path.abspath(sys.argv[0]))
    make_body().write(f"{PATH}/3d_models/body.obj", "body")
    make_wheel().write(f"{PATH}/3d_models/wheel.obj", "wheel")
    print("wrote body.obj, wheel.obj")
