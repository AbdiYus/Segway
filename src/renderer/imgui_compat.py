"""
Fixes so pyvista-imgui (last released April 2024) works with current
pyvista (>= 0.45) and imgui-bundle (>= 1.92).

Kept in its own file so the renderer reads cleanly, and so this file can be
deleted if pyvista-imgui is ever updated. Usage is identical to ImguiPlotter.
"""

from imgui_bundle import imgui
from pyvista_imgui import ImguiPlotter
from pyvista_imgui._backend_imgui_bundle import RendererBackendImguiBundle
from pyvista_imgui.imgui_render_window import register_backend

COMPATIBLE_BACKEND = "imgui_bundle_compatible"


@register_backend(COMPATIBLE_BACKEND)
class CompatibleImguiBundleBackend(RendererBackendImguiBundle):
    """
    Same as the original backend, only render() is rewritten.

    imgui-bundle 1.92 changed imgui.image() to take an ImTextureRef instead
    of a plain OpenGL texture id, and begin_child() to take child flags
    instead of a border bool.
    """

    def render(self, size=None) -> None:
        size = size or imgui.get_content_region_avail()
        width, height = size.x, size.y

        # 1. Let VTK draw the scene into its OpenGL texture at the current size
        self.render_window.size = (width, height)
        self.render_window.render()
        self.interactor.SetSize(int(width), int(height))
        self.interactor.ConfigureEvent()

        # 2. Show that texture as an imgui image (flipped: OpenGL's origin is bottom-left)
        imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
        no_scroll = (
            imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse
        )
        imgui.begin_child("##Viewport", size, imgui.ChildFlags_.none, no_scroll)
        imgui.image(
            self.texture_reference(),
            imgui.get_content_region_avail(),
            imgui.ImVec2(0, 1),
            imgui.ImVec2(1, 0),
        )

        # 3. Pass mouse and keyboard input on to VTK (orbit, zoom, ...)
        self.process_events()
        imgui.end_child()
        imgui.pop_style_var()

    def texture_reference(self):
        texture_id = self.render_window.texture_id
        if hasattr(imgui, "ImTextureRef"):  # imgui-bundle >= 1.92
            return imgui.ImTextureRef(texture_id)
        return texture_id


class CompatibleImguiPlotter(ImguiPlotter):
    """ImguiPlotter that uses the backend above and survives pyvista >= 0.45."""

    def __init__(self, **kwargs) -> None:
        super().__init__(imgui_backend=COMPATIBLE_BACKEND, **kwargs)

    # ImguiPlotter first creates its own render window (the one that draws into
    # an imgui texture) and then calls pyvista's BasePlotter.__init__. Newer
    # pyvista sets `self.ren_win = None` in there, throwing that window away.
    # This property ignores that one reset during construction; closing the
    # plotter later still clears it as normal.
    @property
    def ren_win(self):
        return self.__dict__.get("_texture_render_window")

    @ren_win.setter
    def ren_win(self, value) -> None:
        still_constructing = not self.__dict__.get("_initialized", False)
        if value is None and still_constructing:
            return
        self.__dict__["_texture_render_window"] = value
