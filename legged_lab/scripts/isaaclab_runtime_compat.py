"""Runtime compatibility shims for narrow IsaacLab/Isaac Sim mismatches."""

from __future__ import annotations

from typing import Type


def patch_physx_backward_compatibility_setting(app_launcher_cls: Type) -> None:
    """Patch IsaacLab 2.1 AppLauncher for PhysX builds without this constant.

    Some Isaac Sim 5.x builds expose the same carb setting path but do not
    export ``SETTING_BACKWARD_COMPATIBILITY`` from ``omni.physx.bindings``.
    IsaacLab only needs the string constant before setting it to 0.
    """

    if getattr(app_launcher_cls, "_t4_physx_backward_compat_patch", False):
        return

    original_load_extensions = app_launcher_cls._load_extensions

    def patched_load_extensions(self):
        import omni.physx.bindings._physx as physx_impl

        if not hasattr(physx_impl, "SETTING_BACKWARD_COMPATIBILITY"):
            physx_impl.SETTING_BACKWARD_COMPATIBILITY = "/physics/backwardCompatibility"
        return original_load_extensions(self)

    app_launcher_cls._load_extensions = patched_load_extensions
    app_launcher_cls._t4_physx_backward_compat_patch = True
