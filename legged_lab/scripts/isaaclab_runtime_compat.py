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


def patch_missing_physx_material_attributes() -> None:
    """Skip PhysX material attributes removed from the active Isaac Sim build."""

    import isaaclab.sim.utils as sim_utils_module
    import isaaclab.sim.spawners.materials.physics_materials as physics_materials

    if getattr(sim_utils_module, "_t4_physx_material_attr_patch", False):
        return

    original_safe_set_attribute = sim_utils_module.safe_set_attribute_on_usd_schema

    def patched_safe_set_attribute(schema_api, name, value, camel_case):
        try:
            return original_safe_set_attribute(schema_api, name, value, camel_case)
        except TypeError:
            if name == "improve_patch_friction":
                return None
            raise

    sim_utils_module.safe_set_attribute_on_usd_schema = patched_safe_set_attribute
    physics_materials.safe_set_attribute_on_usd_schema = patched_safe_set_attribute
    sim_utils_module._t4_physx_material_attr_patch = True
