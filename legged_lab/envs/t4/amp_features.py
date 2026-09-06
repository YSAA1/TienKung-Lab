"""Compatibility adapter for existing t4 expert-generation entrypoints."""

from legged_lab.assets.t4.locomotion import T4_LOCOMOTION
from legged_lab.locomotion.amp_features import AmpFeatureBuilder, site_pos_in_root_frame


class T4AmpFeatureBuilder(AmpFeatureBuilder):
    def __init__(self, robot, device):
        super().__init__(robot, device, T4_LOCOMOTION)
