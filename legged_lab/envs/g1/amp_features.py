"""Compatibility adapter for existing g1 expert-generation entrypoints."""

from legged_lab.assets.unitree_g1.locomotion import G1_LOCOMOTION
from legged_lab.locomotion.amp_features import AmpFeatureBuilder, site_pos_in_root_frame


class G1AmpFeatureBuilder(AmpFeatureBuilder):
    def __init__(self, robot, device):
        super().__init__(robot, device, G1_LOCOMOTION)
