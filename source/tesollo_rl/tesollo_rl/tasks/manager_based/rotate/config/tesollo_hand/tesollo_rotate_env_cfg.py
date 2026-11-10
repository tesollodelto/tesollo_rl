# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ... import rotate_cylinder_env_cfg

##
# Pre-defined configs
##
from tesollo_rl.assets.robots.tesollo import DG5FM_RIGHT_CFG

@configclass
class TesolloRotateEnvCfg(rotate_cylinder_env_cfg.TesolloEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = DG5FM_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")   #type:ignore
        
@configclass
class TesolloRotateEnvCfg_PLAY(TesolloRotateEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.terminations.time_out = None   #type:ignore
