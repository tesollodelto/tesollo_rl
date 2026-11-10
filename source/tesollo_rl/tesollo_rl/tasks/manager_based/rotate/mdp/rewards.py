# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def object_rotation_reward( ##
    env: ManagerBasedRLEnv, 
    axis: tuple = (0.0, 0.0, 1.0),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    asset: RigidObject = env.scene[object_cfg.name]
    ang_vel_w = asset.data.root_ang_vel_w
    target_axis = torch.tensor(axis, device=ang_vel_w.device)
    
    return torch.sum(ang_vel_w * target_axis, dim=-1)

def object_to_tcp_distance_reward( ##
        env: ManagerBasedRLEnv, 
        std: float
    ) -> torch.Tensor:
    
    tcp_pos_w = env.scene.sensors["tcp_frame"].data.target_pos_w[:, 0, :]
    object_pos_w = env.scene["object"].data.root_pos_w
    distance = torch.norm(object_pos_w - tcp_pos_w, p=2, dim=-1)
    return torch.exp(-torch.square(distance) / (std ** 2)) 


def object_unstable_rotation_penalty(   ##
    env: ManagerBasedRLEnv, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:

    asset: RigidObject = env.scene[object_cfg.name]
    ang_vel_w = asset.data.root_ang_vel_w
    
    return torch.sum(torch.square(ang_vel_w[:, :2]), dim=1)


def joint_deviation_penalty(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    diff = robot.data.joint_pos - robot.data.default_joint_pos
    return torch.sum(diff ** 2, dim=-1)
