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


def all_contact_forces(env: ManagerBasedRLEnv, sensor_names: list[str]) -> torch.Tensor:
    force_tensors = []
    
    for name in sensor_names:
        sensor: ContactSensor = env.scene[name]
        force_tensors.append(sensor.data.net_forces_w.view(env.num_envs, -1))
        
    return torch.cat(force_tensors, dim=-1)