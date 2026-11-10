# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from tesollo_rl.assets import ASSET_DATA_DIR

##
# Configuration - Actuators.
##


DG5FM_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(
            ASSET_DATA_DIR / "robots" / "dg5f_right" / "dg5f_right.usd"
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.002,
            rest_offset=0.0
        ),  
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.2, 0.0, 0.5),
        rot=(0.6087614, 0, -0.7933533, 0),

        joint_pos={
            "joint_1_1": math.radians(11.8),
            "joint_1_2": math.radians(-90.0),
            "joint_1_3": math.radians(36.9),
            "joint_1_4": math.radians(0),

            "joint_2_1": math.radians(-27.1),
            "joint_2_2": math.radians(66.9),
            "joint_2_3": math.radians(18.1),
            "joint_2_4": math.radians(21.6),

            "joint_3_1": math.radians(0.0),
            "joint_3_2": math.radians(64.6),
            "joint_3_3": math.radians(0.0),
            "joint_3_4": math.radians(0.0),

            "joint_4_1": math.radians(20.0),
            "joint_4_2": math.radians(72.1),
            "joint_4_3": math.radians(0.0),
            "joint_4_4": math.radians(0.0),

            "joint_5_1": math.radians(30.0),
            "joint_5_2": math.radians(72.1),
            "joint_5_3": math.radians(72.7),
            "joint_5_4": math.radians(23.4),
        },
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=10.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
