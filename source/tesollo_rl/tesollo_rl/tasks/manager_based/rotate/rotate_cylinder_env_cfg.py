# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

from . import mdp
from tesollo_rl.assets import ASSET_DATA_DIR

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.sensors import ContactSensorCfg

##
# Scene definition
##


@configclass
class TesolloSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene with an object and a dexterous hand."""

    # robots
    robot: ArticulationCfg = MISSING   # type:ignore 

    tcp_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/robot/link_base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/link_base",
                name="tcp",
                offset=OffsetCfg(
                    pos=(0.1, 0, 0.18),
                    rot=(0.9238795, 0, 0.3826834, 0),
                ),  
            ),
        ],
        debug_vis=True,
    )

    # objects
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(
                ASSET_DATA_DIR / "objects" / "cylinder" / "object.usd"
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.01, 
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0, 0.55),
            rot=(0.9238795, 0, -0.3826834, 0)
        ),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.95, 0.95, 0.95), intensity=1000.0),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/domeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.02, 0.02, 0.02), intensity=1000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    null_command = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        alpha=0.95,
        rescale_to_limits=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class KinematicObsGroupCfg(ObsGroup):
        ## observation terms
        # robot
        joint_pos = ObsTerm(
            func=mdp.joint_pos_limit_normalized, 
            noise=Gnoise(std=0.01)
        )

        # action
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    ## observation groups
    policy: KinematicObsGroupCfg = KinematicObsGroupCfg()


@configclass
class EventCfg:
    # object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material, #type:ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (0.3, 0.7),
            "dynamic_friction_range": (0.3, 0.7),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 400,
        },
    )

    # reset
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_within_limits_range,  #type:ignore
        mode="reset",
        params={
            "position_range": {".*": [-0.0, 0.0]},
            "velocity_range": {".*": [0.0, 0.0]},
            "use_default_offset": True,
            "operation": "scale",
        },
    )

@configclass
class RewardsCfg:
    rotate_z = RewTerm(
        func=mdp.object_rotation_reward, 
        weight=1.0, 
        params={
            "axis": (0.0, 0.0, 1.0),
            "object_cfg": SceneEntityCfg("object"),
        }
    )

    track_pos = RewTerm(
        func=mdp.object_to_tcp_distance_reward,
        weight=5.0,
        params={
            "std": 0.02
        }
    )

    unstable_penalty = RewTerm(
        func=mdp.object_unstable_rotation_penalty, 
        weight=-0.4, 
        params={"object_cfg": SceneEntityCfg("object")}
    )

    joint_deviation = RewTerm(
        func=mdp.joint_deviation_penalty,
        weight=-0.4,
    )

    joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01,
    )

    object_dropped_penalty = RewTerm(
        func=mdp.root_height_below_minimum,
        weight=-50.0,
        params={"minimum_height": 0.4, "asset_cfg": SceneEntityCfg("object")}
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(
        func=mdp.time_out, 
        time_out=True
    )

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={"minimum_height": 0.4, "asset_cfg": SceneEntityCfg("object")},
        time_out=True
    )


##
# Environment configuration
##


@configclass
class TesolloEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the in hand reorientation environment."""

    # Scene settings
    scene: TesolloSceneCfg = TesolloSceneCfg(num_envs=1028, env_spacing=0.5)
    
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            gpu_collision_stack_size = 2**30,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**25,
            gpu_max_rigid_patch_count=2**25
        ),
    )
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """post initialization."""

        # general settings
        self.decimation = 2
        self.episode_length_s = 20.0

        # simulation settings
        self.sim.dt = 1.0 / 100.0
        self.sim.render_interval = self.decimation

        # change viewer settings
        self.viewer.eye = (-2.0, 2.0, 3.0)
