# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp
from .lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from .ThreeFingers import UR5_CFG_PD_CFG  # isort: skip


@configclass
class UR5CubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = UR5_CFG_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["shoulder_pan_joint",
                                             "shoulder_lift_joint",
                                             "elbow_joint",
                                             "wrist_1_joint",
                                             "wrist_2_joint",
                                             "wrist_3_joint",
                                             "joint1",
                                             "joint2"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint3"],
            open_command_expr={"joint3": 0.0},
            close_command_expr={"joint3": 0.02},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "body_link"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0, 0.06], rot=[0.5, -0.5, -0.5, 0.5]),
            spawn=UsdFileCfg(
                usd_path=f"/home/yang/Repo_USD/card.usd",
                scale=(0.1, 0.002, 0.05),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        self.scene.desk = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/DeskUp",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0, 0.03], rot=[1, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(
                usd_path=f"/home/yang/Repo_USD/card.usd",
                scale=(0.4, 0.4, 0.02),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/eeFrame"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5_physics/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur5_physics/body_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.12, 0.0],
                    ),
                ),
            ],
        )
        # 可视化物块坐标系
        card_marker_cfg = FRAME_MARKER_CFG.copy()
        card_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        card_marker_cfg.prim_path = "/Visuals/CardFrame"
        self.scene.card_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5_physics/base_link",  
            debug_vis=True,
            visualizer_cfg=card_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Object/card", 
                    name="card_frame",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),  
                ),
            ],
        )


@configclass
class UR5CubeLiftEnvCfg_PLAY(UR5CubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False