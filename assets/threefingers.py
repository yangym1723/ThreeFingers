# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control
* :obj:`FRANKA_ROBOTIQ_GRIPPER_CFG`: Franka robot with Robotiq_2f_85 gripper

Reference: https://github.com/frankaemika/franka_ros
"""


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

THREEFINGERS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/yang/Repo_USD/v1.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.04,
            "elbow_joint": 1.04,
            "wrist_1_joint": -1.57,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": 0.0,
            "joint1" : 0.0,
            "joint2" : 0.0,
            "joint3" : 0.0,
        },
    ),
    actuators={
        "ur5_shoulder_elbow": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint","shoulder_lift_joint","elbow_joint"],
            effort_limit_sim=87.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "ur5_wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*_joint"],
            effort_limit_sim=12.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "ur5_hand": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],
            effort_limit_sim=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


THREEFINGERS_CFG_PD_CFG = THREEFINGERS_CFG.copy()
THREEFINGERS_CFG_PD_CFG.spawn.rigid_props.disable_gravity = True
THREEFINGERS_CFG_PD_CFG.actuators["ur5_shoulder_elbow"].stiffness = 400.0
THREEFINGERS_CFG_PD_CFG.actuators["ur5_shoulder_elbow"].damping = 80.0
THREEFINGERS_CFG_PD_CFG.actuators["ur5_wrist"].stiffness = 400.0
THREEFINGERS_CFG_PD_CFG.actuators["ur5_wrist"].damping = 80.0
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""