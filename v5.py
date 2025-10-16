# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# 版权声明：本代码遵循BSD-3-Clause许可证
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
脚本功能：运行带有抓取和举起状态机的环境
本脚本实现了一个机械臂抓取物体的完整状态机控制系统

状态机实现在内核函数 `infer_state_machine` 中
使用 `warp` 库在GPU上并行运行状态机，提高计算效率

使用方法示例：
... code-block:: bash

    ./isaaclab.sh -p scripts/environments/state_machine/lift_cube_sm.py --num_envs 32

"""

"""首先启动 Omniverse Toolkit（Isaac Sim的基础平台）"""

# 导入命令行参数解析库，用于处理脚本运行时的参数
import argparse

# 从isaaclab.app导入AppLauncher，这是Isaac Lab的应用启动器
from isaaclab.app import AppLauncher

# 创建命令行参数解析器，用于配置环境参数
# 参数解析器可以接收用户在命令行中输入的配置选项
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")

# 添加是否禁用fabric的参数选项
# fabric是一种优化的I/O操作方式，禁用后会使用标准的USD I/O操作（较慢但更稳定）
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

# 添加环境数量参数，默认为1个环境
# 可以同时运行多个并行环境以加速训练或测试
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")

# 将AppLauncher的命令行参数添加到解析器中
# 这包括headless模式、渲染设置等Isaac Lab特有的参数
AppLauncher.add_app_launcher_args(parser)

# 解析所有命令行参数并保存到args_cli变量中
args_cli = parser.parse_args()

# 创建并启动Omniverse应用
# headless参数决定是否显示图形界面（headless=True时不显示，适合服务器运行）
app_launcher = AppLauncher(headless=args_cli.headless)

# 获取simulation_app应用实例，这是整个仿真的核心对象
simulation_app = app_launcher.app

"""导入其他所有必需的库和模块"""

# 导入gymnasium（OpenAI Gym的继任者），用于强化学习环境接口
import gymnasium as gym

# 导入PyTorch，用于张量计算和深度学习
import torch

# 导入Sequence类型，用于类型提示
from collections.abc import Sequence

# 导入warp库，NVIDIA开发的用于GPU并行计算的Python框架
import warp as wp

# 导入刚体对象数据类，用于访问物体的物理状态信息
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData

# 导入isaaclab_tasks模块，包含预定义的任务环境
# noqa: F401 表示告诉代码检查工具忽略"未使用的导入"警告
import isaaclab_tasks  # noqa: F401

# 导入本地的assets模块，包含机器人配置（threefingers.py等）
import assets  # noqa: F401

# 从isaaclab_tasks导入LiftEnvCfg，这是举起任务的环境配置基类
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

# 导入环境配置解析工具，用于根据环境名称创建配置对象
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# 初始化warp库，准备GPU计算环境
wp.init()


class GripperState:
    """夹爪状态类：定义夹爪的两种状态常量"""
    
    # OPEN状态：夹爪打开，值为1.0
    # wp.constant表示这是一个warp常量，在GPU上执行时更高效
    OPEN = wp.constant(1.0)
    
    # CLOSE状态：夹爪闭合，值为-1.0
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """抓取状态机状态类：定义抓取任务的5个主要状态"""
    
    # REST: 初始休息状态，状态值为0
    REST = wp.constant(0)
    
    # APPROACH_ABOVE_OBJECT: 接近物体上方状态，状态值为1
    # 机械臂先移动到物体正上方的安全位置
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    
    # APPROACH_OBJECT: 接近物体状态，状态值为2
    # 机械臂从上方下降到物体位置
    APPROACH_OBJECT = wp.constant(2)
    
    # GRASP_OBJECT: 抓取物体状态，状态值为3
    # 夹爪闭合抓住物体
    GRASP_OBJECT = wp.constant(3)
    
    # LIFT_OBJECT: 举起物体状态，状态值为4
    # 将物体举起到目标位置（最终状态）
    LIFT_OBJECT = wp.constant(4)


class PickSmWaitTime:
    """状态切换等待时间类：定义每个状态在切换前需要等待的时间（单位：秒）"""
    
    # REST状态等待时间：0.2秒
    # 给系统一个短暂的初始化时间
    REST = wp.constant(0.2)
    
    # APPROACH_ABOVE_OBJECT状态等待时间：1.0秒
    # 确保机械臂稳定到达物体上方位置
    APPROACH_ABOVE_OBJECT = wp.constant(1.0)
    
    # APPROACH_OBJECT状态等待时间：1.0秒
    # 确保机械臂稳定下降到物体位置
    APPROACH_OBJECT = wp.constant(1.0)
    
    # GRASP_OBJECT状态等待时间：1.0秒
    # 给夹爪充足时间完全闭合并稳定抓握
    GRASP_OBJECT = wp.constant(1.0)
    
    # LIFT_OBJECT状态等待时间：1.0秒
    # 确保物体被稳定举起到目标位置
    LIFT_OBJECT = wp.constant(1.0)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    """
    判断当前位置与目标位置的距离是否小于阈值
    
    这是一个warp函数装饰器标记的函数，可以在GPU上执行
    
    参数:
        current_pos: 当前位置（3D向量）
        desired_pos: 目标位置（3D向量）
        threshold: 距离阈值
    
    返回:
        bool: 如果距离小于阈值返回True，否则返回False
    """
    # 打印当前位置（用于调试）
    print(current_pos)
    # 打印目标位置（用于调试）
    print(desired_pos)
    # 计算两个位置之间的欧几里得距离，并与阈值比较
    # wp.length计算向量的长度（模）
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),                      # 时间步长数组
    sm_state: wp.array(dtype=int),                  # 状态机当前状态数组
    sm_wait_time: wp.array(dtype=float),            # 状态机等待时间数组
    ee_pose: wp.array(dtype=wp.transform),          # 末端执行器（End-Effector）位姿数组
    object_pose: wp.array(dtype=wp.transform),      # 物体当前位姿数组
    des_object_pose: wp.array(dtype=wp.transform),  # 物体目标位姿数组
    des_ee_pose: wp.array(dtype=wp.transform),      # 末端执行器期望位姿数组（输出）
    gripper_state: wp.array(dtype=float),           # 夹爪状态数组（输出）
    offset: wp.array(dtype=wp.transform),           # 接近物体时的偏移量
    position_threshold: float,                       # 位置判定阈值
):
    """
    状态机推理内核函数
    
    这是一个warp kernel，会在GPU上并行执行，每个环境一个线程
    根据当前状态和传感器数据，计算下一个状态和期望的末端执行器位姿
    
    该函数实现了完整的抓取状态机逻辑
    """
    # 获取当前线程ID，对应环境索引
    # 在GPU并行计算中，每个环境由一个线程处理
    tid = wp.tid()
    
    # 获取当前环境的状态机状态
    state = sm_state[tid]
    
    # 根据当前状态执行相应的逻辑
    # 状态机采用if-elif结构，每个状态有独立的处理逻辑
    
    if state == PickSmState.REST:
        # REST状态：初始状态，机械臂保持在当前位置
        
        # 期望位姿设置为当前末端执行器位姿（保持不动）
        des_ee_pose[tid] = ee_pose[tid]
        
        # 夹爪设置为打开状态
        gripper_state[tid] = GripperState.OPEN
        
        # 打印当前状态（用于调试）
        print("PickSmState:REST")
        
        # 检查是否已等待足够时间
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # 切换到下一个状态：APPROACH_ABOVE_OBJECT
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            # 重置等待时间计数器
            sm_wait_time[tid] = 0.0
            
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        # APPROACH_ABOVE_OBJECT状态：接近物体上方
        
        # 计算目标位置：物体位置 + 偏移量
        # wp.transform_multiply用于组合两个变换（位置和旋转）
        # offset定义了相对于物体的偏移（通常是正上方某个高度）
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        
        # 夹爪保持打开状态
        gripper_state[tid] = GripperState.OPEN
        
        # 打印当前状态
        print("PickSmState:APPROACH_ABOVE_OBJECT")
        
        # 检查末端执行器是否已到达目标位置
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),      # 当前位置
            wp.transform_get_translation(des_ee_pose[tid]),  # 目标位置
            position_threshold,                               # 位置阈值
        ):
            # 到达目标位置后，检查是否已等待足够时间
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # 切换到下一个状态：APPROACH_OBJECT
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                # 重置等待时间
                sm_wait_time[tid] = 0.0
                
    elif state == PickSmState.APPROACH_OBJECT:
        # APPROACH_OBJECT状态：从上方下降到物体位置
        
        # 期望位姿直接设置为物体当前位姿
        # 机械臂将下降到与物体相同的高度
        des_ee_pose[tid] = object_pose[tid]
        
        # 夹爪保持打开状态，准备抓取
        gripper_state[tid] = GripperState.OPEN
        
        # 打印当前状态
        print("PickSmState:APPROACH_OBJECT")
        
        # 检查末端执行器是否已到达物体位置
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),      # 当前位置
            wp.transform_get_translation(des_ee_pose[tid]),  # 目标位置（物体位置）
            position_threshold,                               # 位置阈值
        ):
            # 到达物体位置后，检查等待时间
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # 切换到下一个状态：GRASP_OBJECT
                sm_state[tid] = PickSmState.GRASP_OBJECT
                # 重置等待时间
                sm_wait_time[tid] = 0.0
                
    elif state == PickSmState.GRASP_OBJECT:
        # GRASP_OBJECT状态：抓取物体
        
        # 保持末端执行器在物体位置
        des_ee_pose[tid] = object_pose[tid]
        
        # 夹爪切换到闭合状态，抓住物体
        gripper_state[tid] = GripperState.CLOSE
        
        # 打印当前状态
        print("PickSmState:GRASP_OBJECT")
        
        # 等待足够时间让夹爪完全闭合并稳定抓握
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # 切换到最后一个状态：LIFT_OBJECT
            sm_state[tid] = PickSmState.LIFT_OBJECT
            # 重置等待时间
            sm_wait_time[tid] = 0.0
            
    elif state == PickSmState.LIFT_OBJECT:
        # LIFT_OBJECT状态：举起物体到目标位置（最终状态）
        
        # 期望位姿设置为目标物体位姿（通常是某个举起的高度）
        des_ee_pose[tid] = des_object_pose[tid]
        
        # 夹爪保持闭合状态，继续抓住物体
        gripper_state[tid] = GripperState.CLOSE
        
        # 打印当前状态
        print("PickSmState:LIFT_OBJECT")
        
        # 检查是否已举起到目标位置
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),          # 当前位置
            wp.transform_get_translation(des_ee_pose[tid]),      # 目标位置
            position_threshold,                                   # 位置阈值
        ):
            # 到达目标位置后，检查等待时间
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                # 保持在LIFT_OBJECT状态（任务完成）
                sm_state[tid] = PickSmState.LIFT_OBJECT
                # 重置等待时间
                sm_wait_time[tid] = 0.0
                
    # 增加等待时间计数器
    # 每次调用kernel时，等待时间增加一个时间步长
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """
    抓取和举起状态机类
    
    这是一个简单的机器人任务空间状态机，用于抓取和举起物体
    
    状态机使用warp kernel实现，输入当前机器人末端执行器和物体的状态，
    输出期望的末端执行器状态和夹爪状态
    
    状态机包含以下5个状态的有限状态机：
    1. REST: 机器人处于休息状态
    2. APPROACH_ABOVE_OBJECT: 机器人移动到物体上方
    3. APPROACH_OBJECT: 机器人移动到物体位置
    4. GRASP_OBJECT: 机器人抓取物体
    5. LIFT_OBJECT: 机器人将物体举起到期望位姿（最终状态）
    
    空间位置配置说明：
    - 机械臂的初始关节位置在threefingers.py中定义（THREEFINGERS_CFG.init_state.joint_pos）
    - UR5机械臂包含6个关节：shoulder_pan_joint, shoulder_lift_joint, elbow_joint, 
      wrist_1_joint, wrist_2_joint, wrist_3_joint
    - 三指手包含3个关节：joint1, joint2, joint3
    - 末端执行器偏移量在joint_pos_env_cfg.py和ik_abs_env_cfg.py中配置
    - IK控制器在ik_abs_env_cfg.py中配置，使用DifferentialIKControllerCfg
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.1):
        """
        初始化状态机
        
        参数:
            dt: 环境时间步长（单位：秒），决定了仿真的时间精度
            num_envs: 要同时仿真的环境数量，支持并行仿真多个环境
            device: 运行状态机的设备（"cpu"或"cuda"），GPU可以显著加速计算
            position_threshold: 位置判定阈值（单位：米），用于判断末端执行器是否到达目标位置
        """
        # 保存时间步长参数（转换为float类型）
        self.dt = float(dt)
        
        # 保存环境数量
        self.num_envs = num_envs
        
        # 保存计算设备
        self.device = device
        
        # 保存位置阈值
        self.position_threshold = position_threshold
        
        # 初始化状态机的时间步长数组
        # 所有环境使用相同的时间步长
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        
        # 初始化状态机状态数组
        # 所有环境从状态0（REST）开始
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        
        # 初始化等待时间数组
        # 所有环境的等待时间从0开始
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # 初始化期望末端执行器位姿数组
        # 7维：[x, y, z, qx, qy, qz, qw]（位置3维 + 四元数4维）
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        
        # 初始化期望夹爪状态数组
        # 初始值为0.0
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # 初始化接近物体上方时的偏移量数组
        # 这个偏移定义了"物体上方"的具体位置
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        
        # 设置Z轴偏移（高度）为0.08米，表示在物体上方8厘米
        # 这个值与joint_pos_env_cfg.py中的OffsetCfg配置相关
        self.offset[:, 2] = 0.08
        
        # 设置Y轴偏移为0.1米，表示在物体Y方向偏移10厘米
        self.offset[:, 1] = 0.1
        
        # 设置四元数的w分量为1.0（表示无旋转）
        # warp期望四元数格式为(x, y, z, w)
        self.offset[:, -1] = 1.0

        # 将PyTorch张量转换为warp数组，以便在GPU kernel中使用
        # warp数组可以直接在GPU上高效访问
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """
        重置指定环境的状态机
        
        当环境完成任务或失败时，需要重置状态机到初始状态
        
        参数:
            env_ids: 要重置的环境ID列表，如果为None则重置所有环境
        """
        # 如果没有指定环境ID，则重置所有环境
        if env_ids is None:
            env_ids = slice(None)
            
        # 将指定环境的状态重置为0（REST状态）
        self.sm_state[env_ids] = 0
        
        # 将指定环境的等待时间重置为0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor) -> torch.Tensor:
        """
        计算机器人末端执行器和夹爪的期望状态
        
        这是状态机的主要计算函数，每个时间步调用一次
        
        参数:
            ee_pose: 末端执行器当前位姿 [num_envs, 7]，格式为[x,y,z,w,qx,qy,qz]
            object_pose: 物体当前位姿 [num_envs, 7]，格式为[x,y,z,w,qx,qy,qz]
            des_object_pose: 物体期望位姿 [num_envs, 7]，格式为[x,y,z,w,qx,qy,qz]
        
        返回:
            期望动作 [num_envs, 8]：前7维是期望位姿[x,y,z,w,qx,qy,qz]，最后1维是夹爪状态
        
        注意：
        - Isaac Lab使用四元数格式(w,x,y,z)，而warp使用(x,y,z,w)
        - 需要在输入和输出时进行格式转换
        """
        # 将四元数格式从(w,x,y,z)转换为(x,y,z,w)以适配warp
        # 索引[0,1,2,4,5,6,3]表示：保持位置[0,1,2]，交换四元数[w,x,y,z]为[x,y,z,w]
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # 将PyTorch张量转换为warp数组
        # contiguous()确保内存连续，这对GPU访问很重要
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        # 在GPU上启动状态机kernel
        # dim参数指定并行线程数（等于环境数量）
        # 每个环境由一个独立线程处理
        wp.launch(
            kernel=infer_state_machine,  # 要执行的kernel函数
            dim=self.num_envs,            # 并行线程数
            inputs=[
                self.sm_dt_wp,             # 时间步长
                self.sm_state_wp,          # 状态机状态
                self.sm_wait_time_wp,      # 等待时间
                ee_pose_wp,                # 末端执行器位姿
                object_pose_wp,            # 物体位姿
                des_object_pose_wp,        # 目标物体位姿
                self.des_ee_pose_wp,       # 输出：期望末端执行器位姿
                self.des_gripper_state_wp, # 输出：夹爪状态
                self.offset_wp,            # 偏移量
                self.position_threshold,   # 位置阈值
            ],
            device=self.device,            # 计算设备
        )

        # 将四元数格式从(x,y,z,w)转换回(w,x,y,z)以适配Isaac Lab
        # 索引[0,1,2,6,3,4,5]表示：保持位置[0,1,2]，交换四元数[x,y,z,w]为[w,x,y,z]
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        
        # 将期望位姿和夹爪状态拼接成完整的动作向量
        # unsqueeze(-1)将夹爪状态从[num_envs]变为[num_envs, 1]
        # 最终返回[num_envs, 8]的张量
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
    """
    主函数：设置并运行仿真环境
    
    该函数完成以下任务：
    1. 解析和创建环境配置
    2. 创建gymnasium环境
    3. 初始化状态机
    4. 运行仿真循环
    5. 清理资源
    """
    # 解析环境配置
    # "Isaac-Lift-Cube-UR5-IK-Abs-v0"是在assets/__init__.py中注册的环境名称
    # 该环境使用ik_abs_env_cfg.py中定义的UR5CubeLiftEnvCfg配置
    # 配置包括：
    # - 机器人配置：使用THREEFINGERS_CFG_PD_CFG（threefingers.py中定义）
    # - IK控制器：DifferentialIKControllerCfg（ik_abs_env_cfg.py第45行）
    # - 末端执行器偏移：body_offset (0.0, 0.0, 0.105)（ik_abs_env_cfg.py第46行）
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-UR5-IK-Abs-v0",        # 环境名称
        device=args_cli.device,                  # 计算设备（CPU或CUDA）
        num_envs=args_cli.num_envs,             # 环境数量
        use_fabric=not args_cli.disable_fabric, # 是否使用fabric加速
    )
    
    # 使用gymnasium创建环境实例
    # cfg参数传入上面解析的配置
    env = gym.make("Isaac-Lift-Cube-UR5-IK-Abs-v0", cfg=env_cfg)
    
    # 重置环境到初始状态
    # 这会初始化所有机器人、物体和传感器
    env.reset()

    # 创建动作缓冲区
    # 动作空间维度由环境定义，对于IK控制通常是8维（位姿7维 + 夹爪1维）
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    
    # 初始化动作的四元数w分量为1.0（表示无旋转）
    # 这确保初始姿态是有效的
    actions[:, 3] = 1.0
    
    # 定义期望的物体方向（四元数）
    # 这个方向在状态机中用于计算目标位姿
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    
    # 设置期望方向的四元数
    # [0.707, -0.707, 0, 0] 表示绕X轴旋转90度
    desired_orientation[:, 0] = 0.707
    desired_orientation[:, 1] = -0.707

    # 创建抓取状态机实例
    # 参数说明：
    # - dt: 时间步长 = 仿真时间步 × decimation（控制频率下采样因子）
    # - num_envs: 环境数量
    # - device: 计算设备
    # - position_threshold: 0.01米（1厘米）的位置判定阈值
    pick_sm = PickAndLiftSm(
        env_cfg.sim.dt * env_cfg.decimation,  # 控制时间步长
        env.unwrapped.num_envs,                # 环境数量
        env.unwrapped.device,                  # 计算设备
        position_threshold=0.01                # 位置阈值
    )

    # 仿真主循环
    # 持续运行直到仿真应用关闭
    while simulation_app.is_running():
        # 在推理模式下运行（禁用梯度计算，节省内存和提高速度）
        with torch.inference_mode():
            # 执行一步环境仿真
            # 返回值包括：observation, reward, terminated, truncated, info
            # 我们只关心dones（terminated或truncated）
            dones = env.step(actions)[-2]

            # 获取观测数据
            # -- 末端执行器坐标系
            # ee_frame是在joint_pos_env_cfg.py第68-81行定义的FrameTransformerCfg
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            
            # 获取末端执行器的世界坐标位置
            # target_pos_w [..., 0, :]选择第一个目标帧（end_effector）
            # 减去env_origins将世界坐标转换为相对环境原点的坐标
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            
            # 获取末端执行器的世界坐标方向（四元数）
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            
            # -- 物体坐标系
            # 获取物体的刚体数据
            # object是在joint_pos_env_cfg.py第47-62行定义的RigidObjectCfg
            object_data: RigidObjectData = env.unwrapped.scene["object"].data
            
            # 获取物体的相对位置
            # root_pos_w是物体根部在世界坐标系中的位置
            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
            
            # -- 目标物体坐标系
            # 从命令管理器获取物体的目标位置（前3维是xyz坐标）
            # 这个目标位置由环境的reward和termination逻辑使用
            desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]

            # 推进状态机计算
            # 输入当前的末端执行器位姿、物体位姿和目标位姿
            # 输出期望的动作（位姿 + 夹爪状态）
            actions = pick_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),  # 当前末端执行器位姿
                torch.cat([object_position, desired_orientation], dim=-1),      # 当前物体位姿
                torch.cat([desired_position, desired_orientation], dim=-1),     # 目标物体位姿
            )

            # 重置状态机
            # 如果有环境完成任务或失败（dones为True）
            if dones.any():
                # 找出需要重置的环境ID
                # nonzero返回非零元素的索引
                # squeeze(-1)移除最后一个维度
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

    # 关闭环境，释放资源
    env.close()


if __name__ == "__main__":
    # 程序入口点
    # 运行主函数
    main()
    
    # 关闭仿真应用
    # 这会清理所有Isaac Sim资源
    simulation_app.close()
