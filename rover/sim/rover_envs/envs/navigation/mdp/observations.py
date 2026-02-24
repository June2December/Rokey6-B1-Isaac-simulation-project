from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def angle_to_target_observation(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """로봇에서 목표까지의 각도 (로봇 기준 상대 벡터 기반)"""
    target_vector_b = env.command_manager.get_command(command_name)[:, :2]
    angle = torch.atan2(target_vector_b[:, 1], target_vector_b[:, 0])
    return angle.unsqueeze(-1)


def distance_to_target_euclidean(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """목표까지의 유클리드 거리"""
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]
    distance: torch.Tensor = torch.norm(target_position, p=2, dim=-1)
    return distance.unsqueeze(-1)


def height_scan_rover(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """로봇 주변 높이 스캔"""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - 0.26878


def angle_diff(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """로봇 헤딩과 목표 방향의 각도 차이"""
    heading_angle_diff = env.command_manager.get_command(command_name)[:, 3]
    return heading_angle_diff.unsqueeze(-1)


def get_robot_world_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    로봇의 월드 좌표(XY)를 반환.
    eval.py에서 직접 호출하여 광물/베이스먼트 거리 판정에 사용.

    Returns:
        shape (num_envs, 2) — XY 좌표
    """
    robot = env.scene["robot"]
    pos_w = robot.data.root_pos_w  # shape: (num_envs, 3)
    return pos_w[:, :2]


def get_robot_world_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    로봇의 월드 쿼터니언을 반환.
    eval.py에서 1인칭 카메라 방향 계산에 사용.

    Returns:
        shape (num_envs, 4) — w, x, y, z
    """
    robot = env.scene["robot"]
    return robot.data.root_quat_w  # shape: (num_envs, 4)


def override_command_target(env: ManagerBasedRLEnv, command_name: str, world_target_xy: torch.Tensor) -> None:
    """
    command_manager의 목표 좌표를 월드 XY 좌표로 강제 설정.
    eval.py에서 광물/베이스먼트로 목표를 전환할 때 사용.

    내부적으로 command는 로봇 기준 상대 벡터이므로,
    월드 좌표 → 로봇 기준 상대 좌표로 변환 후 주입.

    Args:
        env: 환경 인스턴스
        command_name: 커맨드 이름 (예: "target_pose")
        world_target_xy: 목표의 월드 XY 좌표 shape (2,) or (num_envs, 2)
    """
    robot = env.scene["robot"]
    robot_pos_w = robot.data.root_pos_w[:, :2]   # (num_envs, 2)
    robot_quat_w = robot.data.root_quat_w          # (num_envs, 4)  w,x,y,z

    num_envs = robot_pos_w.shape[0]
    device = robot_pos_w.device

    # world_target_xy를 (num_envs, 2) 형태로 맞추기
    if world_target_xy.dim() == 1:
        world_target_xy = world_target_xy.unsqueeze(0).expand(num_envs, -1)
    world_target_xy = world_target_xy.to(device)

    # 월드 상대 벡터 (아직 로봇 회전 미적용)
    delta_w = world_target_xy - robot_pos_w  # (num_envs, 2)

    # 로봇 yaw 각도 추출 (쿼터니언 → yaw)
    w = robot_quat_w[:, 0]
    x = robot_quat_w[:, 1]
    y = robot_quat_w[:, 2]
    z = robot_quat_w[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))  # (num_envs,)

    # 월드 벡터를 로봇 바디 프레임으로 회전 (2D 역회전)
    cos_yaw = torch.cos(-yaw)
    sin_yaw = torch.sin(-yaw)
    rel_x = cos_yaw * delta_w[:, 0] - sin_yaw * delta_w[:, 1]
    rel_y = sin_yaw * delta_w[:, 0] + cos_yaw * delta_w[:, 1]

    # 헤딩 오차 계산
    angle_to_target = torch.atan2(rel_y, rel_x)

    # command buffer에 직접 주입
    # command 구조: [rel_x, rel_y, 0(z), heading_diff]
    cmd = env.command_manager.get_command(command_name)  # (num_envs, 4)
    cmd[:, 0] = rel_x
    cmd[:, 1] = rel_y
    cmd[:, 2] = 0.0
    cmd[:, 3] = angle_to_target
