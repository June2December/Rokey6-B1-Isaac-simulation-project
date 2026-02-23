"""
command_utils.py — Isaac Lab command_manager 조작 유틸리티

핵심:
  command_manager.compute()는 env.step() 내부에서 매 스텝 커맨드를 덮어씀.
  따라서 커맨드 강제 주입은 반드시 env.step() 이후에 수행해야 유지됨.
"""
import torch
from isaaclab.envs import ManagerBasedRLEnv


def get_target_world_pos(raw_env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    command 버퍼(로봇 기준 상대 좌표)를 역변환 → 월드 XY 좌표.
    수집 판정 시 현재 목표 광물의 실제 위치를 구할 때 사용.

    Returns: shape (num_envs, 2)
    """
    robot = raw_env.scene["robot"]
    robot_pos_w  = robot.data.root_pos_w[:, :2]
    robot_quat_w = robot.data.root_quat_w

    cmd   = raw_env.command_manager.get_command("target_pose")
    rel_x = cmd[:, 0]
    rel_y = cmd[:, 1]

    w, x, y, z = robot_quat_w[:,0], robot_quat_w[:,1], robot_quat_w[:,2], robot_quat_w[:,3]
    yaw = torch.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

    cos_yaw  = torch.cos(yaw)
    sin_yaw  = torch.sin(yaw)
    world_dx = cos_yaw * rel_x - sin_yaw * rel_y
    world_dy = sin_yaw * rel_x + cos_yaw * rel_y

    return robot_pos_w + torch.stack([world_dx, world_dy], dim=-1)


def resample_target(raw_env: ManagerBasedRLEnv, env_idx: int):
    """
    env_idx 로봇의 광물 커맨드만 새 위치로 리샘플.
    env.reset() 없이 해당 로봇의 목표 광물 위치만 교체.
    """
    try:
        env_ids = torch.tensor([env_idx], device=raw_env.device)
        raw_env.command_manager._terms["target_pose"].reset(env_ids=env_ids)
        print(f"[Robot{env_idx}] 다음 광물 위치로 커맨드 리샘플")
    except Exception as e:
        print(f"[Robot{env_idx}] 커맨드 리샘플 실패: {e}")


def freeze_resample_timer(raw_env: ManagerBasedRLEnv, env_idx: int):
    """리샘플 타이머 차단 — 베이스 복귀 중 광물 위치로 자동 변경 방지."""
    try:
        term = raw_env.command_manager._terms["target_pose"]
        attr = "time_left" if hasattr(term, "time_left") else "_time_left"
        getattr(term, attr)[env_idx] = 999999.0
    except Exception:
        pass


def set_command_to_basecamp(raw_env: ManagerBasedRLEnv, env_idx: int,
                             cx: float, cy: float, device):
    """
    env_idx 로봇의 커맨드를 베이스캠프 방향으로 강제 주입.

    ★ 반드시 env.step() 이후에 호출해야 함.
      step() 내부 command_manager.compute()가 커맨드를 덮어쓰기 때문.

    pos_command_w(절대 좌표)와 cmd 버퍼(상대 좌표) 모두 갱신해서
    command_manager.compute()가 재계산해도 베이스 방향이 유지되도록 함.
    """
    term = raw_env.command_manager._terms["target_pose"]

    # 절대 좌표 목표 덮어쓰기 (compute()가 이걸 기준으로 상대 좌표 재계산)
    if hasattr(term, "pos_command_w"):
        term.pos_command_w[env_idx, 0] = cx
        term.pos_command_w[env_idx, 1] = cy

    # 현재 프레임 상대 커맨드 버퍼 갱신
    robot  = raw_env.scene["robot"]
    pos_w  = robot.data.root_pos_w[env_idx:env_idx+1, :2]
    quat_w = robot.data.root_quat_w[env_idx:env_idx+1, :]

    target = torch.tensor([[cx, cy]], dtype=torch.float32, device=device)
    delta  = target - pos_w

    w  = quat_w[:,0]; xq = quat_w[:,1]; yq = quat_w[:,2]; zq = quat_w[:,3]
    yaw     = torch.atan2(2.0*(w*zq + xq*yq), 1.0 - 2.0*(yq*yq + zq*zq))
    cos_yaw = torch.cos(-yaw)
    sin_yaw = torch.sin(-yaw)
    rel_x   = cos_yaw * delta[:,0] - sin_yaw * delta[:,1]
    rel_y   = sin_yaw * delta[:,0] + cos_yaw * delta[:,1]
    angle   = torch.atan2(rel_y, rel_x)

    cmd = raw_env.command_manager.get_command("target_pose")
    cmd[env_idx, 0] = rel_x[0]
    cmd[env_idx, 1] = rel_y[0]
    cmd[env_idx, 2] = 0.0
    cmd[env_idx, 3] = angle[0]

    freeze_resample_timer(raw_env, env_idx)