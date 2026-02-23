"""
robot_utils.py — 로봇 물리 상태 조작 유틸리티
"""
from isaaclab.envs import ManagerBasedRLEnv
import torch


def teleport_to_basecamp(raw_env: ManagerBasedRLEnv, env_idx: int,
                          cx: float, cy: float):
    """
    env_idx 로봇만 베이스캠프 근처로 순간이동. 다른 로봇은 건드리지 않음.

    Y 오프셋으로 두 로봇이 겹치지 않게 배치:
      Robot0 → cy + 3m,  Robot1 → cy - 3m

    ★ z값은 고정값 대신 지형 높이맵에서 실제 높이를 조회해서 설정.
      경사지에 스폰해도 땅 아래로 꺼지거나 공중에 뜨지 않음.
    """
    robot      = raw_env.scene["robot"]
    root_state = robot.data.root_state_w.clone()
    y_offset   = 3.0 * (1 - 2 * env_idx)

    tx = cx + 2.0
    ty = cy + y_offset

    # 지형 높이맵에서 실제 z 조회
    hm = raw_env.scene.terrain._terrainManager._heightmap_manager
    target_xy = torch.tensor([[tx, ty]], dtype=torch.float32, device=raw_env.device)
    terrain_z = float(hm.get_height_at(target_xy)[0])

    root_state[env_idx, 0]    = tx
    root_state[env_idx, 1]    = ty
    root_state[env_idx, 2]    = terrain_z + 0.5  # 지형 위 0.5m
    root_state[env_idx, 7:13] = 0.0               # 속도 초기화
    robot.write_root_state_to_sim(root_state)