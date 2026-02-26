import argparse
import math
import random
import sys

import gymnasium as gym
import torch
from isaaclab.app import AppLauncher
import firebase_admin
from firebase_admin import credentials, db

# ── argparse ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser("RLRoverLab Eval - Dual Robot Mineral Collection")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--task", type=str, default="AAURoverEnv-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--agent", type=str, default="PPO")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--dataset_dir", type=str, default="./datasets")
parser.add_argument("--dataset_name", type=str, default=None)
parser.add_argument("--dataset_type", type=str, default="RL", choices=["IL", "RL"])

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports (앱 실행 후) ───────────────────────────────────────────────────────
from isaaclab.envs import ManagerBasedRLEnv          # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper       # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg        # noqa: E402
from skrl.agents.torch.base import Agent              # noqa: E402
from skrl.utils import set_seed                       # noqa: E402

import omni.usd                                       # noqa: E402
import omni.kit.commands                              # noqa: E402
from pxr import UsdGeom, UsdShade, Gf, Sdf           # noqa: E402

import rover_envs                                     # noqa: E402
import rover_envs.envs.navigation.robots              # noqa: E402
from rover_envs.envs.navigation.mdp.observations import get_robot_world_pos  # noqa: E402
from rover_envs.learning.agents import create_agent   # noqa: E402
from rover_envs.utils.config import parse_skrl_cfg    # noqa: E402
from rover_envs.utils.logging_utils import (          # noqa: E402
    configure_datarecorder, log_setup, video_record,
)


# ─────────────────────────────────────────────────────────────────────────────
# 미션 설정
# ─────────────────────────────────────────────────────────────────────────────

NUM_ROBOTS          = 2      # 로봇 대수
MINERALS_PER_ROBOT  = 5      # 로봇 1대당 수집할 광물 수 (5개로 설정됨)
MINERAL_COLLECT_RADIUS = 1.0 # 광물 수집 판정 반경 (m)
BASEMENT_RADIUS     = 2.0    # 베이스캠프 복귀 판정 반경 (m)

# 로봇별 마커 색상: 로봇0=빨강, 로봇1=파랑
MINERAL_COLORS = [
    (0.9, 0.1, 0.05),
    (0.05, 0.3, 0.95),
]
MINERAL_COLLECTED_COLOR = (0.3, 0.3, 0.3)


# ─────────────────────────────────────────────────────────────────────────────
# USD 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _apply_color(stage, prim_path: str, color: tuple):
    mat_path = prim_path + "/Material"
    material = UsdShade.Material.Define(stage, mat_path)
    shader   = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(0.3)
    shader.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(0.8)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(prim_path)).Bind(material)


def _safe_xform(stage, prim_path: str, translate: tuple, scale: tuple):
    prim = stage.GetPrimAtPath(prim_path)
    xf   = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()

    t_attr = prim.GetAttribute("xformOp:translate")
    if t_attr.IsValid():
        t_attr.Set(Gf.Vec3d(*translate)); t_op = UsdGeom.XformOp(t_attr)
    else:
        t_op = xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        t_op.Set(Gf.Vec3d(*translate))

    s_attr = prim.GetAttribute("xformOp:scale")
    if s_attr.IsValid():
        s_attr.Set(Gf.Vec3d(*scale)); s_op = UsdGeom.XformOp(s_attr)
    else:
        s_op = xf.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble)
        s_op.Set(Gf.Vec3d(*scale))

    xf.SetXformOpOrder([t_op, s_op])


def spawn_basecamp_marker(stage, cx: float, cy: float):
    omni.kit.commands.execute("CreatePrimWithDefaultXform", prim_type="Xform",     prim_path="/World/Mission")
    omni.kit.commands.execute("CreatePrimWithDefaultXform", prim_type="Cylinder",  prim_path="/World/Mission/Basecamp")
    _safe_xform(stage, "/World/Mission/Basecamp",
                translate=(cx, cy, -15), scale=(0.01, 0.01, 0.01))
    _apply_color(stage, "/World/Mission/Basecamp", (0.05, 0.9, 0.15))
    print(f"[Mission] 베이스캠프 마커 생성: ({cx:.1f}, {cy:.1f})")


# ─────────────────────────────────────────────────────────────────────────────
# 베이스캠프 중심
# ─────────────────────────────────────────────────────────────────────────────

def get_basecamp_center(raw_env: ManagerBasedRLEnv):
    terrain = raw_env.scene.terrain
    hm = terrain._terrainManager._heightmap_manager
    return (hm.min_x + hm.max_x) / 2.0, (hm.min_y + hm.max_y) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# ★ 추가됨: 카메라 및 멀티 뷰포트 설정
# ─────────────────────────────────────────────────────────────────────────────

def attach_custom_camera(stage, env_idx: int, view_type: str):
    """로봇의 몸체에 카메라를 달아줍니다."""
    cam_path = f"/World/envs/env_{env_idx}/Robot/Body/Camera_{view_type}"
    omni.kit.commands.execute("CreatePrimWithDefaultXform", prim_type="Camera", prim_path=cam_path)
    
    cam_prim = stage.GetPrimAtPath(cam_path)
    if not cam_prim.IsValid(): return None

    # 시점 분기: 1P(1인칭), 3P(3인칭)
    if view_type == "1P":
        translate, rotate, focal = (2.5, 0.0, 0.4), (90.0, 0.0, 90.0), 15.0
    else:
        translate, rotate, focal = (-2.5, 0.0, 1.2), (90.0, 0.0, 90.0), 24.0

    xf = UsdGeom.Xformable(cam_prim)
    xf.ClearXformOpOrder()
    
    t_op = xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    t_op.Set(Gf.Vec3d(*translate))
    r_op = xf.AddRotateXYZOp(precision=UsdGeom.XformOp.PrecisionDouble)
    r_op.Set(Gf.Vec3d(*rotate))
    xf.SetXformOpOrder([t_op, r_op])

    cam_geom = UsdGeom.Camera(cam_prim)
    cam_geom.GetHorizontalApertureAttr().Set(36.0)
    cam_geom.GetFocalLengthAttr().Set(focal)
    return cam_path

def setup_dual_viewports(stage):
    """팝업 창을 2개 띄워서 각각 로봇 0, 로봇 1의 시점을 연결합니다."""
    cam0 = attach_custom_camera(stage, env_idx=0, view_type="1P")
    cam1 = attach_custom_camera(stage, env_idx=1, view_type="1P")

    try:
        import omni.kit.viewport.utility as vp_util
        # 메인 뷰포트는 전체 관전 모드로 둠
        main_vp = vp_util.get_active_viewport()
        if main_vp:
            main_vp.set_active_camera("/OmniverseKit_Persp")

        # 로봇 0 뷰포트 (1인칭)
        vp0 = vp_util.create_viewport_window("Robot 0 (1인칭 시점)", width=500, height=300)
        if vp0 and cam0: vp0.viewport_api.set_active_camera(cam0)

        # 로봇 1 뷰포트 (3인칭)
        vp1 = vp_util.create_viewport_window("Robot 1 (3인칭 시점)", width=500, height=300)
        if vp1 and cam1: vp1.viewport_api.set_active_camera(cam1)
        print("[Camera] 멀티 뷰포트 생성 완료!")
    except Exception as e:
        print(f"[Camera] 멀티 뷰포트 생성 실패 (무시 가능): {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Command 역변환
# ─────────────────────────────────────────────────────────────────────────────

def get_target_world_pos(raw_env: ManagerBasedRLEnv) -> torch.Tensor:
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


# ─────────────────────────────────────────────────────────────────────────────
# env_idx별 command 리샘플
# ─────────────────────────────────────────────────────────────────────────────

def resample_target(raw_env: ManagerBasedRLEnv, env_idx: int):
    try:
        env_ids = torch.tensor([env_idx], device=raw_env.device)
        raw_env.command_manager._terms["target_pose"].reset(env_ids=env_ids)
        print(f"[Robot{env_idx}] 다음 광물 위치로 커맨드 리샘플")
    except Exception as e:
        print(f"[Robot{env_idx}] 커맨드 리샘플 실패: {e}")

def freeze_resample_timer(raw_env: ManagerBasedRLEnv, env_idx: int):
    try:
        term = raw_env.command_manager._terms["target_pose"]
        attr = "time_left" if hasattr(term, "time_left") else "_time_left"
        getattr(term, attr)[env_idx] = 999999.0
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# ★ 수정됨: 베이스 복귀용 커맨드 강제 주입
# ─────────────────────────────────────────────────────────────────────────────

def set_command_to_basecamp(raw_env: ManagerBasedRLEnv, env_idx: int, cx: float, cy: float, device):
    term = raw_env.command_manager._terms["target_pose"]
    
    # ★ 중요: 환경이 인식하는 절대(World) 목표를 베이스캠프(cx, cy)로 강제 지정합니다.
    if hasattr(term, "pos_command_w"):
        term.pos_command_w[env_idx, 0] = cx
        term.pos_command_w[env_idx, 1] = cy

    # 아래는 현재 프레임의 상대 커맨드 버퍼 갱신
    robot = raw_env.scene["robot"]
    pos_w  = robot.data.root_pos_w[env_idx:env_idx+1, :2]
    quat_w = robot.data.root_quat_w[env_idx:env_idx+1, :]

    target = torch.tensor([[cx, cy]], dtype=torch.float32, device=device)
    delta  = target - pos_w

    w  = quat_w[:,0]; xq = quat_w[:,1]; yq = quat_w[:,2]; zq = quat_w[:,3]
    yaw = torch.atan2(2.0*(w*zq + xq*yq), 1.0 - 2.0*(yq*yq + zq*zq))
    cos_yaw = torch.cos(-yaw); sin_yaw = torch.sin(-yaw)
    rel_x = cos_yaw * delta[:,0] - sin_yaw * delta[:,1]
    rel_y = sin_yaw * delta[:,0] + cos_yaw * delta[:,1]
    angle = torch.atan2(rel_y, rel_x)

    cmd = raw_env.command_manager.get_command("target_pose")
    cmd[env_idx, 0] = rel_x[0]
    cmd[env_idx, 1] = rel_y[0]
    cmd[env_idx, 2] = 0.0
    cmd[env_idx, 3] = angle[0]

    freeze_resample_timer(raw_env, env_idx)


# ─────────────────────────────────────────────────────────────────────────────
# 로봇 텔레포트
# ─────────────────────────────────────────────────────────────────────────────

def teleport_to_basecamp(raw_env: ManagerBasedRLEnv, env_idx: int, cx: float, cy: float):
    robot = raw_env.scene["robot"]
    root_state = robot.data.root_state_w.clone()
    y_offset = 3.0 * (1 - 2 * env_idx)   # Robot0=+3m, Robot1=-3m

    tx = cx + 2.0
    ty = cy + y_offset

    # 목표 XY의 실제 지형 높이 조회 (reset_root_state_rover와 동일한 방식)
    hm = raw_env.scene.terrain._terrainManager._heightmap_manager
    target_xy = torch.tensor([[tx, ty]], dtype=torch.float32, device=raw_env.device)
    terrain_z = float(hm.get_height_at(target_xy)[0])

    root_state[env_idx, 0] = tx
    root_state[env_idx, 1] = ty
    root_state[env_idx, 2] = terrain_z + 2  # 지형 위 0.5m (reset과 동일한 z_offset)
    root_state[env_idx, 7:13] = 0.0
    robot.write_root_state_to_sim(root_state)

# ─────────────────────────────────────────────────────────────────────────────
# 미션 루프 
# ─────────────────────────────────────────────────────────────────────────────

def run_mission(env, agent, simulation_app, cx: float, cy: float):
    device     = torch.device("cuda:0")
    base_pos_t = torch.tensor([cx, cy], dtype=torch.float32, device=device)

    def new_state(robot_idx: int, round_num: int = 1) -> dict:
        return {
            "phase":     "collect",   # collect | return
            "collected": 0,
            "round":     round_num,
        }

    states, _ = env.reset()
    agent.set_running_mode("eval")
    raw_env: ManagerBasedRLEnv = env.unwrapped

    for i in range(NUM_ROBOTS):
        teleport_to_basecamp(raw_env, i, cx, cy)
        resample_target(raw_env, i)   

    robot_states   = {i: new_state(i) for i in range(NUM_ROBOTS)}
    pending_teleport: set = set()

    step = 0
    print("\n" + "="*60)
    print(f"🚀 듀얼 로봇 미션 시작! 각자 광물 {MINERALS_PER_ROBOT}개씩 독립 수집")
    print("="*60 + "\n")

    while simulation_app.is_running():
        raw_env = env.unwrapped

        if pending_teleport:
            for i in list(pending_teleport):
                teleport_to_basecamp(raw_env, i, cx, cy)
                resample_target(raw_env, i)   
            pending_teleport.clear()

        # 베이스 복귀 중인 로봇만 커맨드 강제 주입
        for i in range(NUM_ROBOTS):
            if robot_states[i]["phase"] == "return":
                set_command_to_basecamp(raw_env, i, cx, cy, device)

        # ── 에이전트 행동 (속도 증폭 2.0배) ──────────────────────────
        with torch.no_grad():
            actions = agent.act(states, timestep=step, timesteps=999999)[0]
            # 로봇이 언덕에서 멈추지 않도록 엑셀 출력을 2배로 극대화
            actions[:, 0] = torch.clamp(actions[:, 0] * 2.0, -1.0, 1.0)

        states, rewards, terminated, truncated, infos = env.step(actions)
        all_robot_xy = get_robot_world_pos(raw_env)   
        all_target_xy = get_target_world_pos(raw_env) 

        if step % 100 == 0:
            for i in range(NUM_ROBOTS):
                st = robot_states[i]
                rx = all_robot_xy[i]
                if st["phase"] == "collect":
                    dist = torch.norm(rx - all_target_xy[i]).item()
                    print(f"[Step {step:6d}] Robot{i} R{st['round']} | 💎 [{st['collected']}/{MINERALS_PER_ROBOT}] | 광물까지 {dist:.1f}m")
                else:
                    dist = torch.norm(rx - base_pos_t).item()
                    print(f"[Step {step:6d}] Robot{i} R{st['round']} | 🏠 복귀 중 | 베이스까지 {dist:.1f}m")

        # ── 각 로봇 독립 판정 ─────────────────────────────────────────
        for i in range(NUM_ROBOTS):
            st = robot_states[i]
            rx = all_robot_xy[i]

            if st["phase"] == "collect":
                dist = torch.norm(rx - all_target_xy[i]).item()
                if dist < MINERAL_COLLECT_RADIUS:
                    st["collected"] += 1
                    print(f"\n  Robot{i} 💎 광물 #{st['collected']} 수집! ({dist:.2f}m) [{st['collected']}/{MINERALS_PER_ROBOT}]")

                    if st["collected"] >= MINERALS_PER_ROBOT:
                        st["phase"] = "return"
                        print(f"  Robot{i} 🏠 전부 수집! 베이스캠프 복귀 시작\n")
                    else:
                        resample_target(raw_env, i)

            elif st["phase"] == "return":
                dist = torch.norm(rx - base_pos_t).item()
                if dist < BASEMENT_RADIUS:
                    next_round = st["round"] + 1
                    print(f"\n  Robot{i} ✅ 라운드 {st['round']} 완료! → 라운드 {next_round} 즉시 시작\n")
                    robot_states[i] = new_state(i, next_round)
                    pending_teleport.add(i)

        if terminated.any() or truncated.any():
            for i in range(NUM_ROBOTS):
                if terminated[i] or truncated[i]:
                    reason = "충돌/전복" if terminated[i] else "타임아웃"
                    print(f"  [Step {step}] Robot{i} 에피소드 종료({reason}) → 재위치 예약 (수집 유지: {robot_states[i]['collected']}/{MINERALS_PER_ROBOT})")
                    pending_teleport.add(i)

        # ── 로봇별 현재 거리 계산 (DB 업로드용) ──────────────────────────
        dist_per_robot = {}
        for i in range(NUM_ROBOTS):
            rx = all_robot_xy[i]
            if robot_states[i]["phase"] == "collect":
                dist_per_robot[i] = round(torch.norm(rx - all_target_xy[i]).item(), 2)
            else:
                dist_per_robot[i] = round(torch.norm(rx - base_pos_t).item(), 2)

        # ── Firebase DB 업로드 (100 step마다) ─────────────────────────────
        if step % 100 == 0:
            ref = db.reference('robot_status')
            ref.set({
                'step': step,
                'robot0': {
                    'phase':     robot_states[0]['phase'],
                    'collected': robot_states[0]['collected'],
                    'round':     robot_states[0]['round'],
                    'distance':  dist_per_robot[0],
                    'velocity':  round(float(actions[0, 0]), 3),
                },
                'robot1': {
                    'phase':     robot_states[1]['phase'],
                    'collected': robot_states[1]['collected'],
                    'round':     robot_states[1]['round'],
                    'distance':  dist_per_robot[1],
                    'velocity':  round(float(actions[1, 0]), 3),
                },
                'total_collected': robot_states[0]['collected'] + robot_states[1]['collected'],
            })
        step += 1

# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args_cli_seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 100000000)
    args_cli.num_envs = NUM_ROBOTS

    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cuda:0" if not args_cli.cpu else "cpu",
        num_envs=args_cli.num_envs,
    )

    env_cfg.episode_length_s = 50000.0
    if hasattr(env_cfg, "terminations"):
        if hasattr(env_cfg.terminations, "is_success"):
            env_cfg.terminations.is_success = None
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None

    if args_cli.dataset_name is not None:
        env_cfg = configure_datarecorder(
            env_cfg, args_cli.dataset_dir,
            args_cli.dataset_name, args_cli.dataset_type,
        )

    experiment_cfg_file = gym.spec(args_cli.task).kwargs.get("skrl_cfgs")[args_cli.agent.upper()]
    experiment_cfg = parse_skrl_cfg(experiment_cfg_file)
    log_dir = log_setup(experiment_cfg, env_cfg, args_cli.agent)

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, viewport=args_cli.video, render_mode=render_mode)
    env = video_record(env, log_dir, args_cli.video, args_cli.video_length, args_cli.video_interval)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])

    agent: Agent = create_agent(args_cli.agent, env, experiment_cfg)
    agent_policy_path = gym.spec(args_cli.task).kwargs.pop("best_model_path")
    agent.load(agent_policy_path)
    print(f"[Agent] 체크포인트 로드: {agent_policy_path}")

    stage = omni.usd.get_context().get_stage()
    raw_env: ManagerBasedRLEnv = env.unwrapped
    cx, cy = get_basecamp_center(raw_env)
    print(f"[Mission] 베이스캠프 중심: ({cx:.2f}, {cy:.2f})")

    spawn_basecamp_marker(stage, cx, cy)

    # ★ 추가됨: 메인 화면 외에 카메라 뷰포트 2개 띄우기
    setup_dual_viewports(stage)

    # Firebase 초기화 (한 번만) — reat_from_firebase.py와 동일한 키 파일 사용
    SERVICE_ACCOUNT_KEY_PATH = "./examples/03_inference/rokey-93910-firebase-adminsdk-fbsvc-a6e6f601c3.json"
    DATABASE_URL = "https://rokey-93910-default-rtdb.asia-southeast1.firebasedatabase.app"
    try:
        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
        firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})
    except ValueError:
        print("[Firebase] 앱이 이미 초기화되어 있습니다. 기존 앱을 사용합니다.")

    run_mission(env, agent, simulation_app, cx, cy)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
# if __name__ == "__main__":
#     main()
#--------------------------------------기존코드
# import argparse
# import math
# import os
# import random
# import sys
# from datetime import datetime

# import gymnasium as gym
# from isaaclab.app import AppLauncher

# # add argparse arguments
# parser = argparse.ArgumentParser("Welcome to Isaac Lab: Omniverse Robotics Environments!")
# parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
# parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
# parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
# parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# parser.add_argument("--task", type=str, default="AAURoverEnv-v0", help="Name of the task.")
# parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# parser.add_argument("--agent", type=str, default="PPO", help="Name of the agent.")
# parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
# parser.add_argument("--dataset_dir", type=str, default="./datasets", help="Path to the dataset directory.")
# parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset.")
# parser.add_argument("--dataset_type", type=str, default="RL", choices=["IL", "RL"], help="Type of dataset to use. Options: IL or RL.")

# AppLauncher.add_app_launcher_args(parser)
# args_cli, hydra_args = parser.parse_known_args()

# # always enable cameras to record video
# if args_cli.video:
#     args_cli.enable_cameras = True

# # clear out sys.argv for Hydra
# sys.argv = [sys.argv[0]] + hydra_args

# # app_launcher = AppLauncher(launcher_args=args_cli, experience=app_experience)

# app_launcher = AppLauncher(args_cli)

# from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

# simulation_app = app_launcher.app

# from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
# from isaaclab.managers import DatasetExportMode  # noqa: E402
# from isaaclab.utils.dict import print_dict  # noqa: E402
# from isaaclab.utils.io import dump_pickle, dump_yaml  # noqa: E402
# from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
# from skrl.agents.torch.base import Agent  # noqa: E402
# from skrl.trainers.torch import SequentialTrainer  # noqa: E402
# from skrl.utils import set_seed  # noqa: E402, F401

# import rover_envs  # noqa: E402
# import rover_envs.envs.navigation.robots  # noqa: E402, F401
# # Import the general agent factory
# from rover_envs.learning.agents import create_agent  # noqa: E402
# # Import to ensure navigation agents are registered
# #import rover_envs.envs.navigation.learning.skrl.agents  # noqa: E402, F401
# from rover_envs.utils.config import parse_skrl_cfg  # noqa: E402
# from rover_envs.utils.logging_utils import configure_datarecorder, log_setup, video_record  # noqa: E402


# def main():
#     args_cli_seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 100000000)
#     env_cfg = parse_env_cfg(args_cli.task, device="cuda:0" if not args_cli.cpu else "cpu", num_envs=args_cli.num_envs)

#     if args_cli.dataset_name is not None:
#         env_cfg = configure_datarecorder(env_cfg, args_cli.dataset_dir, args_cli.dataset_name, args_cli.dataset_type)


#     # key = agent name, value = path to config file
#     experiment_cfg_file = gym.spec(args_cli.task).kwargs.get("skrl_cfgs")[args_cli.agent.upper()]
#     experiment_cfg = parse_skrl_cfg(experiment_cfg_file)

#     log_dir = log_setup(experiment_cfg, env_cfg, args_cli.agent)

#     # Create the environment
#     render_mode = "rgb_array" if args_cli.video else None
#     env = gym.make(args_cli.task, cfg=env_cfg, viewport=args_cli.video, render_mode=render_mode)
#     # Check if video recording is enabled
#     env = video_record(env, log_dir, args_cli.video, args_cli.video_length, args_cli.video_interval)
#     # Wrap the environment
#     env = SkrlVecEnvWrapper(env, ml_framework="torch")
#     set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])

#     # Get the observation and action spaces
#     num_obs = env.observation_manager.group_obs_dim["policy"][0]
#     num_actions = env.action_manager.action_term_dim[0]
#     #observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
#     #action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actions,))

#     trainer_cfg = experiment_cfg["trainer"]
#     trainer_cfg["timesteps"] = 1000000

#     agent: Agent = create_agent(args_cli.agent, env, experiment_cfg)

#     # Get the checkpoint path from the experiment configuration
#     print(f'args_cli.task: {args_cli.task}')
#     agent_policy_path = gym.spec(args_cli.task).kwargs.pop("best_model_path")

#     agent.load(agent_policy_path)
#     trainer_cfg = experiment_cfg["trainer"]
#     print(trainer_cfg)

#     trainer = SequentialTrainer(cfg=trainer_cfg, agents=agent, env=env)
#     trainer.eval()

#     env.close()
#     simulation_app.close()


# if __name__ == "__main__":
#     main()
