from __future__ import annotations

import argparse
import random
import sys
import threading
import time
import json
import os

import gymnasium as gym
import torch
from isaaclab.app import AppLauncher

# ── argparse ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser("Dual Robot Mission + ROS2")
parser.add_argument("--video",          action="store_true", default=False)
parser.add_argument("--video_length",   type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs",       type=int, default=2)
parser.add_argument("--task",           type=str, default="AAURoverEnv-v0")
parser.add_argument("--seed",           type=int, default=None)
parser.add_argument("--agent",          type=str, default="TRPO")
parser.add_argument("--checkpoint",     type=str, default=None)
parser.add_argument("--dataset_dir",    type=str, default="./datasets")
parser.add_argument("--dataset_name",   type=str, default=None)
parser.add_argument("--dataset_type",   type=str, default="RL", choices=["IL", "RL"])

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Lab imports (앱 실행 후) ────────────────────────────────────────────
from isaaclab.envs import ManagerBasedRLEnv          # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper       # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg        # noqa: E402
from skrl.agents.torch.base import Agent              # noqa: E402
from skrl.utils import set_seed                       # noqa: E402
import omni.usd                                       # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rover_envs                                     # noqa: E402
import rover_envs.envs.navigation.robots.aau_rover
from rover_envs.envs.navigation.mdp.observations import get_robot_world_pos  # noqa: E402
from rover_envs.learning.agents import create_agent   # noqa: E402
from rover_envs.utils.config import parse_skrl_cfg    # noqa: E402
from rover_envs.utils.logging_utils import (
    configure_datarecorder, log_setup, video_record,
)

# ── ROS2 imports ──────────────────────────────────────────────────────────────
# 1. extension 먼저 활성화 → Isaac Sim 내부 rclpy 로드
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
enable_extension("isaacsim.ros2.bridge")

# 2. Isaac Sim python3.11 rclpy 경로 우선 적용
#    source /opt/ros/humble 이 python3.10 경로를 심어놓을 수 있으므로 제거 후 삽입
ROS2_RCLPY_DIR = os.environ["ROS2_RCLPY_DIR"]
ROS2_LIB_DIR   = os.environ["ROS2_LIB_DIR"]
sys.path = [p for p in sys.path if "python3.10" not in p and "/opt/ros" not in p]
sys.path.insert(0, ROS2_LIB_DIR)
sys.path.insert(0, ROS2_RCLPY_DIR)

# 3. visualization_msgs 는 Isaac Sim 내부에 없으므로 import 하지 않음
#    Marker 생성은 ROS 환경의 mission_viz_node.py 가 담당
import rclpy                                          # noqa: E402
from rclpy.executors import MultiThreadedExecutor     # noqa: E402
from rclpy.node import Node                           # noqa: E402
from geometry_msgs.msg import PoseStamped             # noqa: E402
from std_msgs.msg import String                       # noqa: E402
from sensor_msgs.msg import Image                     # noqa: E402

# ── 미션 유틸리티 ──────────────────────────────────────────────────────────────
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from firebase_db import init_firebase, upload_robot_status  # noqa: E402
from mission import (                                 # noqa: E402
    spawn_basecamp_marker,
    get_basecamp_center,
    setup_dual_viewports,
    get_target_world_pos,
    resample_target,
    set_command_to_basecamp,
    teleport_to_basecamp,
)

# ─────────────────────────────────────────────────────────────────────────────
# 미션 설정
# ─────────────────────────────────────────────────────────────────────────────
NUM_ROBOTS             = 2
MINERALS_PER_ROBOT     = 5
MINERAL_COLLECT_RADIUS = 0.5
BASEMENT_RADIUS        = 2.0

# ─────────────────────────────────────────────────────────────────────────────
# ROS2 퍼블리셔 노드 (Isaac Sim 측 — 데이터만 퍼블리시)
# ─────────────────────────────────────────────────────────────────────────────
class MissionPublisher(Node):
    """
    Isaac Sim 측 퍼블리셔. visualization_msgs 없이 데이터만 내보냄.
    Marker 생성은 ROS 환경의 mission_viz_node.py 가 담당.

    퍼블리시 토픽:
      /robot{i}/pose             PoseStamped
      /mission/status            String JSON (phase/collected/round/distance/x/y)
      /robot{i}/camera/image_raw Image
    """

    def __init__(self):
        super().__init__("mission_publisher")

        self.pose_pubs = [
            self.create_publisher(PoseStamped, f"/robot{i}/pose", 10)
            for i in range(NUM_ROBOTS)
        ]
        self.status_pub = self.create_publisher(String, "/mission/status", 10)
        self.image_pubs = [
            self.create_publisher(Image, f"/robot{i}/camera/image_raw", 1)
            for i in range(NUM_ROBOTS)
        ]

        self.robot_positions = [None] * NUM_ROBOTS
        self.robot_states    = {}
        self.robot_images    = [None] * NUM_ROBOTS
        self._lock = threading.Lock()

        self.create_timer(0.1, self._publish)  # 10Hz

    def update(self, robot_positions, robot_states, robot_images=None):
        with self._lock:
            self.robot_positions = [
                p.clone() if p is not None else None for p in robot_positions
            ]
            self.robot_states = {k: dict(v) for k, v in robot_states.items()}
            if robot_images is not None:
                self.robot_images = robot_images

    def _publish(self):
        with self._lock:
            now = self.get_clock().now().to_msg()

            # 위치
            for i, pos in enumerate(self.robot_positions):
                if pos is None:
                    continue
                msg = PoseStamped()
                msg.header.stamp       = now
                msg.header.frame_id    = "world"
                msg.pose.position.x    = float(pos[0])
                msg.pose.position.y    = float(pos[1])
                msg.pose.position.z    = 0.0
                msg.pose.orientation.w = 1.0
                self.pose_pubs[i].publish(msg)

            # 카메라
            for i, img in enumerate(self.robot_images):
                if img is None:
                    continue
                try:
                    msg = Image()
                    msg.header.stamp    = now
                    msg.header.frame_id = f"robot{i}_camera"
                    msg.height   = img.shape[0]
                    msg.width    = img.shape[1]
                    msg.encoding = "rgba8"
                    msg.step     = msg.width * 4
                    msg.data     = img.tobytes()
                    self.image_pubs[i].publish(msg)
                except Exception:
                    pass

            # 상태 JSON (distance/x/y 포함 — viz_node 가 Marker 텍스트로 사용)
            status = {
                str(i): {
                    "phase":     self.robot_states.get(i, {}).get("phase",     "unknown"),
                    "collected": self.robot_states.get(i, {}).get("collected", 0),
                    "round":     self.robot_states.get(i, {}).get("round",     1),
                    "distance":  self.robot_states.get(i, {}).get("distance",  None),
                    "x":         self.robot_states.get(i, {}).get("x",         None),
                    "y":         self.robot_states.get(i, {}).get("y",         None),
                }
                for i in range(NUM_ROBOTS)
            }
            msg = String()
            msg.data = json.dumps(status)
            self.status_pub.publish(msg)

# ─────────────────────────────────────────────────────────────────────────────
# 미션 루프
# ─────────────────────────────────────────────────────────────────────────────
def run_mission(env, agent, simulation_app, cx, cy, ros_node: MissionPublisher):
    device     = torch.device("cuda:0")
    base_pos_t = torch.tensor([cx, cy], dtype=torch.float32, device=device)

    def new_state(round_num: int = 1) -> dict:
        return {"phase": "collect", "collected": 0, "round": round_num,
                "distance": None, "x": None, "y": None}

    states, _ = env.reset()
    agent.set_running_mode("eval")
    raw_env: ManagerBasedRLEnv = env.unwrapped

    for i in range(NUM_ROBOTS):
        teleport_to_basecamp(raw_env, i, cx, cy)
        resample_target(raw_env, i)

    # 카메라 초기화
    try:
        from isaacsim.sensors.camera import Camera
        cameras = []
        for i in range(NUM_ROBOTS):
            cam = Camera(
                prim_path=f"/World/envs/env_{i}/Robot/Body/Camera_1P",
                resolution=(320, 240),
            )
            cam.initialize()
            cameras.append(cam)
        print("[Camera] 초기화 완료")
    except Exception as e:
        print(f"[Camera] 초기화 실패 (무시): {e}")
        cameras = []

    robot_states     = {i: new_state() for i in range(NUM_ROBOTS)}
    pending_teleport: set = set()

    step       = 0
    start_time = time.time()
    print("\n" + "="*60)
    print(f"🚀 듀얼 로봇 미션 시작! 각자 광물 {MINERALS_PER_ROBOT}개씩 독립 수집")
    print(f"   베이스캠프: ({cx:.1f}, {cy:.1f})")
    print("="*60 + "\n")

    while simulation_app.is_running():
        raw_env = env.unwrapped

        if pending_teleport:
            for i in list(pending_teleport):
                teleport_to_basecamp(raw_env, i, cx, cy)
                resample_target(raw_env, i)
            pending_teleport.clear()

        with torch.no_grad():
            actions = agent.act(states, timestep=step, timesteps=999999)[0]
            actions[:, 0] = torch.clamp(actions[:, 0] * 2.0, -1.0, 1.0)

        states, rewards, terminated, truncated, infos = env.step(actions)

        for i in range(NUM_ROBOTS):
            if robot_states[i]["phase"] == "return":
                set_command_to_basecamp(raw_env, i, cx, cy, device)

        all_robot_xy  = get_robot_world_pos(raw_env)
        all_target_xy = get_target_world_pos(raw_env)

        # 위치 + 남은거리 업데이트
        for i in range(NUM_ROBOTS):
            rx = all_robot_xy[i]
            robot_states[i]["x"] = float(rx[0])
            robot_states[i]["y"] = float(rx[1])
            if robot_states[i]["phase"] == "collect":
                robot_states[i]["distance"] = float(torch.norm(rx - all_target_xy[i]))
            else:
                robot_states[i]["distance"] = float(torch.norm(rx - base_pos_t))

        # 카메라 이미지 캡처 (10스텝마다)
        robot_images = [None] * NUM_ROBOTS
        if cameras and step % 10 == 0:
            for i, cam in enumerate(cameras):
                try:
                    cam_data = cam.get_rgba()
                    if cam_data is not None:
                        robot_images[i] = cam_data
                except Exception:
                    pass

        ros_node.update(
            robot_positions=[all_robot_xy[i] for i in range(NUM_ROBOTS)],
            robot_states=robot_states,
            robot_images=robot_images,
        )

        if step % 100 == 0:
            for i in range(NUM_ROBOTS):
                st   = robot_states[i]
                dist = st.get("distance", 0) or 0
                icon = "💎" if st["phase"] == "collect" else "🏠"
                print(f"[Step {step:6d}] Robot{i} R{st['round']} | "
                      f"{icon} [{st['collected']}/{MINERALS_PER_ROBOT}] | {dist:.1f}m")

        for i in range(NUM_ROBOTS):
            st = robot_states[i]
            rx = all_robot_xy[i]

            if st["phase"] == "collect":
                dist = torch.norm(rx - all_target_xy[i]).item()
                angle = raw_env.command_manager.get_command("target_pose")[i, 3].item()
                if dist < MINERAL_COLLECT_RADIUS and abs(angle) < 0.2:
                    st["collected"] += 1
                    print(f"\n  Robot{i} 💎 광물 #{st['collected']} 수집! "
                          f"({dist:.2f}m) [{st['collected']}/{MINERALS_PER_ROBOT}]")
                    if st["collected"] >= MINERALS_PER_ROBOT:
                        st["phase"] = "return"
                        print(f"  Robot{i} 🏠 전부 수집! 베이스캠프 복귀 시작\n")
                        set_command_to_basecamp(raw_env, i, cx, cy, device)
                    else:
                        resample_target(raw_env, i)

            elif st["phase"] == "return":
                dist = torch.norm(rx - base_pos_t).item()
                if dist < BASEMENT_RADIUS:
                    next_round = st["round"] + 1
                    print(f"\n  Robot{i} ✅ 라운드 {st['round']} 완료! "
                          f"→ 라운드 {next_round} 즉시 시작\n")
                    robot_states[i] = new_state(next_round)
                    pending_teleport.add(i)

        if terminated.any() or truncated.any():
            for i in range(NUM_ROBOTS):
                if terminated[i] or truncated[i]:
                    reason = "충돌/전복" if terminated[i] else "타임아웃"
                    print(f"  [Step {step}] Robot{i} 에피소드 종료({reason}) → 재위치 예약")
                    pending_teleport.add(i)

        if step % 100 == 0:
            upload_robot_status(step, time.time() - start_time, robot_states, actions)

        step += 1

# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args_cli_seed     = args_cli.seed if args_cli.seed is not None else random.randint(0, 100000000)
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
    experiment_cfg      = parse_skrl_cfg(experiment_cfg_file)
    log_dir             = log_setup(experiment_cfg, env_cfg, args_cli.agent)

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, viewport=args_cli.video, render_mode=render_mode)
    env = video_record(env, log_dir, args_cli.video, args_cli.video_length, args_cli.video_interval)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])

    agent: Agent = create_agent(args_cli.agent, env, experiment_cfg)
    agent_policy_path = gym.spec(args_cli.task).kwargs.pop("best_model_path")
    agent.load(agent_policy_path)
    print(f"[Agent] 체크포인트 로드: {agent_policy_path}")

    stage   = omni.usd.get_context().get_stage()
    raw_env: ManagerBasedRLEnv = env.unwrapped
    cx, cy  = get_basecamp_center(raw_env)
    print(f"[Mission] 베이스캠프 중심: ({cx:.2f}, {cy:.2f})")

    spawn_basecamp_marker(stage, cx, cy, visual_radius=10.0)
    setup_dual_viewports(stage)

    init_firebase()

    rclpy.init()
    ros_node = MissionPublisher()
    executor = MultiThreadedExecutor()
    executor.add_node(ros_node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()
    print("[ROS2] 퍼블리셔 스레드 시작")

    try:
        run_mission(env, agent, simulation_app, cx, cy, ros_node)
    finally:
        rclpy.shutdown()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()