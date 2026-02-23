"""
02_eval.py — 로버 광물 수집 미션
디폴트로 2대 작동 중

사용법:
    cd rover/sim
    python scripts/02_dual_mineral_eval.py --task AAURoverEnv-v0

동작:
  로봇 2대가 완전히 독립적으로 광물 N개씩 순차 수집 후 베이스 복귀.
  상대 로봇을 기다리지 않고 즉시 다음 라운드 시작.
"""
import argparse
import random
import sys

import gymnasium as gym
import torch
from isaaclab.app import AppLauncher

# ── argparse ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser("Dual Robot Mineral Collection Mission")
parser.add_argument("--video",          action="store_true", default=False)
parser.add_argument("--video_length",   type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs",       type=int, default=2)
parser.add_argument("--task",           type=str, default="AAURoverEnv-v0")
parser.add_argument("--seed",           type=int, default=None)
parser.add_argument("--agent",          type=str, default="PPO")
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

# ── imports (앱 실행 후) ───────────────────────────────────────────────────────
from isaaclab.envs import ManagerBasedRLEnv          # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper       # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg        # noqa: E402
from skrl.agents.torch.base import Agent              # noqa: E402
from skrl.utils import set_seed                       # noqa: E402
import omni.usd                                       # noqa: E402

import rover_envs                                     # noqa: E402
import rover_envs.envs.navigation.robots              # noqa: E402
from rover_envs.envs.navigation.mdp.observations import get_robot_world_pos  # noqa: E402
from rover_envs.learning.agents import create_agent   # noqa: E402
from rover_envs.utils.config import parse_skrl_cfg    # noqa: E402
from rover_envs.utils.logging_utils import (          # noqa: E402
    configure_datarecorder, log_setup, video_record,
)

# 미션 utils (rover/sim/mission/)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mission import (                                  # noqa: E402
    spawn_basecamp_marker,
    get_basecamp_center,
    setup_dual_viewports,
    get_target_world_pos,
    resample_target,
    set_command_to_basecamp,
    teleport_to_basecamp,
)
# ─────────────────────────────────────────────────────────────────────────────
# 미션 설정 — 여기만 수정
# ─────────────────────────────────────────────────────────────────────────────
NUM_ROBOTS             = 2      # 로봇 대수
MINERALS_PER_ROBOT     = 5      # 로봇 1대당 수집할 광물 수
MINERAL_COLLECT_RADIUS = 0.2    # 광물 수집 판정 반경 (m)
BASEMENT_RADIUS        = 2.0    # 베이스캠프 복귀 판정 반경 (m)
# ─────────────────────────────────────────────────────────────────────────────
# 미션 루프
# ─────────────────────────────────────────────────────────────────────────────
def run_mission(env, agent, simulation_app, cx: float, cy: float):
    device     = torch.device("cuda:0")
    base_pos_t = torch.tensor([cx, cy], dtype=torch.float32, device=device)

    def new_state(round_num: int = 1) -> dict:
        return {"phase": "collect", "collected": 0, "round": round_num}

    states, _ = env.reset()
    agent.set_running_mode("eval")
    raw_env: ManagerBasedRLEnv = env.unwrapped

    for i in range(NUM_ROBOTS):
        teleport_to_basecamp(raw_env, i, cx, cy)
        resample_target(raw_env, i)

    robot_states     = {i: new_state() for i in range(NUM_ROBOTS)}
    pending_teleport: set = set()

    step = 0
    print("\n" + "="*60)
    print(f"🚀 듀얼 로봇 미션 시작! 각자 광물 {MINERALS_PER_ROBOT}개씩 독립 수집")
    print(f"   베이스캠프: ({cx:.1f}, {cy:.1f})")
    print("="*60 + "\n")

    while simulation_app.is_running():
        raw_env = env.unwrapped

        # 텔레포트 큐 처리
        if pending_teleport:
            for i in list(pending_teleport):
                teleport_to_basecamp(raw_env, i, cx, cy)
                resample_target(raw_env, i)
            pending_teleport.clear()

        # 에이전트 행동 (2.0배 증폭)
        with torch.no_grad():
            actions = agent.act(states, timestep=step, timesteps=999999)[0]
            actions[:, 0] = torch.clamp(actions[:, 0] * 2.0, -1.0, 1.0)

        states, rewards, terminated, truncated, infos = env.step(actions)

        # ★ step() 이후 커맨드 주입 — command_manager.compute() 덮어쓰기 방지
        for i in range(NUM_ROBOTS):
            if robot_states[i]["phase"] == "return":
                set_command_to_basecamp(raw_env, i, cx, cy, device)

        all_robot_xy  = get_robot_world_pos(raw_env)
        all_target_xy = get_target_world_pos(raw_env)

        # 100스텝마다 로그
        if step % 100 == 0:
            for i in range(NUM_ROBOTS):
                st = robot_states[i]
                rx = all_robot_xy[i]
                if st["phase"] == "collect":
                    dist = torch.norm(rx - all_target_xy[i]).item()
                    print(f"[Step {step:6d}] Robot{i} R{st['round']} | "
                          f"💎 [{st['collected']}/{MINERALS_PER_ROBOT}] | 광물까지 {dist:.1f}m")
                else:
                    dist = torch.norm(rx - base_pos_t).item()
                    print(f"[Step {step:6d}] Robot{i} R{st['round']} | "
                          f"🏠 복귀 중 | 베이스까지 {dist:.1f}m")

        # 각 로봇 독립 판정
        for i in range(NUM_ROBOTS):
            st = robot_states[i]
            rx = all_robot_xy[i]

            if st["phase"] == "collect":
                dist = torch.norm(rx - all_target_xy[i]).item()
                if dist < MINERAL_COLLECT_RADIUS:
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

        # 충돌/전복 처리
        if terminated.any() or truncated.any():
            for i in range(NUM_ROBOTS):
                if terminated[i] or truncated[i]:
                    reason = "충돌/전복" if terminated[i] else "타임아웃"
                    print(f"  [Step {step}] Robot{i} 에피소드 종료({reason}) → 재위치 예약 "
                          f"(수집 유지: {robot_states[i]['collected']}/{MINERALS_PER_ROBOT})")
                    pending_teleport.add(i)

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

    # 도달 자동종료 완전 차단
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
    run_mission(env, agent, simulation_app, cx, cy)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()