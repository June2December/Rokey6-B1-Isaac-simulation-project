#!/bin/bash
# run_ros2.sh — 전체 자동 실행 (Isaac Sim + viz_node + RViz2)
# 사용법: ./run_ros2.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ── 1. mission_viz_node (ROS python3.10 환경) ─────────────────────────────────
bash -c "
  source /opt/ros/humble/setup.bash
  python3 '$SCRIPT_DIR/scripts/mission_viz_node.py'
" &
echo "[VizNode] mission_viz_node 시작"

# ── 2. RViz2 (ROS 환경, Isaac Sim LD_LIBRARY_PATH 와 분리) ────────────────────
bash -c "
  source /opt/ros/humble/setup.bash
  rviz2 -d '$SCRIPT_DIR/config/mission_monitor.rviz'
" &
echo "[RViz2] 모니터링 창 시작"

# ── 3. Isaac Sim 전용 환경변수 설정 ──────────────────────────────────────────
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rokey/isaacsim/exts/isaacsim.ros2.bridge/humble/lib
export PYTHONNOUSERSITE=1

# python3.10 경로 제거 (source /opt/ros 가 심어놓은 것 방지)
export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' \
  | grep -v "python3\.10" \
  | grep -v "/opt/ros" \
  | tr '\n' ':' \
  | sed 's/:$//')

# ── 4. Isaac Sim 실행 ─────────────────────────────────────────────────────────
cd ~/IsaacLab
./isaaclab.sh -p "$SCRIPT_DIR/scripts/03_eval_ros2.py" \
    --task AAURoverEnv-v0 \
    "$@"
