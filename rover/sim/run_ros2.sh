#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 필수 환경변수 (없으면 종료)
: "${ISAACLAB_DIR:?Set ISAACLAB_DIR (e.g. export ISAACLAB_DIR=\$HOME/IsaacLab)}"
: "${ROS2_BRIDGE_DIR:?Set ROS2_BRIDGE_DIR (e.g. export ROS2_BRIDGE_DIR=\$HOME/isaacsim/exts/isaacsim.ros2.bridge/humble)}"

# 경로 검증
[[ -x "$ISAACLAB_DIR/isaaclab.sh" ]] || { echo "[ERROR] $ISAACLAB_DIR/isaaclab.sh not found"; exit 1; }
[[ -d "$ROS2_BRIDGE_DIR/rclpy" && -d "$ROS2_BRIDGE_DIR/lib" ]] || { echo "[ERROR] Invalid ROS2_BRIDGE_DIR: $ROS2_BRIDGE_DIR"; exit 1; }

# 1) mission_viz_node (ROS python3.10)
bash -c "source /opt/ros/humble/setup.bash; python3 '$SCRIPT_DIR/scripts/mission_viz_node.py'" &
echo "[VizNode] mission_viz_node 시작"

# 2) RViz2
bash -c "source /opt/ros/humble/setup.bash; rviz2 -d '$SCRIPT_DIR/config/mission_monitor.rviz'" &
echo "[RViz2] 모니터링 창 시작"

# 3) Isaac Sim 전용 env
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ROS2_BRIDGE_DIR/lib"

# /opt/ros + python3.10 PYTHONPATH 제거
export PYTHONPATH=$(echo "${PYTHONPATH:-}" | tr ':' '\n' \
  | grep -v "python3\.10" \
  | grep -v "/opt/ros" \
  | tr '\n' ':' \
  | sed 's/:$//')

# 03_eval_ros2.py에서 쓰도록 전달
export ROS2_RCLPY_DIR="$ROS2_BRIDGE_DIR/rclpy"
export ROS2_LIB_DIR="$ROS2_BRIDGE_DIR/lib"

# 4) Isaac Sim 실행
cd "$ISAACLAB_DIR"
./isaaclab.sh -p "$SCRIPT_DIR/scripts/03_eval_ros2.py" \
  --task AAURoverEnv-v0 \
  "$@"
