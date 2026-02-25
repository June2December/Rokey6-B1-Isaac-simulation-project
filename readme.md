# README.MD

# 디지털 트윈 기반 서비스 로봇 운영 시스템 구성 프로젝트

- 주요기능
- 시스템 설계, 플로우차트 그림
- 운영체제 환경
- 사용장비 목록
- 의존성(requirements.txt)
- 간단 실행순서(launch 순서, 스크립트)


## 📦 Terrain Assets Setup

Due to GitHub file size limits, terrain assets are not included in the repository.

### 1. Download assets

Download from Releases:

👉 https://github.com/June2December/Rokey6-B1-Isaac-simulation-project/releases

Download:

mars_terrain_assets.zip

---

### 2. Extract

Extract the zip file to:

rover/sim/rover_envs/assets/terrains/mars/

---

### 3. Verify structure

Final structure should look like:
```
assets/
└── terrains/
    └── mars/
        └── terrain1/
            ├── terrain_merged.usd
            ├── terrain_only.usd
            ├── rocks_merged.usd
            └── SubUSDs/
```
🔧 Environment Variables Setup (Required)

run_ros2.sh 실행 전에 아래 환경변수가 설정되어 있어야 합니다.
```
export ISAACLAB_DIR=$HOME/IsaacLab
export ROS2_BRIDGE_DIR=$HOME/isaacsim/exts/isaacsim.ros2.bridge/humble
```
매번 설정하기 번거롭다면 .bashrc에 추가합니다.
```
echo 'export ISAACLAB_DIR=$HOME/IsaacLab' >> ~/.bashrc
echo 'export ROS2_BRIDGE_DIR=$HOME/isaacsim/exts/isaacsim.ros2.bridge/humble' >> ~/.bashrc
source ~/.bashrc
```
---

### 4. Run simulation
```
# Executing simulation with .sh
~/Rokey6-B1-Isaac-simulation-project/rover/sim/run_ros2.sh

# Executing monitoring node, after ros_set
source /opt/ros/humble/setup.bash
python3 ~/Rokey6-B1-Isaac-simulation-project/rover/sim/scripts/mission_monitor.py
```
