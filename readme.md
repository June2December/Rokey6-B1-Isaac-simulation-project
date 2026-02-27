# README.MD
# 🚀 화성 광물 수집 자율주행 6륜 Rover (PPO + Ackermann)

1. Issac sim 기반 자율주행 Rover 시스템입니다.
2. PPO 출력[조향, 속도] 을 Ackermann 알고리즘을 통해 6륜 Rover가 주행합니다.
3. PPO 는 Navigation, Alignment, Stability 학습은 위해 8개 reward 를 바탕으로 학습했습니다.

---

## 📌 주요 기능(Key Features)

---

### 1️⃣ 강화학습 기반 자율주행

- PPO(RL) 기반  로버 제어
- Ackermann 조향 모델 적용
- 목표 지점 탐색 및 자율 경로 생성

---

### 2️⃣ 화성 지형 시뮬레이션

- 암석(충돌 감지 대상) 포함 Terrain 구성
- Basement 위치 Rover 생성
- Raycaster 기반 지형 높이 계산

---

### 3️⃣ 물리 및 센서 시뮬레이션

- PhysX 물리 엔진 기반 동역학 처리
- Contact Sensor 기반 충돌 감지
- Camera / RGBD 센서 확장 가능 구조

---

### 4️⃣ ROS2 연동 구조

- Isaac ROS2 Bridge 활용
- 카메라/상태 토픽 퍼블리시
- 외부 모니터링 노드 연동 가능

---

### 5️⃣ 시나리오 실행 및 평가 시스템

- 미리 학습된 정책 로드 후 데모 실행
- 특정 목표 좌표로 이동 시나리오
- 베이스캠프 생성 및 마커 시각화

---

## 🛠️ 시스템 설계(System Architecture)

---

### 1️⃣ **자율주행**  2️⃣ 주행정보 **모니터링** 3️⃣ 모니터링 내역 **DB 관리**

1. 자율주행(PPO) : 학습 모델로 랜덤 생성된 광물의 위치 정보를 전달 받아 주행합니다.
2. 모니터링 : Rviz2 와 별도 모니터링 Node 를 통해 현재 광물 수집 경과를 모니터링 합니다.
    1. Rviz2 : 주행시 Rover 의 시점(image_raw)
    2. monitor node : 현재 주행정보 [수집 광물, 목표 위치까지 거리, 속도, 현 위치, 평균 수집량]
        
        ![모니터링 캡쳐](images/monitoring_capture.png)
        
3. DB : 광물 수집 경과에 대한 DB 를 쌓아 해당 task에 대한 사후평가 자료를 정리합니다.

## 🔃 Flow Chart

---

![Flow Chart](images/mermaid-diagram-5.png)

## 💻 개발 환경(Environment)

---

- OS : Ubuntu 22.04 LTS
- Middleware : ROS2 Humble
- Language : Python 3.10 (IssacSim Python 3.11)
    
    gymnasium==1.2.3
    isaaclab==2.3.2.post1
    isaacsim==5.1.0.0
    numpy==1.21.5
    omni==0.0.2
    skrl==1.4.3
    torch==2.10.0
    

## ⚙️ 사용 장비(Hardware Setup)

---

PC : MSI Vector 16 

CPU : intel Ultra 9 275HX (intel AI Boost, NPU)

GPU : nvidia gefore RTX 5080 Laptop GPU (16GB GDDR7, 1.334 AI TOPS)

RAM : 64GB

## 📂 **의존성 설치(Installation)**

---

### 📦 Mars Terrain Assets Setup

GitHub 파일 용량 제한으로 인해 **지형(terrain) 에셋은 저장소에 포함되어 있지 않습니다.**

아래 절차에 따라 수동으로 다운로드 후 배치해 주세요.

---

### 1️⃣ 에셋 다운로드

GitHub Releases 페이지에서 다운로드합니다.

👉 https://github.com/June2December/Rokey6-B1-Isaac-simulation-project/releases

다운로드 파일:

```
mars_terrain_assets.zip
```

---

### 2️⃣ 압축 해제

다운로드한 `mars_terrain_assets.zip` 파일을 아래 경로에 압축 해제합니다.

```
rover/sim/rover_envs/assets/terrains/mars/
```

---

### 3️⃣ 폴더 구조 확인

최종 디렉토리 구조는 다음과 같아야 합니다.

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

### 🔧 Environment Variables Setup (Required)

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

## 🚀 실행 순서(How to Run)

1. Executing simulation with .sh
    
    ```bash
    ~/Rokey6-B1-Isaac-simulation-project/rover/sim/run_ros2.sh
    
    ```
    
2. Executing monitoring node, after ros setting
    
    ```bash
    source /opt/ros/humble/setup.bash
    python3 ~/Rokey6-B1-Isaac-simulation-project/rover/sim/scripts/mission_monitor.py
    ```
