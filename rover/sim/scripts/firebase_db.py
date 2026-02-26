"""
Firebase Realtime Database 유틸리티.
eval.py, 03_eval_ros2.py 등 여러 스크립트에서 공유하는 DB 초기화 및 업로드 함수.

외부 패키지 없이 표준 라이브러리(urllib)만으로 Firebase REST API를 사용합니다.
DB 규칙에서 ".write": true 가 설정되어 있어야 합니다.

업로드 스키마 (robot_status 노드):
  step             : int    — 현재 시뮬레이션 스텝
  elapsed_seconds  : float  — 미션 시작 후 경과 시간 (초)
  total_collected  : int    — 전체 로봇 누적 수집량 (현재 라운드 기준)
  robot0 / robot1:
    phase      : str   — "collect" | "return"
    collected  : int   — 현재 라운드 수집 수
    round      : int   — 현재 라운드 번호
    distance   : float — 목표(광물 or 베이스)까지 거리 (m)
    x          : float — 로봇 월드 X 좌표
    y          : float — 로봇 월드 Y 좌표
    velocity   : float — action[0] 기반 속도 출력 (-1.0 ~ 1.0)
"""
from __future__ import annotations

import json as _json
import urllib.request

DATABASE_URL = "https://rokey-93910-default-rtdb.asia-southeast1.firebasedatabase.app"

def init_firebase():
    """REST API 방식에서는 초기화 불필요. 호환성을 위해 유지."""
    print("[Firebase] REST API 모드 — 초기화 생략")


def upload_robot_status(step: int, elapsed_seconds: float, robot_states: dict, actions):
    """
    robot_status 노드에 현재 상태를 REST API(PUT)로 업로드.

    Args:
        step:            현재 시뮬레이션 스텝
        elapsed_seconds: 미션 시작 후 경과 시간 (초)
        robot_states:    {robot_idx: {"phase", "collected", "round", "distance", "x", "y"}}
        actions:         에이전트 액션 텐서 (shape: [NUM_ROBOTS, action_dim])
    """
    def _robot_data(i: int) -> dict:
        st = robot_states[i]
        return {
            "phase":     st["phase"],
            "collected": st["collected"],
            "round":     st["round"],
            "distance":  st.get("distance"),
            "x":         st.get("x"),
            "y":         st.get("y"),
            "velocity":  round(float(actions[i, 0]), 3),
        }

    payload = {
        "step":            step,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "robot0":          _robot_data(0),
        "robot1":          _robot_data(1),
        "total_collected": robot_states[0]["collected"] + robot_states[1]["collected"],
    }

    url  = f"{DATABASE_URL}/robot_status.json"
    data = _json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(url, data=data, method="PUT")
    try:
        urllib.request.urlopen(req, timeout=2)
    except Exception as e:
        print(f"[Firebase] 업로드 실패 (무시): {e}")
