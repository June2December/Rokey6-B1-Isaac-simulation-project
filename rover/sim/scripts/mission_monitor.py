"""
mission_monitor.py — 미션 상태 터미널 모니터

사용법 (별도 터미널에서):
  source /opt/ros/humble/setup.bash
  python3 scripts/mission_monitor.py
"""
import json
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import os


def clear():
    os.system("clear")


class MissionMonitor(Node):
    def __init__(self):
        super().__init__("mission_monitor")

        self.status = {}
        self.poses  = {}

        # 속도 계산용
        self._prev_pos  = {}   # {idx: (x, y)}
        self._prev_time = {}   # {idx: float}
        self.speeds     = {}   # {idx: float} m/s

        # 수집 통계용
        self._start_time    = time.time()
        self._prev_collected = {}  # {idx: int}
        self.collect_per_min = {}  # {idx: float}

        self.create_subscription(String,      "/mission/status", self._on_status, 10)
        self.create_subscription(PoseStamped, "/robot0/pose", lambda m: self._on_pose(0, m), 10)
        self.create_subscription(PoseStamped, "/robot1/pose", lambda m: self._on_pose(1, m), 10)

        self.create_timer(1.0, self._display)  # 1Hz 화면 갱신

    def _on_status(self, msg):
        try:
            self.status = json.loads(msg.data)
        except Exception:
            pass

    def _on_pose(self, idx, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        now = time.time()

        # 속도 계산 (이전 위치와 현재 위치 차이 / 시간)
        if idx in self._prev_pos:
            px, py = self._prev_pos[idx]
            dt = now - self._prev_time[idx]
            if dt > 0.05:  # 너무 짧은 간격 무시
                dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                self.speeds[idx] = dist / dt

        self._prev_pos[idx]  = (x, y)
        self._prev_time[idx] = now
        self.poses[idx]      = (x, y)

    def _display(self):
        now     = time.time()
        elapsed = now - self._start_time
        elapsed_min = elapsed / 60.0

        clear()
        print("=" * 56)
        print("         🚀 듀얼 로버 미션 모니터")
        print(f"         경과 시간: {int(elapsed//60):02d}:{int(elapsed%60):02d}")
        print("=" * 56)

        total_collected = 0

        for i in range(2):
            st  = self.status.get(str(i), {})
            pos = self.poses.get(i)

            phase     = st.get("phase",     "—")
            collected = st.get("collected", 0)
            total     = 5
            rnd       = st.get("round",     1)
            distance  = st.get("distance",  None)
            speed     = self.speeds.get(i,  None)

            # 누적 수집량 = (라운드-1)*5 + 현재 수집
            total_this_robot = (rnd - 1) * total + collected
            total_collected += total_this_robot

            # 분당 수집 (누적 / 경과분)
            cpm = total_this_robot / elapsed_min if elapsed_min > 0.1 else 0.0

            phase_str = "💎 수집 중" if phase == "collect" else "🏠 복귀 중"
            bar       = "█" * collected + "░" * (total - collected)
            pos_str   = f"({pos[0]:.1f}, {pos[1]:.1f})" if pos else "—"
            dist_str  = f"{distance:.1f}m" if distance is not None else "—"
            spd_str   = f"{speed:.2f} m/s" if speed is not None else "—"

            print(f"\n  Robot{i}  |  라운드 {rnd}  |  {phase_str}")
            print(f"  광물:   [{bar}] {collected}/{total}  (누적 {total_this_robot}개)")
            print(f"  목표까지: {dist_str}")
            print(f"  현재 속도: {spd_str}")
            print(f"  위치:   {pos_str}")
            print(f"  평균 수집량: {cpm:.2f} 개/min")

        # 전체 통계
        overall_cpm = total_collected / elapsed_min if elapsed_min > 0.1 else 0.0
        print(f"\n{'─' * 56}")
        print(f"  전체 누적 수집: {total_collected}개   평균 {overall_cpm:.2f} 개/min")
        print("=" * 56)
        print("  Ctrl+C 로 종료")


def main():
    rclpy.init()
    node = MissionMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()