"""
mission_monitor.py — 미션 상태 터미널 모니터

사용법 (별도 터미널에서):
  source /opt/ros/humble/setup.bash
  python3 scripts/mission_monitor.py
"""
import csv
import json
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped


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
        self._start_time     = time.time()
        self._prev_collected = {}  # {idx: int}  — 이벤트 감지용
        self._prev_round     = {}  # {idx: int}  — 베이스캠프 복귀 감지용
        self.collect_per_min = {}  # {idx: float}

        # ── CSV 저장 설정 ────────────────────────────────────────────────────
        self._ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _script_dir  = os.path.dirname(os.path.abspath(__file__))
        self._log_dir = os.path.normpath(
            os.path.join(_script_dir, "..", "..", "analysis", "logs")
        )
        os.makedirs(self._log_dir, exist_ok=True)

        # 이벤트 CSV (mineral_collect / basecamp_return)
        _event_path = os.path.join(self._log_dir, f"mission_events_{self._ts}.csv")
        self._event_f = open(_event_path, "w", newline="", encoding="utf-8")
        self._event_w = csv.writer(self._event_f)
        self._event_w.writerow(
            ["elapsed_s", "event_type", "robot_id", "round", "mineral_num", "x", "y"]
        )

        # 시계열 로그 CSV (속도 / 거리 / 위치)
        _log_path = os.path.join(self._log_dir, f"mission_log_{self._ts}.csv")
        self._log_f = open(_log_path, "w", newline="", encoding="utf-8")
        self._log_w = csv.writer(self._log_f)
        self._log_w.writerow(
            ["elapsed_s",
             "r0_speed", "r0_distance", "r0_x", "r0_y",
             "r1_speed", "r1_distance", "r1_x", "r1_y"]
        )

        print(f"[CSV] 저장 경로  : {self._log_dir}")
        print(f"[CSV] 이벤트 파일: mission_events_{self._ts}.csv")
        print(f"[CSV] 로그 파일  : mission_log_{self._ts}.csv")
        # ─────────────────────────────────────────────────────────────────────

        self.create_subscription(String,      "/mission/status", self._on_status, 10)
        self.create_subscription(PoseStamped, "/robot0/pose", lambda m: self._on_pose(0, m), 10)
        self.create_subscription(PoseStamped, "/robot1/pose", lambda m: self._on_pose(1, m), 10)

        self.create_timer(0.1, self._display)  # 1Hz 화면 갱신

    def _on_status(self, msg):
        try:
            self.status = json.loads(msg.data)
        except Exception:
            return

        elapsed = time.time() - self._start_time

        for i in range(2):
            st        = self.status.get(str(i), {})
            collected = st.get("collected", 0)
            round_num = st.get("round",     1)
            x         = st.get("x")
            y         = st.get("y")

            prev_col = self._prev_collected.get(i, 0)
            prev_rnd = self._prev_round.get(i,     1)

            # 광물 수집 이벤트
            if collected > prev_col:
                self._event_w.writerow([
                    round(elapsed, 3), "mineral_collect",
                    i, round_num, collected, x, y,
                ])
                self._event_f.flush()

            # 베이스캠프 복귀 이벤트 (라운드 증가 시)
            if round_num > prev_rnd:
                self._event_w.writerow([
                    round(elapsed, 3), "basecamp_return",
                    i, prev_rnd, "", x, y,
                ])
                self._event_f.flush()

            self._prev_collected[i] = collected
            self._prev_round[i]     = round_num

    def _on_pose(self, idx, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        now = time.time()

        # 속도 계산 (이전 위치와 현재 위치 차이 / 시간)
        if idx in self._prev_pos:
            px, py = self._prev_pos[idx]
            dt = now - self._prev_time[idx]
            if dt > 0:  # 0.05 조건 제거
                dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                raw_speed = dist / dt

                if idx in self.speeds:
                    self.speeds[idx] = 0.05 * raw_speed + 0.95 * self.speeds[idx]
                else:
                    self.speeds[idx] = raw_speed

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

        # ── 시계열 로그 CSV 기록 ────────────────────────────────────────────
        row = [round(elapsed, 3)]
        for i in range(2):
            st  = self.status.get(str(i), {})
            pos = self.poses.get(i)
            row.append(round(self.speeds.get(i) or 0.0, 4) if self.speeds.get(i) is not None else "")
            row.append(round(st.get("distance") or 0.0, 4) if st.get("distance") is not None else "")
            row.append(round(pos[0], 4) if pos else "")
            row.append(round(pos[1], 4) if pos else "")
        self._log_w.writerow(row)
        self._log_f.flush()
        # ─────────────────────────────────────────────────────────────────────


def main():
    rclpy.init()
    node = MissionMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._event_f.close()
        node._log_f.close()
        print(f"\n[CSV] 저장 완료: {node._log_dir}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()