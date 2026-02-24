"""
mission_monitor.py — 미션 상태 터미널 모니터

사용법 (별도 터미널에서):
  source /opt/ros/humble/setup.bash
  python scripts/mission_monitor.py
"""
import json
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

        self.status   = {}
        self.poses    = {}

        self.create_subscription(String,      "/mission/status", self._on_status, 10)
        self.create_subscription(PoseStamped, "/robot0/pose",    lambda m: self._on_pose(0, m), 10)
        self.create_subscription(PoseStamped, "/robot1/pose",    lambda m: self._on_pose(1, m), 10)

        # 1Hz로 화면 갱신
        self.create_timer(1.0, self._display)

    def _on_status(self, msg):
        try:
            self.status = json.loads(msg.data)
        except Exception:
            pass

    def _on_pose(self, idx, msg):
        self.poses[idx] = (msg.pose.position.x, msg.pose.position.y)

    def _display(self):
        clear()
        print("=" * 50)
        print("       🚀 듀얼 로버 미션 모니터")
        print("=" * 50)

        for i in range(2):
            st  = self.status.get(str(i), {})
            pos = self.poses.get(i, None)

            phase     = st.get("phase",     "—")
            collected = st.get("collected", 0)
            total     = 5
            rnd       = st.get("round",     1)

            phase_str = "💎 수집 중" if phase == "collect" else "🏠 복귀 중"
            bar       = "█" * collected + "░" * (total - collected)
            pos_str   = f"({pos[0]:.1f}, {pos[1]:.1f})" if pos else "—"

            print(f"\n  Robot{i}  |  라운드 {rnd}  |  {phase_str}")
            print(f"  광물: [{bar}] {collected}/{total}")
            print(f"  위치: {pos_str}")

        print("\n" + "=" * 50)
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
