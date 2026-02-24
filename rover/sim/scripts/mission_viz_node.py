#!/usr/bin/env python3
"""
mission_viz_node.py — ROS2 표현 전용 노드

python3.10 ROS 환경에서 실행. visualization_msgs 정상 동작.
Isaac Sim(03_eval_ros2.py) 이 퍼블리시하는 /mission/status 와 /robot{i}/pose 를
구독해서 RViz 용 MarkerArray(/mission/markers) 를 생성.

실행:
  source /opt/ros/humble/setup.bash
  python3 ~/Rokey6-B1-Isaac-simulation-project/rover/sim/scripts/mission_viz_node.py
"""
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray


class MissionVizNode(Node):
    def __init__(self):
        super().__init__("mission_viz_node")

        self.num_robots = 2
        self.status     = {}
        self.poses      = [None] * self.num_robots

        # 구독
        self.create_subscription(String, "/mission/status", self._on_status, 10)
        for i in range(self.num_robots):
            self.create_subscription(
                PoseStamped, f"/robot{i}/pose",
                lambda msg, idx=i: self._on_pose(msg, idx),
                10,
            )

        # 퍼블리시
        self.marker_pub = self.create_publisher(MarkerArray, "/mission/markers", 10)
        self.hud_pub    = self.create_publisher(String,      "/mission/hud",     10)

        self.create_timer(0.1, self._publish_markers)  # 10Hz

    def _on_status(self, msg: String):
        try:
            self.status = json.loads(msg.data)
        except Exception:
            self.status = {}

    def _on_pose(self, msg: PoseStamped, idx: int):
        self.poses[idx] = msg

    def _publish_markers(self):
        now     = self.get_clock().now().to_msg()
        markers = MarkerArray()
        hud_lines = []

        for i in range(self.num_robots):
            pose = self.poses[i]
            if pose is None:
                continue

            st        = self.status.get(str(i), self.status.get(i, {}))
            phase     = st.get("phase",     "unknown")
            collected = st.get("collected", 0)
            round_num = st.get("round",     1)
            dist      = st.get("distance",  None)
            px        = pose.pose.position.x
            py        = pose.pose.position.y

            dist_str  = "dist: ?" if dist is None else f"dist: {float(dist):.1f}m"
            text = (
                f"Robot{i}\n"
                f"{phase} (R{round_num})\n"
                f"{dist_str}\n"
                f"collected: {collected}\n"
                f"pos: ({px:.1f}, {py:.1f})"
            )

            m = Marker()
            m.header.stamp    = now
            m.header.frame_id = "world"
            m.ns              = "mission_status"
            m.id              = i
            m.type            = Marker.TEXT_VIEW_FACING
            m.action          = Marker.ADD
            m.pose.position.x = px
            m.pose.position.y = py
            m.pose.position.z = 2.0
            m.pose.orientation.w = 1.0
            m.scale.z         = 0.8
            m.color.r         = 1.0
            m.color.g         = 1.0
            m.color.b         = 1.0
            m.color.a         = 1.0
            m.text            = text
            markers.markers.append(m)

            hud_lines.append(
                f"[Robot{i}] {phase} R{round_num} | {dist_str} | "
                f"pos=({px:.1f},{py:.1f}) | collected={collected}"
            )

        self.marker_pub.publish(markers)

        hud      = String()
        hud.data = "\n".join(hud_lines) if hud_lines else "waiting..."
        self.hud_pub.publish(hud)


def main():
    rclpy.init()
    node = MissionVizNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
