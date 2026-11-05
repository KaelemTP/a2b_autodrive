#!/usr/bin/env python3
"""
lidar_loc_bridge_node  -  lidar odometry -> robominer_msgs/LocalizationData

Chain on the mini truck:
    RoboSense Airy (rslidar_sdk)  -> /rslidar_points (PointCloud2)
    KISS-ICP lidar odometry       -> /kiss/odometry  (nav_msgs/Odometry, pose of the lidar)
    this node                     -> /localization/localization_data (LocalizationData)
    a2b_planner                   -> /pwm_duty -> Modbus driver -> truck

The odometry frame starts where the lidar was when KISS-ICP started: x = straight ahead,
y = left, yaw 0 = the starting direction (not east). Use goal_relative B ("N metres ahead").

The planner's position is the rear axle, so the lidar pose is shifted back by
lidar_x_offset (how far the lidar sits in front of the rear axle), and the frame is moved so
the rear axle starts at (0, 0).
Speed is taken from the odometry twist if it has one, otherwise from position differences.
If odometry stops for odom_timeout s, or jumps faster than the truck can move (lidar odometry
lost track), is_valid goes false and the planner stops the truck.
"""
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from robominer_msgs.msg import LocalizationData, LocalizationStatus


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class LidarLocBridge(Node):
    def __init__(self):
        super().__init__("lidar_loc_bridge")
        self.declare_parameter("odom_topic", "/kiss/odometry")
        self.declare_parameter("localization_topic", "/localization/localization_data")
        self.declare_parameter("lidar_x_offset", 0.0)      # m, lidar in front of rear axle (+)
        self.declare_parameter("lidar_y_offset", 0.0)      # m, lidar left of centre line (+)
        self.declare_parameter("lidar_yaw_offset_deg", 0.0)  # mounting yaw, lidar x vs truck forward
        self.declare_parameter("odom_timeout", 0.5)        # s without odometry -> is_valid false
        self.declare_parameter("speed_filter", 0.4)        # 0..1, weight of the newest speed sample
        self.declare_parameter("rate_hz", 50.0)            # LocalizationData output rate
        self.declare_parameter("max_speed", 3.0)           # m/s; faster jumps = odometry lost
        self.declare_parameter("max_yaw_rate_deg", 120.0)  # deg/s; faster turns = odometry lost
        self.declare_parameter("lost_hold", 2.0)           # s to keep is_valid false after a jump
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.dx, self.dy = g("lidar_x_offset"), g("lidar_y_offset")
        self.yaw_off = math.radians(g("lidar_yaw_offset_deg"))
        self.timeout = g("odom_timeout")
        self.alpha = g("speed_filter")
        self.max_speed, self.max_yaw_rate = g("max_speed"), math.radians(g("max_yaw_rate_deg"))
        self.lost_hold = g("lost_hold")
        self.lost_until = 0.0
        self.prev_odom = None     # (t, x, y, yaw) of the last odometry message

        self.pose = None          # (x, y, yaw) of the rear axle
        self.speed = 0.0
        self.last = None          # (t, x, y) for speed by differences
        self.odom_t = 0.0
        self.count = 0

        qos_in = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_out = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(Odometry, g("odom_topic"), self.on_odom, qos_in)
        self.pub = self.create_publisher(LocalizationData, g("localization_topic"), qos_out)
        self.create_timer(1.0 / g("rate_hz"), self.tick)
        self.create_timer(5.0, self.report)
        self.get_logger().info(f"lidar -> LocalizationData: {g('odom_topic')} -> {g('localization_topic')}, "
                               f"lidar {self.dx:.2f} m ahead of rear axle")

    def on_odom(self, msg):
        p = msg.pose.pose
        yaw = yaw_from_quat(p.orientation) - self.yaw_off          # truck heading
        c, s = math.cos(yaw), math.sin(yaw)
        # lidar -> rear axle, then shift so the rear axle starts at (0, 0): the odometry frame's
        # origin is the lidar's start position, which is (dx, dy) ahead of the rear axle
        x = p.position.x - (c * self.dx - s * self.dy) + self.dx
        y = p.position.y - (s * self.dx + c * self.dy) + self.dy
        now = time.monotonic()

        # sanity check: lidar odometry that jumps faster than the truck can move has lost track
        # (e.g. KISS-ICP diverging). Then report invalid so the planner stops the truck.
        if self.prev_odom is not None:
            dt = max(now - self.prev_odom[0], 1e-3)
            v_jump = math.hypot(x - self.prev_odom[1], y - self.prev_odom[2]) / dt
            w_jump = abs(math.atan2(math.sin(yaw - self.prev_odom[3]), math.cos(yaw - self.prev_odom[3]))) / dt
            if dt < 1.0 and (v_jump > self.max_speed or w_jump > self.max_yaw_rate):
                if now >= self.lost_until:
                    self.get_logger().error(
                        f"lidar odometry jumped ({v_jump:.1f} m/s, {math.degrees(w_jump):.0f} deg/s): "
                        "localization LOST, reporting invalid. Restart KISS-ICP to recover.")
                self.lost_until = now + self.lost_hold
        self.prev_odom = (now, x, y, yaw)

        tw = msg.twist.twist.linear
        v_twist = math.hypot(tw.x, tw.y)
        if v_twist > 1e-6:
            v = v_twist
        elif self.last is not None and now - self.last[0] > 1e-3:
            v = math.hypot(x - self.last[1], y - self.last[2]) / (now - self.last[0])
        else:
            v = 0.0
        self.speed += self.alpha * (v - self.speed)
        self.last = (now, x, y)
        self.pose = (x, y, yaw)
        self.odom_t = now
        self.count += 1

    def tick(self):
        m = LocalizationData()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "odom"
        now = time.monotonic()
        fresh = self.pose is not None and now - self.odom_t < self.timeout and now >= self.lost_until
        m.is_valid = bool(fresh)
        if self.pose is not None:
            x, y, yaw = self.pose
            m.x, m.y = float(x), float(y)
            m.yaw = float(math.degrees(yaw))
            m.speed = float(self.speed)
            m.vx, m.vy = float(self.speed * math.cos(yaw)), float(self.speed * math.sin(yaw))
        m.rtk_status = LocalizationStatus(status=LocalizationStatus.RTKOTHER if fresh else LocalizationStatus.RTKINIT)
        self.pub.publish(m)

    def report(self):
        if self.pose is None:
            self.get_logger().warn("no lidar odometry yet - is KISS-ICP running on the lidar topic?")
        else:
            x, y, yaw = self.pose
            self.get_logger().info(f"odom {self.count / 5.0:.0f} Hz  x={x:.2f} y={y:.2f} "
                                   f"yaw={math.degrees(yaw):.1f} deg  v={self.speed:.2f} m/s")
        self.count = 0


def main():
    rclpy.init()
    node = LidarLocBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
