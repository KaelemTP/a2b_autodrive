#!/usr/bin/env python3
"""
mujoco_sim_node  -  MuJoCo truck + lidar, replaces truck_sim + fake_lidar in the lidar test.

    /pwm_duty (Float32MultiArray)      -> MuJoCo truck (planar force model, collisions)
    /sim/ground_truth (LocalizationData)  true rear-axle pose, for the viewers / error print only
    /mujoco/points (PointCloud2)       lidar sweep ray-cast against the 3D yard, in the lidar frame
    TF world -> base_link

The planner never sees the true pose: the points go through ground_filter -> KISS-ICP ->
lidar_loc_bridge like on the real truck.

gui:=true opens the MuJoCo viewer with the route (cyan), B (green), the lidar estimate (amber
arrow + trail) and the latest lidar returns drawn into the 3D scene. Needs a display (WSLg works).
Same 0.5 s rule as the real PWM driver: no /pwm_duty for 0.5 s -> everything 7.5 %.
"""
import math
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from robominer_msgs.msg import LocalizationData, LocalizationStatus

from a2b_autodrive.core import Params, NEUTRAL
from a2b_autodrive import mujoco_world as mw


class MujocoSim(Node):
    def __init__(self):
        super().__init__("mujoco_sim")
        self.declare_parameter("start_x", 0.0)
        self.declare_parameter("start_y", 0.0)
        self.declare_parameter("start_yaw_deg", 0.0)
        self.declare_parameter("top_speed", 2.0)        # m/s at 10 % duty
        self.declare_parameter("wheelbase", 0.30)
        self.declare_parameter("max_steer_deg", 30.0)
        self.declare_parameter("truth_topic", "/sim/ground_truth")
        self.declare_parameter("points_topic", "/mujoco/points")
        self.declare_parameter("estimate_topic", "/localization/localization_data")
        self.declare_parameter("frame_id", "lidar")
        self.declare_parameter("lidar_rate_hz", 10.0)
        self.declare_parameter("noise", 0.01)           # m, range noise
        self.declare_parameter("gui", True)             # MuJoCo 3D viewer window
        self.declare_parameter("gui_scan_points", 1500)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        p = Params(wheelbase=g("wheelbase"), max_steer_deg=g("max_steer_deg"))
        self.start = (g("start_x"), g("start_y"), math.radians(g("start_yaw_deg")))
        self.world = mw.MujocoTruckWorld(p, start=self.start, top_speed=g("top_speed"), lidar_noise=g("noise"))
        self.frame = g("frame_id")
        self.pwm, self.pwm_t = [NEUTRAL] * 3, 0.0
        self.est, self.path = None, []
        self.est_trail, self.truth_trail = [], []
        self.scan_points = g("gui_scan_points")

        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(Float32MultiArray, "/pwm_duty", self.on_pwm, 10)
        self.create_subscription(LocalizationData, g("estimate_topic"), self.on_est, be)
        self.create_subscription(Path, "/a2b/path", self.on_path, 1)
        self.pub_truth = self.create_publisher(LocalizationData, g("truth_topic"), be)
        self.pub_pts = self.create_publisher(PointCloud2, g("points_topic"), be)
        self.tf = TransformBroadcaster(self)

        self.viewer = None
        if g("gui") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            # MuJoCo's viewer exits the whole process without a display, so check first
            self.get_logger().warn("gui:=true but no DISPLAY - running headless. "
                                   "Use the browser view (sim_viewer) instead.")
        elif g("gui"):
            try:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.world.model, self.world.data,
                                                           show_left_ui=False, show_right_ui=False)
                with self.viewer.lock():
                    self.viewer.cam.lookat[:] = [self.start[0] + 4.0, self.start[1] + 1.0, 0.0]
                    self.viewer.cam.distance, self.viewer.cam.elevation, self.viewer.cam.azimuth = 10.0, -38.0, 60.0
            except Exception as e:  # no display, no GL
                self.get_logger().warn(f"MuJoCo viewer not available ({e}); running headless. "
                                       "Use the browser view (sim_viewer) instead.")
                self.viewer = None

        self.dt = 0.02
        self.lidar_every = max(1, int(round(1.0 / (g("lidar_rate_hz") * self.dt))))
        self.k = 0
        self.create_timer(self.dt, self.tick)
        self.create_timer(2.0, self.report)
        self.get_logger().info(f"MuJoCo sim (50 Hz physics, lidar {g('lidar_rate_hz'):.0f} Hz -> "
                               f"{g('points_topic')}), truck at ({self.start[0]}, {self.start[1]}), "
                               f"yaw {g('start_yaw_deg')} deg")

    # -- inputs -------------------------------------------------------------
    def on_pwm(self, msg):
        if len(msg.data) >= 3:
            self.pwm = list(msg.data[:3])
            self.pwm_t = time.monotonic()

    def on_est(self, m):
        if m.is_valid:
            self.est = mw.odom_to_world(self.start, m.x, m.y, math.radians(m.yaw))
            if not self.est_trail or math.hypot(self.est[0] - self.est_trail[-1][0],
                                                self.est[1] - self.est_trail[-1][1]) > 0.05:
                self.est_trail.append(self.est[:2])

    def on_path(self, msg):
        # the planner's path is in the odometry frame (rear axle starts at 0, 0)
        self.path = [mw.odom_to_world(self.start, ps.pose.position.x, ps.pose.position.y, 0.0)[:2]
                     for ps in msg.poses]

    # -- loop ---------------------------------------------------------------
    def tick(self):
        if time.monotonic() - self.pwm_t > 0.5:
            self.pwm = [NEUTRAL] * 3
        w = self.world
        if self.viewer is not None:
            with self.viewer.lock():
                x, y, yaw, v = w.step(self.pwm, self.dt)
        else:
            x, y, yaw, v = w.step(self.pwm, self.dt)
        stamp = self.get_clock().now().to_msg()

        m = LocalizationData()
        m.header.stamp, m.header.frame_id = stamp, "world"
        m.is_valid = True
        m.x, m.y, m.yaw, m.speed = x, y, math.degrees(yaw), v
        m.vx, m.vy = v * math.cos(yaw), v * math.sin(yaw)
        m.rtk_status = LocalizationStatus(status=LocalizationStatus.RTKFIX)
        self.pub_truth.publish(m)

        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, "world", "base_link"
        t.transform.translation.x, t.transform.translation.y = x, y
        t.transform.rotation.z, t.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.tf.sendTransform(t)

        if not self.truth_trail or math.hypot(x - self.truth_trail[-1][0], y - self.truth_trail[-1][1]) > 0.05:
            self.truth_trail.append((x, y))

        if self.k % self.lidar_every == 0:
            self.publish_scan(w.scan(), stamp)
        self.k += 1
        if self.viewer is not None:
            self.draw()

    def publish_scan(self, pts, stamp):
        msg = PointCloud2()
        msg.header.stamp, msg.header.frame_id = stamp, self.frame
        msg.height, msg.width = 1, len(pts)
        msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                      for i, n in enumerate("xyz")]
        msg.is_bigendian = False
        msg.point_step, msg.row_step = 12, 12 * len(pts)
        msg.is_dense = True
        msg.data = pts.tobytes()
        self.pub_pts.publish(msg)

    def draw(self):
        if not self.viewer.is_running():
            self.get_logger().info("MuJoCo viewer closed; sim keeps running headless")
            self.viewer = None
            return
        with self.viewer.lock():
            scn = self.viewer.user_scn
            scn.ngeom = 0
            if len(self.path) > 1:
                pts = self.path[::5] + [self.path[-1]]
                mw.draw_polyline(scn, pts, (0.2, 0.95, 1.0, 1.0), width=0.022)
                mw.draw_sphere(scn, self.path[-1][0], self.path[-1][1], 0.15, 0.15, (0.3, 0.95, 0.45, 1.0))
            if len(self.truth_trail) > 1:
                mw.draw_polyline(scn, self.truth_trail[-300:], (0.55, 1.0, 0.8, 1.0), width=0.015, z=0.02)
            if len(self.est_trail) > 1:
                mw.draw_polyline(scn, self.est_trail[-300:], (1.0, 0.72, 0.25, 1.0), width=0.015, z=0.045)
            if self.est is not None:
                mw.draw_arrow(scn, *self.est, (1.0, 0.72, 0.25, 1.0), length=0.5, z=0.55)
            mw.draw_scan(scn, self.world.last_scan_world, max_points=self.scan_points)
        self.viewer.sync()

    def report(self):
        x, y, yaw = self.world.pose
        line = f"TRUE x={x:6.2f} y={y:6.2f} yaw={math.degrees(yaw):6.1f}"
        if self.est is not None:
            ex, ey, eyaw = self.est
            line += (f" | LIDAR x={ex:6.2f} y={ey:6.2f} yaw={math.degrees(eyaw):6.1f}"
                     f" | error {math.hypot(ex - x, ey - y):.2f} m")
        else:
            line += " | LIDAR estimate: none yet"
        self.get_logger().info(line)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


def main():
    rclpy.init()
    node = MujocoSim()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
