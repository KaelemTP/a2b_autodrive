#!/usr/bin/env python3
"""
fake_lidar_node  -  a simulated 3D lidar in a virtual room, for testing lidar localization
without the real Airy.

It reads the fake truck's TRUE pose (truck_sim publishes it on /sim/ground_truth in this test),
ray-casts a simple room (4 walls, pillars, boxes, slightly uneven ground) from the lidar's position and
publishes a sensor_msgs/PointCloud2 in the lidar's own frame, like a real driver does.
KISS-ICP turns those clouds into odometry; the algorithm never sees the true pose.

Every 2 s it also prints the true pose next to the lidar-based estimate
(/localization/localization_data from lidar_loc_bridge), so you can see the error.
"""
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
from robominer_msgs.msg import LocalizationData

# Room in the world frame (truck starts at 0,0 facing +x). Segments: (x1, y1, x2, y2, height)
WALLS = [(-6, -8, 22, -8, 3.0), (22, -8, 22, 8, 3.0), (22, 8, -6, 8, 3.0), (-6, 8, -6, -8, 3.0)]
# Boxes as rectangles (xmin, ymin, xmax, ymax, height)
BOXES = [(2.5, -5.5, 4.0, -4.0, 1.0), (9.0, 3.5, 10.5, 5.5, 1.5), (16.0, -6.0, 18.0, -4.5, 1.2),
         (6.0, -7.5, 7.0, -6.5, 2.0)]
# Pillars as circles (cx, cy, r, height)
PILLARS = [(5.0, 4.0, 0.3, 3.0), (8.0, -3.0, 0.25, 3.0), (12.5, 5.0, 0.3, 3.0), (14.5, -2.5, 0.3, 3.0),
           (19.0, 2.0, 0.4, 3.0), (-3.0, 4.5, 0.3, 3.0)]


def _segments():
    segs = list(WALLS)
    for x0, y0, x1, y1, h in BOXES:
        segs += [(x0, y0, x1, y0, h), (x1, y0, x1, y1, h), (x1, y1, x0, y1, h), (x0, y1, x0, y0, h)]
    return np.array(segs, dtype=float)


SEGS = _segments()
CIRC = np.array(PILLARS, dtype=float)


def raycast_2d(ox, oy, angles):
    """Horizontal distance and obstacle height for each ray from (ox, oy)."""
    dx, dy = np.cos(angles)[:, None], np.sin(angles)[:, None]           # (N,1)
    # ray vs segments
    ax, ay, bx, by, h = (SEGS[:, i][None, :] for i in range(5))         # (1,M)
    ex, ey = bx - ax, by - ay
    den = dx * ey - dy * ex
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((ax - ox) * ey - (ay - oy) * ex) / den                     # distance along ray
        u = ((ax - ox) * dy - (ay - oy) * dx) / den                     # position along segment
    ok = (np.abs(den) > 1e-9) & (t > 0.05) & (u >= 0) & (u <= 1)
    t = np.where(ok, t, np.inf)
    # ray vs circles
    cx, cy, r, hc = (CIRC[:, i][None, :] for i in range(4))
    fx, fy = ox - cx, oy - cy
    b = fx * dx + fy * dy
    c = fx * fx + fy * fy - r * r
    disc = b * b - c
    tc = -b - np.sqrt(np.clip(disc, 0, None))
    tc = np.where((disc >= 0) & (tc > 0.05), tc, np.inf)
    t_all = np.concatenate([t, tc], axis=1)
    h_all = np.concatenate([np.broadcast_to(h, t.shape), np.broadcast_to(hc, tc.shape)], axis=1)
    idx = np.argmin(t_all, axis=1)
    rows = np.arange(len(angles))
    return t_all[rows, idx], h_all[rows, idx]


def ground_height(wx, wy):
    """Small world-fixed bumps (+-8 cm) on the floor."""
    return 0.05 * np.sin(0.7 * wx) * np.cos(0.5 * wy) + 0.03 * np.sin(2.1 * wx + 1.3 * wy)


def make_scan(x, y, yaw, height, n_az=720, elevations=None, noise=0.01, max_range=40.0, rng=None):
    """Points (K,3) in the lidar frame for a lidar at (x, y, height) with heading yaw."""
    if elevations is None:
        elevations = np.radians(np.linspace(-25.0, 60.0, 24))   # wide vertical view, Airy-like
    rng = rng or np.random.default_rng()
    # real spinning lidars do not fire at the same angles every scan: random start offset
    az = np.linspace(-math.pi, math.pi, n_az, endpoint=False) + rng.uniform(0, 2 * math.pi / n_az)
    d, h_obj = raycast_2d(x, y, az + yaw)                           # world-frame rays
    pts = []
    for e in elevations:
        te = math.tan(e)
        if e < 0:
            dg = height / -te                                       # distance to the ground
            hit_wall = d < dg
            r = np.where(hit_wall, d, dg)
            # slightly uneven ground, fixed in the world, so the floor is not a featureless plane
            wx, wy = x + r * np.cos(az + yaw), y + r * np.sin(az + yaw)
            z = np.where(hit_wall, d * te, -height + ground_height(wx, wy))
            valid = r < max_range
        else:
            r = d
            z = d * te
            valid = (z + height <= h_obj) & (r < max_range)        # above the obstacle = no return
        rr = r[valid] + rng.normal(0, noise, valid.sum())
        a = az[valid]
        pts.append(np.stack([rr * np.cos(a), rr * np.sin(a), z[valid] + rng.normal(0, noise, valid.sum())], 1))
    return np.concatenate(pts).astype(np.float32)


class FakeLidar(Node):
    def __init__(self):
        super().__init__("fake_lidar")
        self.declare_parameter("truth_topic", "/sim/ground_truth")
        self.declare_parameter("points_topic", "/fake_lidar/points")
        self.declare_parameter("estimate_topic", "/localization/localization_data")
        self.declare_parameter("frame_id", "lidar")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("lidar_x_offset", 0.25)   # lidar ahead of rear axle, m
        self.declare_parameter("lidar_height", 0.4)      # m above ground
        self.declare_parameter("noise", 0.01)            # m, range noise
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.frame = g("frame_id")
        self.off, self.h, self.noise = g("lidar_x_offset"), g("lidar_height"), g("noise")
        self.truth = None
        self.est = None
        self.rng = np.random.default_rng(1)
        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(LocalizationData, g("truth_topic"), self.on_truth, be)
        self.create_subscription(LocalizationData, g("estimate_topic"), self.on_est, be)
        self.pub = self.create_publisher(PointCloud2, g("points_topic"), be)
        self.create_timer(1.0 / g("rate_hz"), self.tick)
        self.create_timer(2.0, self.report)
        self.get_logger().info(f"fake lidar: {g('truth_topic')} -> {g('points_topic')} "
                               f"({g('rate_hz'):.0f} Hz, room 28 x 16 m with pillars and boxes)")

    def on_truth(self, m):
        self.truth = (m.x, m.y, math.radians(m.yaw))

    def on_est(self, m):
        if m.is_valid:
            self.est = (m.x, m.y, m.yaw)

    def tick(self):
        if self.truth is None:
            return
        x, y, yaw = self.truth
        lx, ly = x + self.off * math.cos(yaw), y + self.off * math.sin(yaw)
        pts = make_scan(lx, ly, yaw, self.h, noise=self.noise, rng=self.rng)
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame
        msg.height, msg.width = 1, len(pts)
        msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                      for i, n in enumerate("xyz")]
        msg.is_bigendian = False
        msg.point_step, msg.row_step = 12, 12 * len(pts)
        msg.is_dense = True
        msg.data = pts.tobytes()
        self.pub.publish(msg)

    def report(self):
        if self.truth is None:
            self.get_logger().warn("no ground truth yet - is truck_sim publishing /sim/ground_truth?")
            return
        x, y, yaw = self.truth
        line = f"TRUE x={x:6.2f} y={y:6.2f} yaw={math.degrees(yaw):6.1f}"
        if self.est:
            ex, ey, eyaw = self.est
            err = math.hypot(ex - x, ey - y)
            line += f" | LIDAR x={ex:6.2f} y={ey:6.2f} yaw={eyaw:6.1f} | error {err:.2f} m"
        else:
            line += " | LIDAR estimate: none yet"
        self.get_logger().info(line)


def main():
    rclpy.init()
    node = FakeLidar()
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
