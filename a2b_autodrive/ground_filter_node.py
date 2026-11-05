#!/usr/bin/env python3
"""
ground_filter_node  -  removes floor points (and the truck's own body) from a point cloud
before lidar odometry.

Why: on a flat floor the floor points form the same rings around the lidar in every scan,
wherever the truck is. KISS-ICP then thinks the truck moves less than it does (in the test
room it measured 5-22 % too short; with the floor removed, 0.6 %).

Keeps every field of the input cloud (x, y, z, intensity, ring, timestamp, ...) so deskewing
still works. Points are in the lidar's own frame, so this assumes the lidar is mounted level:
  keep a point if  z > min_z  and  min_range <= horizontal distance <= max_range
Set min_z to about -(lidar height above ground - 0.1 m).
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField

_NP = {PointField.INT8: np.int8, PointField.UINT8: np.uint8, PointField.INT16: np.int16,
       PointField.UINT16: np.uint16, PointField.INT32: np.int32, PointField.UINT32: np.uint32,
       PointField.FLOAT32: np.float32, PointField.FLOAT64: np.float64}


class GroundFilter(Node):
    def __init__(self):
        super().__init__("ground_filter")
        self.declare_parameter("input_topic", "/rslidar_points")
        self.declare_parameter("output_topic", "/points_no_ground")
        self.declare_parameter("min_z", -0.3)        # m in the lidar frame; below = floor
        self.declare_parameter("min_range", 0.0)     # m; points closer = the truck itself
        self.declare_parameter("max_range", 100.0)   # m
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.min_z, self.min_r, self.max_r = g("min_z"), g("min_range"), g("max_range")
        self.pub = self.create_publisher(PointCloud2, g("output_topic"), qos_profile_sensor_data)
        self.create_subscription(PointCloud2, g("input_topic"), self.on_cloud, qos_profile_sensor_data)
        self.n_in = self.n_out = 0
        self.create_timer(10.0, self.report)
        self.get_logger().info(f"ground filter: {g('input_topic')} -> {g('output_topic')}, "
                               f"keep z > {self.min_z} m, range {self.min_r}-{self.max_r} m")

    def _field(self, rows, msg, name):
        f = next(f for f in msg.fields if f.name == name)
        dt = np.dtype(_NP[f.datatype]).newbyteorder(">" if msg.is_bigendian else "<")
        return rows[:, f.offset:f.offset + dt.itemsize].copy().view(dt)[:, 0].astype(np.float64)

    def on_cloud(self, msg):
        n = msg.width * msg.height
        if n == 0:
            return
        rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, msg.point_step)
        x, y, z = (self._field(rows, msg, k) for k in "xyz")
        r = np.hypot(x, y)
        keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > self.min_z) \
            & (r >= self.min_r) & (r <= self.max_r)
        out = PointCloud2()
        out.header = msg.header
        out.fields = msg.fields
        out.is_bigendian = msg.is_bigendian
        out.point_step = msg.point_step
        out.height, out.width = 1, int(keep.sum())
        out.row_step = out.point_step * out.width
        out.is_dense = True
        out.data = rows[keep].tobytes()
        self.pub.publish(out)
        self.n_in += n
        self.n_out += out.width

    def report(self):
        if self.n_in:
            self.get_logger().info(f"kept {100.0 * self.n_out / self.n_in:.0f} % of points")
        else:
            self.get_logger().warn("no point clouds received yet")
        self.n_in = self.n_out = 0


def main():
    rclpy.init()
    node = GroundFilter()
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
