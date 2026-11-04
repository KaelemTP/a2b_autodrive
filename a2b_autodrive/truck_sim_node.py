#!/usr/bin/env python3
"""
truck_sim_node  -  stand-in for Simmoulde while you test alone.

Subscribes /pwm_duty (the same [PWM1, PWM2, PWM3] the real Modbus driver uses), moves a
kinematic model of the truck, and publishes what Simmoulde gives Pnc:
    /localization/localization_data   robominer_msgs/LocalizationData (x, y ENU m, yaw deg, speed)
Also publishes TF world->base_link for RViz.

Same 0.5 s rule as the real driver: no /pwm_duty for 0.5 s -> everything 7.5 %.
"""
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from robominer_msgs.msg import LocalizationData, LocalizationStatus

from a2b_autodrive.core import Params, TruckSim, NEUTRAL


class TruckSimNode(Node):
    def __init__(self):
        super().__init__("truck_sim")
        self.declare_parameter("start_x", 0.0)
        self.declare_parameter("start_y", 0.0)
        self.declare_parameter("start_yaw_deg", 0.0)   # ENU degrees, 0 = east
        self.declare_parameter("top_speed", 2.0)       # m/s at 10 % duty
        self.declare_parameter("wheelbase", 0.30)
        self.declare_parameter("max_steer_deg", 30.0)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        p = Params(wheelbase=g("wheelbase"), max_steer_deg=g("max_steer_deg"))
        self.sim = TruckSim(p, g("start_x"), g("start_y"), math.radians(g("start_yaw_deg")), g("top_speed"))
        self.pwm = [NEUTRAL] * 3
        self.pwm_t = 0.0
        self.create_subscription(Float32MultiArray, "/pwm_duty", self.on_pwm, 10)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.pub_loc = self.create_publisher(LocalizationData, "/localization/localization_data", qos)
        self.tf = TransformBroadcaster(self)
        self.dt = 0.02
        self.create_timer(self.dt, self.tick)
        self.get_logger().info(f"truck sim running (50 Hz) at ({g('start_x')}, {g('start_y')}), "
                               f"yaw {g('start_yaw_deg')} deg")

    def on_pwm(self, msg):
        if len(msg.data) >= 3:
            self.pwm = list(msg.data[:3])
            self.pwm_t = time.monotonic()

    def tick(self):
        if time.monotonic() - self.pwm_t > 0.5:
            self.pwm = [NEUTRAL] * 3
        x, y, yaw, v = self.sim.step(self.pwm, self.dt)
        stamp = self.get_clock().now().to_msg()

        m = LocalizationData()
        m.header.stamp, m.header.frame_id = stamp, "world"
        m.is_valid = True
        m.x, m.y = x, y
        m.yaw = math.degrees(yaw)
        m.speed = v
        m.vx, m.vy = v * math.cos(yaw), v * math.sin(yaw)
        m.rtk_status = LocalizationStatus(status=LocalizationStatus.RTKFIX)
        self.pub_loc.publish(m)

        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, "world", "base_link"
        t.transform.translation.x, t.transform.translation.y = x, y
        t.transform.rotation.z, t.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.tf.sendTransform(t)


def main():
    rclpy.init()
    node = TruckSimNode()
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
