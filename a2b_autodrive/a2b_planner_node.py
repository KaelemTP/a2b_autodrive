#!/usr/bin/env python3
"""
a2b_planner_node  -  the A->B algorithm in the "Pnc" slot of the Guardian / WorldSim / Pnc
simulation architecture. Uses the company's robominer_msgs, same topics as Pnc.

Inputs
    /localization/localization_data   robominer_msgs/LocalizationData
        x, y (ENU, m), yaw (ENU, degrees), speed (m/s), is_valid
    /guardian/plan_target             robominer_msgs/PlanTarget
        goal_x, goal_y (ENU, m)  = point B
    /guardian/task_state              robominer_msgs/TaskState
        START_TASK (1) -> drive to B;  PAUSING (17), PAUSE_FINISH (18), STOP_TASK (26),
        ERROR (99) -> stop
    /a2b/enable                       std_msgs/Bool   (manual start/stop for local tests)
    /goal_pose                        geometry_msgs/PoseStamped  (RViz / viewer goal, local tests)
    /a2b/waypoints                    nav_msgs/Path   (points to pass through before B, map frame;
                                                       an empty Path clears them)
    ~obstacle_cloud_topic             sensor_msgs/PointCloud2 in the LIDAR frame with the floor
                                      already removed (ground_filter). "off" = obstacle stop off.
                                      With avoid_enable (default) the truck steers around
                                      obstacles, up to avoid_max_offset m off the route, and comes
                                      back to the route after. If there is no way past: slow down
                                      within obstacle_slow_dist, stop and wait within
                                      obstacle_stop_dist (state BLOCKED), drive on when it is clear.
                                      With reverse_enable (lidar sim only for now) it backs up
                                      0.6 m (state BACKING, PWM3 = reverse_duty) when no way
                                      forward is free, up to 3 times, then waits.

Outputs
    /auto/control_cmd   robominer_msgs/ControlCmd   (to Simmoulde / Guardian, like Pnc)
        target_drive_mode AUTO, target_gear D while driving (N when stopped),
        target_steering degrees, left positive; target_throttle / target_brake 0-100 %;
        target_speed m/s
    /pwm_duty           std_msgs/Float32MultiArray [PWM1, PWM2, PWM3] %  (real truck driver)
    /a2b/path           nav_msgs/Path    (route, RViz)
    /a2b/local_path     nav_msgs/Path    (the path actually followed, incl. detours)
    /a2b/status         std_msgs/String  (state + reason)

Guardian publishes task_state and plan_target BEST_EFFORT, so those subscriptions are
best effort too (a reliable subscriber would receive nothing).
"""

import math
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray, Bool, String
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2

from robominer_msgs.msg import (LocalizationData, ControlCmd, ControlDebug, TaskState, PlanTarget,
                                DriveMode, Gear)

from a2b_autodrive.core import A2BController, Params, NEUTRAL

BEST_EFFORT = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST,
                         depth=5, durability=DurabilityPolicy.VOLATILE)
RELIABLE = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)

START_STATES = {TaskState.START_TASK}
STOP_STATES = {TaskState.PAUSING, TaskState.PAUSE_FINISH, TaskState.STOP_TASK, TaskState.ERROR}


class A2BPlanner(Node):
    def __init__(self):
        super().__init__("a2b_planner")

        self.declare_parameter("localization_topic", "/localization/localization_data")
        self.declare_parameter("plan_target_topic", "/guardian/plan_target")
        self.declare_parameter("task_state_topic", "/guardian/task_state")
        self.declare_parameter("control_topic", "/auto/control_cmd")
        self.declare_parameter("publish_pwm", True)
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("map_frame", "world")
        self.declare_parameter("auto_start", False)
        self.declare_parameter("goal_x", float("nan"))   # fallback B if no plan_target arrives
        self.declare_parameter("goal_y", float("nan"))
        self.declare_parameter("goal_relative", False)   # goal_x/y = metres ahead/left of start
        self.declare_parameter("waypoints", [0.0])       # flat [x1,y1,...] before B; [0.0] = none
        self.declare_parameter("sim_max_steer_deg", 30.0)  # target_steering at full lock
        self.declare_parameter("stop_brake_pct", 60.0)   # target_brake when stopping / stopped
        self.declare_parameter("control_debug_topic", "/auto/control_debug")  # Simmoulde needs it
        self.declare_parameter("accel_gain", 1.0)        # total_accel_cmd = gain * (v_ref - v)
        self.declare_parameter("max_accel", 1.0)         # m/s^2
        self.declare_parameter("max_decel", 1.5)         # m/s^2, used when stopping
        self.declare_parameter("obstacle_cloud_topic", "off")  # e.g. /points_no_ground; "off" = no obstacle stop
        self.declare_parameter("obstacle_cloud_timeout", 1.0)  # s without a cloud -> treat as blocked
        self.declare_parameter("lidar_x_offset", 0.0)    # m, lidar in front of the rear axle

        self.params = Params()
        for name, default in vars(Params()).items():
            self.declare_parameter(name, default)
            setattr(self.params, name, self.get_parameter(name).value)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.ctl = A2BController(self.params)
        self.pose = None
        self.pose_t = 0.0
        self.speed = 0.0
        self.goal = None
        self.goal_from_param = False
        self.task_state = None
        self.last_state = "IDLE"
        self.last_tick = time.monotonic()
        self.map_frame = g("map_frame")
        self.sim_max_steer = g("sim_max_steer_deg")
        self.stop_brake = g("stop_brake_pct")
        self.goal_relative = g("goal_relative")
        self.accel_gain = g("accel_gain")
        self.max_accel = g("max_accel")
        self.max_decel = g("max_decel")
        self.lidar_x = g("lidar_x_offset")
        self.cloud_timeout = g("obstacle_cloud_timeout")
        self.obstacle_on = str(g("obstacle_cloud_topic")).strip().lower() not in ("", "off", "none", "false")
        self.cloud_xyz = None      # (N,3) points of the latest cloud, rear-axle frame
        self.cloud_t = 0.0
        self.obstacle_dist = float("inf")

        self.create_subscription(LocalizationData, g("localization_topic"), self.on_loc, BEST_EFFORT)
        self.create_subscription(PlanTarget, g("plan_target_topic"), self.on_plan_target, BEST_EFFORT)
        self.create_subscription(TaskState, g("task_state_topic"), self.on_task_state, BEST_EFFORT)
        self.create_subscription(PoseStamped, "/goal_pose", self.on_goal_pose, 10)
        self.create_subscription(Bool, "/a2b/enable", self.on_enable, 10)
        self.create_subscription(Path, "/a2b/waypoints", self.on_waypoints, 10)
        self.create_subscription(Bool, "/sim/reset", self.on_reset, 10)        # viewer "Reset truck"
        if self.obstacle_on:
            self.create_subscription(PointCloud2, g("obstacle_cloud_topic"), self.on_cloud,
                                     qos_profile_sensor_data)

        self.pub_ctrl = self.create_publisher(ControlCmd, g("control_topic"), RELIABLE)
        self.pub_debug = self.create_publisher(ControlDebug, g("control_debug_topic"), RELIABLE)
        self.pub_pwm = self.create_publisher(Float32MultiArray, "/pwm_duty", 10) if g("publish_pwm") else None
        self.pub_path = self.create_publisher(Path, "/a2b/path", 1)
        self.pub_local = self.create_publisher(Path, "/a2b/local_path", 1)
        self.pub_status = self.create_publisher(String, "/a2b/status", 10)

        gx, gy = g("goal_x"), g("goal_y")
        if not (math.isnan(gx) or math.isnan(gy)):
            self.goal = (gx, gy)
            self.goal_from_param = True
        wp = list(g("waypoints"))
        self.waypoints = [(wp[i], wp[i + 1]) for i in range(0, len(wp) - 1, 2)] if len(wp) >= 2 else []
        self.wp_relative = self.goal_relative      # waypoints from the parameter follow goal_relative
        self.auto_start = g("auto_start")

        self.publish_stop()
        self.timer = self.create_timer(1.0 / g("rate_hz"), self.tick)
        self.get_logger().info(
            f"A->B planner ready. loc={g('localization_topic')} goal={g('plan_target_topic')} "
            f"task={g('task_state_topic')} control={g('control_topic')} fallback goal={self.goal} "
            f"(relative={self.goal_relative}), waypoints={self.waypoints}, obstacle stop="
            f"{g('obstacle_cloud_topic') if self.obstacle_on else 'off'}. Waiting for task_state START_TASK or /a2b/enable.")

    # ---- callbacks -------------------------------------------------------------
    def on_loc(self, msg):
        if not msg.is_valid:
            return  # stale pose -> localization timeout stops the truck
        self.pose = (msg.x, msg.y, math.radians(msg.yaw))
        self.speed = abs(msg.speed)
        self.pose_t = time.monotonic()
        if self.auto_start and self.goal and self.ctl.state == "IDLE":
            self.start("auto_start")

    def on_plan_target(self, msg):
        new_goal = (msg.goal_x, msg.goal_y)
        if self.goal_from_param or self.goal is None or math.hypot(new_goal[0] - self.goal[0],
                                                                   new_goal[1] - self.goal[1]) > 0.05:
            self.goal, self.goal_from_param = new_goal, False
            self.get_logger().info(f"plan_target: B = ({new_goal[0]:.2f}, {new_goal[1]:.2f})")
            if self.ctl.state == "DRIVING":
                self.start("new plan_target")

    def on_task_state(self, msg):
        prev, self.task_state = self.task_state, msg.state
        if msg.state == prev:
            return
        self.get_logger().info(f"task_state {prev} -> {msg.state} (stop_mode {msg.stop_mode})")
        if msg.state in START_STATES and self.ctl.state != "DRIVING":
            self.start("task_state START_TASK")
        elif msg.state in STOP_STATES:
            self.ctl.stop(f"task_state {msg.state}")

    def on_goal_pose(self, msg):
        self.goal, self.goal_from_param = (msg.pose.position.x, msg.pose.position.y), False
        self.get_logger().info(f"RViz goal B = ({self.goal[0]:.2f}, {self.goal[1]:.2f})")
        if self.ctl.state == "DRIVING":
            self.start("new RViz goal")

    def on_waypoints(self, msg):
        self.waypoints = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.wp_relative = False
        self.get_logger().info(f"waypoints: {[(round(x, 2), round(y, 2)) for x, y in self.waypoints]}")
        if self.ctl.state in ("DRIVING", "BLOCKED", "BACKING"):
            self.start("new waypoints")

    def on_cloud(self, msg):
        try:
            n = msg.width * msg.height
            offs = {f.name: f.offset for f in msg.fields}
            if n == 0:
                xyz = np.zeros((0, 3))
            else:
                rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, msg.point_step)
                xyz = np.stack([rows[:, offs[k]:offs[k] + 4].copy().view(np.float32)[:, 0] for k in "xyz"], 1)
                xyz = xyz[np.isfinite(xyz).all(1)].astype(np.float64)
                xyz[:, 0] += self.lidar_x                     # lidar frame -> rear-axle frame
            self.cloud_xyz, self.cloud_t = xyz, time.monotonic()
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"obstacle cloud decode failed: {e}")

    def nearest_obstacle(self, now):
        """Distance (m) to the closest obstacle in the driving corridor, inf if none,
        -1 if obstacle detection is on but the cloud is missing or too old."""
        if not self.obstacle_on:
            return float("inf")
        if self.cloud_xyz is None or now - self.cloud_t > self.cloud_timeout:
            return -1.0
        p = self.params
        x, y, z = self.cloud_xyz[:, 0], self.cloud_xyz[:, 1], self.cloud_xyz[:, 2]
        steer = (self.ctl.steer_duty - NEUTRAL) / 2.5 * p.steer_sign * math.radians(p.max_steer_deg)
        k = math.tan(steer) / p.wheelbase if p.wheelbase > 0 else 0.0
        y_path = 0.5 * k * x * x                                 # corridor bends with the steering
        hit = (x > 0.0) & (x <= p.obstacle_slow_dist) & (z >= p.obstacle_z_min) & (z <= p.obstacle_z_max) \
            & (np.abs(y - y_path) <= p.corridor_half_width)
        xs = np.sort(x[hit])
        return float(xs[p.obstacle_min_points - 1]) if len(xs) >= p.obstacle_min_points else float("inf")

    def obstacle_points_map(self):
        """Obstacle points (N, 2) in the map frame for avoidance, or None (avoidance off / no data)."""
        p = self.params
        if not (self.obstacle_on and p.avoid_enable) or self.pose is None or self.cloud_xyz is None:
            return None
        xyz = self.cloud_xyz
        xyz = xyz[(xyz[:, 2] >= p.obstacle_z_min) & (xyz[:, 2] <= p.obstacle_z_max)]
        x, y, yaw = self.pose
        c, s = math.cos(yaw), math.sin(yaw)
        return np.stack([x + c * xyz[:, 0] - s * xyz[:, 1], y + s * xyz[:, 0] + c * xyz[:, 1]], 1)

    def publish_local_path(self):
        path = Path()
        path.header.frame_id = self.map_frame
        path.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.ctl.avoider.path:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_local.publish(path)

    def on_reset(self, msg):
        if not msg.data:
            return
        self.ctl.stop("reset")
        self.ctl.state, self.ctl.route, self.ctl.stop_reason = "IDLE", None, ""
        self.ctl.avoider.reset()
        self.pose = None                    # wait for a fresh position after the jump
        self.publish_stop()
        path = Path()
        path.header.frame_id = self.map_frame
        self.pub_path.publish(path)
        self.pub_local.publish(path)
        self.get_logger().info("reset: stopped, waiting for Start")

    def on_enable(self, msg):
        if msg.data:
            self.start("/a2b/enable")
        else:
            self.ctl.stop("disabled by /a2b/enable")

    # ---- logic -----------------------------------------------------------------
    def start(self, why):
        if self.pose is None:
            self.get_logger().warn(f"{why}: no valid localization yet, cannot start")
            return
        if self.goal is None:
            self.get_logger().warn(f"{why}: no goal B (plan_target / goal_x,goal_y), cannot start")
            return
        x, y, yaw = self.pose
        goal, wps = self.goal, self.waypoints
        c, s = math.cos(yaw), math.sin(yaw)
        rel = lambda p: (x + c * p[0] - s * p[1], y + s * p[0] + c * p[1])  # noqa: E731
        if self.goal_relative and self.goal_from_param:
            goal = rel(goal)
        if self.wp_relative:
            wps = [rel(p) for p in wps]
        self.ctl.set_route([(x, y)] + wps + [goal])
        self.publish_path()
        via = f" via {len(wps)} waypoint(s)" if wps else ""
        self.get_logger().info(f"{why}: driving A({x:.2f}, {y:.2f}) -> B({goal[0]:.2f}, {goal[1]:.2f}){via}, "
                               f"route {self.ctl.route.length:.1f} m")

    def tick(self):
        now = time.monotonic()
        dt = min(now - self.last_tick, 0.2)
        self.last_tick = now
        loc_age = now - self.pose_t if self.pose else 1e9
        # A START_TASK that arrived before localization or plan_target was ready: start as soon
        # as both are there (Simmoulde in PncSim mode publishes task_state from the first second).
        if (self.task_state in START_STATES and self.ctl.state == "IDLE"
                and self.pose is not None and loc_age < self.params.loc_timeout and self.goal is not None):
            self.start("task_state START_TASK (now ready)")
        self.obstacle_dist = self.nearest_obstacle(now)
        cmd = self.ctl.step(self.pose, self.speed, dt, self.obstacle_dist, loc_age,
                            obstacle_points=self.obstacle_points_map())
        if self.ctl.avoiding:
            self.publish_local_path()
            if cmd.state == "DRIVING":
                self.obstacle_dist = cmd.info.get("obstacle", self.obstacle_dist)
        # compare with the state published last tick, so stops set by callbacks
        # (task_state, /a2b/enable) between ticks are logged too
        if cmd.state != self.last_state:
            self.get_logger().info(f"{self.last_state} -> {cmd.state} {self.ctl.stop_reason}")
            self.last_state = cmd.state

        ctrl = self.to_control_cmd(cmd)
        self.pub_ctrl.publish(ctrl)
        self.pub_debug.publish(self.to_control_debug(cmd, ctrl))
        if self.pub_pwm:
            self.pub_pwm.publish(Float32MultiArray(data=[float(d) for d in cmd.pwm]))
        obs = "off" if not self.obstacle_on else ("none" if math.isinf(self.obstacle_dist)
                                                   else f"{self.obstacle_dist:.2f}")
        self.pub_status.publish(String(data=f"{cmd.state} {self.ctl.stop_reason} {cmd.info} obstacle_ahead={obs}"))

    def to_control_cmd(self, cmd):
        """Truck command (PWM duties) -> robominer_msgs/ControlCmd for the simulation."""
        p = self.params
        m = ControlCmd()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
        m.target_drive_mode = DriveMode(mode=DriveMode.AUTO)
        # steering: PWM1 duty -> normalised (+1 = left) -> degrees, left positive
        left = (cmd.pwm[0] - NEUTRAL) / 2.5 * p.steer_sign
        m.target_steering = float(max(-1.0, min(1.0, left)) * self.sim_max_steer)
        if cmd.state == "DRIVING":
            m.target_gear = Gear(gear=Gear.D)
            m.target_throttle = float(cmd.throttle * 100.0)
            m.target_brake = 0.0
            m.target_speed = float(cmd.info.get("v_ref", 0.0))
        else:
            m.target_gear = Gear(gear=Gear.D if self.speed > 0.05 else Gear.N)
            m.target_throttle = 0.0
            m.target_brake = float(self.stop_brake)
            m.target_speed = 0.0
        return m

    def to_control_debug(self, cmd, ctrl):
        """/auto/control_debug for Simmoulde. Simmoulde only forwards /auto/control_cmd to SimOne
        when it also has a ControlDebug, and in PncSim mode it takes the gear from
        traj_ctl_debug.target_gear (and, with [DriverMode] driver_mode=1, the acceleration
        from traj_ctl_debug.total_accel_cmd)."""
        d = ControlDebug()
        d.header.stamp = ctrl.header.stamp
        d.header.frame_id = "base_link"
        t = d.traj_ctl_debug
        t.target_gear = int(ctrl.target_gear.gear)
        if cmd.state == "DRIVING":
            v_ref = float(cmd.info.get("v_ref", 0.0))
            t.target_velocity = v_ref
            t.velocity_error = v_ref - self.speed
            t.target_accel = t.total_accel_cmd = float(max(-self.max_decel, min(self.max_accel,
                                                       self.accel_gain * (v_ref - self.speed))))
            t.go_req, t.stop_req = True, False
        else:
            t.target_velocity = 0.0
            t.velocity_error = -self.speed
            t.target_accel = t.total_accel_cmd = float(-self.max_decel if self.speed > 0.05 else 0.0)
            t.go_req, t.stop_req = False, True
        if self.pose and self.ctl.route:
            t.target_position_x, t.target_position_y = (float(v) for v in self.ctl.route.goal)
        return d

    def publish_path(self):
        path = Path()
        path.header.frame_id = self.map_frame
        path.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.ctl.route.pts:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = x, y
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_path.publish(path)

    def publish_stop(self):
        stop = ControlCmd()
        stop.header.stamp = self.get_clock().now().to_msg()
        stop.target_drive_mode = DriveMode(mode=DriveMode.AUTO)
        stop.target_gear = Gear(gear=Gear.N)
        stop.target_brake = float(self.stop_brake)
        self.pub_ctrl.publish(stop)
        if self.pub_pwm:
            self.pub_pwm.publish(Float32MultiArray(data=[NEUTRAL, NEUTRAL, NEUTRAL]))


def main():
    # Handle Ctrl+C ourselves (no rclpy signal handler), so the ROS context is still
    # alive in `finally` and the stop command really goes out on shutdown.
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init()
    node = A2BPlanner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.ctl.stop("shutdown")
        try:
            for _ in range(3):  # make sure a stop goes out on exit
                node.publish_stop()
                time.sleep(0.05)
        except Exception:  # context already gone; receivers time out to stop anyway
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
