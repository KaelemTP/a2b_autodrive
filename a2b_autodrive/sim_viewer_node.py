#!/usr/bin/env python3
"""
sim_viewer_node  -  live 3D view of the A->B simulation in a web browser (Flask + three.js).

Open http://localhost:5000 (from Windows too: WSL forwards localhost).

Shows the room, the truck (true pose) and the lidar estimate, the 3D lidar scan, the route,
the path actually followed (with detours around obstacles), B, waypoints, planner state,
speed and PWM duties. Drag to orbit, right-drag to pan, scroll to zoom.

"Reset truck" stops it and puts it back at its start (/sim/reset: truck_sim jumps back,
lidar_loc_bridge resets KISS-ICP). Obstacles and B stay.

Clicking the ground does what the selected tool says:
    Set B      -> /goal_pose
    Waypoint   -> /a2b/waypoints (nav_msgs/Path), also Shift+click
    Box / Cylinder -> adds an obstacle, published on /sim/obstacles (std_msgs/String, JSON,
                  latched); fake_lidar puts it in the scan, so the planner has to avoid it
    Remove     -> removes the nearest placed obstacle
The viewer also counts CONTACTS: the true truck body touching a wall, box, pillar or placed
obstacle. It should stay 0.

Install Flask once:  sudo apt install python3-flask
The page and three.js are served from this package (share/a2b_autodrive/web), no internet needed.
"""
import json
import logging
import math
import os
import re
import threading
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Bool, Float32MultiArray, String
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from robominer_msgs.msg import LocalizationData

try:
    from flask import Flask, jsonify, request, Response, send_from_directory
except ImportError:  # pragma: no cover
    Flask = None

BE = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
LATCHED = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)

# Truck body for the contact check and the 3D model, metres from the rear axle (measure the truck!)
TRUCK_REAR, TRUCK_FRONT, TRUCK_HALF_WIDTH = -0.10, 0.45, 0.20


def web_dir():
    try:
        from ament_index_python.packages import get_package_share_directory
        d = os.path.join(get_package_share_directory("a2b_autodrive"), "web")
        if os.path.isdir(d):
            return d
    except Exception:  # noqa: BLE001
        pass
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web"))  # source tree


def truck_outline(x, y, yaw, step=0.05):
    """Points around the edge of the truck body (map frame)."""
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for fx in np.arange(TRUCK_REAR, TRUCK_FRONT + 1e-9, step):
        pts += [(fx, -TRUCK_HALF_WIDTH), (fx, TRUCK_HALF_WIDTH)]
    for fy in np.arange(-TRUCK_HALF_WIDTH, TRUCK_HALF_WIDTH + 1e-9, step):
        pts += [(TRUCK_REAR, fy), (TRUCK_FRONT, fy)]
    p = np.array(pts)
    return np.stack([x + c * p[:, 0] - s * p[:, 1], y + s * p[:, 0] + c * p[:, 1]], 1)


def touching(outline, walls, boxes, circles):
    """True if any outline point is inside a box / circle or within 3 cm of a wall line."""
    px, py = outline[:, 0], outline[:, 1]
    for x0, y0, x1, y1 in boxes:
        if ((px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)).any():
            return True
    for cx, cy, r in circles:
        if (np.hypot(px - cx, py - cy) <= r).any():
            return True
    for x0, y0, x1, y1 in walls:
        ex, ey = x1 - x0, y1 - y0
        L2 = ex * ex + ey * ey or 1.0
        t = np.clip(((px - x0) * ex + (py - y0) * ey) / L2, 0, 1)
        if (np.hypot(px - (x0 + t * ex), py - (y0 + t * ey)) < 0.03).any():
            return True
    return False


class SimViewer(Node):
    def __init__(self):
        super().__init__("sim_viewer")
        self.declare_parameter("port", 5000)
        self.declare_parameter("truth_topic", "/sim/ground_truth")
        self.declare_parameter("estimate_topic", "/localization/localization_data")
        self.declare_parameter("points_topic", "/fake_lidar/points")
        self.declare_parameter("show_room", True)
        self.declare_parameter("lidar_x_offset", 0.25)
        self.declare_parameter("lidar_height", 0.4)
        self.declare_parameter("scan_max_points", 3000)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.port = int(g("port"))
        self.lidar_on = bool(g("points_topic"))     # placed obstacles only matter when a lidar sees them
        self.off = g("lidar_x_offset")
        self.lidar_h = g("lidar_height")
        self.max_pts = int(g("scan_max_points"))
        self.lock = threading.Lock()
        self.truth = None
        self.est = None
        self.truth_trail = deque(maxlen=4000)
        self.est_trail = deque(maxlen=4000)
        self.route = []
        self.local_path = []
        self.status = "IDLE"
        self.reason = ""
        self.info = {}
        self.pwm = [7.5, 7.5, 7.5]
        self.scan = np.zeros((0, 3))
        self.goal = None
        self.waypoints = []
        self.obstacle = None
        self.obstacles = []        # placed by the user: dicts kind/x/y/size/height
        self.contacts = 0
        self.touching = False
        self.t0 = time.monotonic()

        self.room = None
        if g("show_room"):
            try:
                from a2b_autodrive import fake_lidar_node as fl
                self.room = {"walls": [list(w) for w in fl.WALLS],
                             "boxes": [list(b) for b in fl.BOXES],
                             "pillars": [list(p) for p in fl.PILLARS]}
            except Exception:  # noqa: BLE001
                self.room = None

        self.create_subscription(LocalizationData, g("truth_topic"), self.on_truth, BE)
        if g("estimate_topic") != g("truth_topic"):
            self.create_subscription(LocalizationData, g("estimate_topic"), self.on_est, BE)
        self.same_topic = g("estimate_topic") == g("truth_topic")
        self.create_subscription(Path, "/a2b/path", self.on_path, 1)
        self.create_subscription(Path, "/a2b/local_path", self.on_local_path, 1)
        self.create_subscription(String, "/a2b/status", self.on_status, 10)
        self.create_subscription(Float32MultiArray, "/pwm_duty", self.on_pwm, 10)
        if g("points_topic"):
            self.create_subscription(PointCloud2, g("points_topic"), self.on_points, BE)
        self.pub_enable = self.create_publisher(Bool, "/a2b/enable", 10)
        self.pub_goal = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.pub_wps = self.create_publisher(Path, "/a2b/waypoints", 10)
        self.pub_obs = self.create_publisher(String, "/sim/obstacles", LATCHED)
        self.pub_reset = self.create_publisher(Bool, "/sim/reset", 10)
        self._publish_obstacles()

    # ---- ROS callbacks ---------------------------------------------------------
    def on_truth(self, m):
        with self.lock:
            self.truth = (m.x, m.y, m.yaw, m.speed)
            self._trail(self.truth_trail, m.x, m.y)
        self._check_contact(m.x, m.y, math.radians(m.yaw))

    def on_est(self, m):
        if not m.is_valid:
            return
        with self.lock:
            self.est = (m.x, m.y, m.yaw, m.speed)
            self._trail(self.est_trail, m.x, m.y)

    @staticmethod
    def _trail(trail, x, y):
        if not trail or math.hypot(trail[-1][0] - x, trail[-1][1] - y) > 0.05:
            trail.append((round(x, 3), round(y, 3)))

    def _check_contact(self, x, y, yaw):
        with self.lock:
            walls = [w[:4] for w in self.room["walls"]] if self.room else []
            boxes = [b[:4] for b in self.room["boxes"]] if self.room else []
            circles = [p[:3] for p in self.room["pillars"]] if self.room else []
            for o in self.obstacles:
                h = o["size"] / 2
                if o["kind"] == "cylinder":
                    circles.append((o["x"], o["y"], h))
                else:
                    boxes.append((o["x"] - h, o["y"] - h, o["x"] + h, o["y"] + h))
        now = touching(truck_outline(x, y, yaw), walls, boxes, circles)
        with self.lock:
            if now and not self.touching:
                self.contacts += 1
                self.get_logger().warn(f"CONTACT: truck body touches an obstacle at ({x:.2f}, {y:.2f})")
            self.touching = now

    @staticmethod
    def _path_pts(msg, every=1):
        pts = [(round(p.pose.position.x, 3), round(p.pose.position.y, 3)) for p in msg.poses]
        return (pts[::every] + ([pts[-1]] if pts and every > 1 else [])) if pts else []

    def on_path(self, msg):
        pts = self._path_pts(msg, 5)
        with self.lock:
            self.route = pts
            if pts:
                self.goal = pts[-1]

    def on_local_path(self, msg):
        with self.lock:
            self.local_path = self._path_pts(msg)

    def on_status(self, msg):
        s = msg.data
        state = s.split(" ", 1)[0]
        info = {}
        for key in ("remaining", "v_ref", "cte", "avoid"):
            mm = re.search(rf"'{key}': (-?[0-9.]+)", s)
            if mm:
                info[key] = float(mm.group(1))
        mm = re.search(r"'reason': '([^']*)'", s)
        mo = re.search(r"obstacle_ahead=(\S+)", s)
        if "'avoid': 'search'" in s:
            info["avoid"] = "search"
        with self.lock:
            self.status, self.info = state, info
            self.reason = mm.group(1) if mm else ""
            self.obstacle = mo.group(1) if mo else None
            if state in ("IDLE", "ARRIVED", "STOPPED"):
                self.local_path = []

    def on_pwm(self, msg):
        if len(msg.data) >= 3:
            with self.lock:
                self.pwm = [round(float(v), 2) for v in msg.data[:3]]

    def on_points(self, msg):
        try:
            offs = {f.name: f.offset for f in msg.fields}
            n = msg.width * msg.height
            if n == 0 or not {"x", "y", "z"} <= offs.keys():
                return
            buf = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, msg.point_step)
            xyz = np.stack([buf[:, offs[k]:offs[k] + 4].copy().view(np.float32)[:, 0] for k in "xyz"], 1)
            xyz = xyz[np.isfinite(xyz).all(1)]
            xyz = xyz[xyz[:, 2] > -(self.lidar_h - 0.1)]          # drop the floor for a clear picture
            if len(xyz) > self.max_pts:
                xyz = xyz[np.linspace(0, len(xyz) - 1, self.max_pts).astype(int)]
            with self.lock:
                self.scan = xyz.astype(np.float64)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"point cloud decode failed: {e}")

    # ---- web state -------------------------------------------------------------
    def snapshot(self):
        with self.lock:
            truth = self.truth
            est = self.est if not self.same_topic else None
            # draw the scan around the lidar-based pose (shows how well the estimate fits the room)
            base = est or truth
            scan_world = []
            if base is not None and len(self.scan):
                x, y, yaw = base[0], base[1], math.radians(base[2])
                c, s = math.cos(yaw), math.sin(yaw)
                lx, ly = x + c * self.off, y + s * self.off
                px, py, pz = self.scan[:, 0], self.scan[:, 1], self.scan[:, 2]
                w = np.stack([lx + c * px - s * py, ly + s * px + c * py, pz + self.lidar_h], 1)
                scan_world = np.round(w, 2).ravel().tolist()      # flat [x, y, z, x, y, z, ...]
            err = None
            if truth and est:
                err = math.hypot(truth[0] - est[0], truth[1] - est[1])
            return {
                "t": round(time.monotonic() - self.t0, 1),
                "truth": truth, "est": est, "error": err,
                "truth_trail": list(self.truth_trail), "est_trail": list(self.est_trail),
                "route": self.route, "local_path": self.local_path, "goal": self.goal, "scan": scan_world,
                "status": self.status, "reason": self.reason, "info": self.info, "pwm": self.pwm,
                "room": self.room, "waypoints": list(self.waypoints), "obstacle": self.obstacle,
                "obstacles": list(self.obstacles), "contacts": self.contacts, "touching": self.touching,
                "truck": {"rear": TRUCK_REAR, "front": TRUCK_FRONT, "half_width": TRUCK_HALF_WIDTH},
                "lidar": self.lidar_on,
            }

    def start(self):
        self.pub_enable.publish(Bool(data=True))

    def stop(self):
        self.pub_enable.publish(Bool(data=False))

    def set_goal(self, x, y):
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = "odom"
        ps.pose.position.x, ps.pose.position.y = float(x), float(y)
        ps.pose.orientation.w = 1.0
        self.pub_goal.publish(ps)
        with self.lock:
            self.goal = (float(x), float(y))

    def _publish_waypoints(self):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "odom"
        for x, y in self.waypoints:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_wps.publish(path)

    def add_waypoint(self, x, y):
        with self.lock:
            self.waypoints.append((round(float(x), 2), round(float(y), 2)))
        self._publish_waypoints()

    def clear_waypoints(self):
        with self.lock:
            self.waypoints = []
        self._publish_waypoints()

    def _publish_obstacles(self):
        with self.lock:
            data = json.dumps(self.obstacles)
        self.pub_obs.publish(String(data=data))

    def add_obstacle(self, x, y, kind="box", size=None, height=None):
        kind = "cylinder" if kind == "cylinder" else "box"
        size = float(size) if size else (0.5 if kind == "cylinder" else 0.6)
        height = float(height) if height else (1.7 if kind == "cylinder" else 0.8)
        o = {"kind": kind, "x": round(float(x), 2), "y": round(float(y), 2),
             "size": round(max(0.1, min(size, 5.0)), 2), "height": round(height, 2)}
        with self.lock:
            self.obstacles.append(o)
        self._publish_obstacles()
        self.get_logger().info(f"added {kind} obstacle at ({o['x']}, {o['y']}), size {o['size']} m")

    def remove_obstacle(self, x, y):
        with self.lock:
            if not self.obstacles:
                return
            k = min(range(len(self.obstacles)),
                    key=lambda i: math.hypot(self.obstacles[i]["x"] - x, self.obstacles[i]["y"] - y))
            o = self.obstacles[k]
            if math.hypot(o["x"] - x, o["y"] - y) > max(1.0, o["size"]):
                return
            self.obstacles.pop(k)
        self._publish_obstacles()

    def clear_obstacles(self):
        with self.lock:
            self.obstacles = []
        self._publish_obstacles()

    def reset_truck(self):
        """Stop, put the fake truck back at its start, reset the lidar odometry, clear the view."""
        self.pub_enable.publish(Bool(data=False))
        self.pub_reset.publish(Bool(data=True))
        with self.lock:
            self.route, self.local_path = [], []
            self.truth = self.est = None
        self.clear_trails()

    def clear_trails(self):
        with self.lock:
            self.truth_trail.clear()
            self.est_trail.clear()
            self.contacts = 0


def make_app(node):
    app = Flask(__name__, static_folder=None)   # /static is served from web_dir() below
    wd = web_dir()

    @app.get("/")
    def index():
        return send_from_directory(wd, "index.html")

    @app.get("/static/<path:name>")
    def static_file(name):
        resp = send_from_directory(wd, name)
        if name.endswith(".js"):
            resp.mimetype = "text/javascript"
        return resp

    @app.get("/state")
    def state():
        return jsonify(node.snapshot())

    @app.post("/cmd/<name>")
    def cmd(name):
        d = request.get_json(force=True, silent=True) or {}
        if name == "start":
            node.start()
        elif name == "stop":
            node.stop()
        elif name == "clear":
            node.clear_trails()
        elif name == "reset":
            node.reset_truck()
        elif name == "waypoint":
            node.add_waypoint(d["x"], d["y"])
        elif name == "clear_waypoints":
            node.clear_waypoints()
        elif name == "goal":
            node.set_goal(d["x"], d["y"])
        elif name == "obstacle":
            node.add_obstacle(d["x"], d["y"], d.get("kind", "box"), d.get("size"), d.get("height"))
        elif name == "remove_obstacle":
            node.remove_obstacle(float(d["x"]), float(d["y"]))
        elif name == "clear_obstacles":
            node.clear_obstacles()
        else:
            return jsonify(ok=False), 404
        return jsonify(ok=True)

    return app


def main():
    rclpy.init()
    node = SimViewer()
    if Flask is None:
        node.get_logger().error("Flask is not installed: sudo apt install python3-flask")
        node.destroy_node()
        rclpy.shutdown()
        return
    app = make_app(node)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)   # no log line per page update
    web = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=node.port, threaded=True,
                                                  use_reloader=False), daemon=True)
    web.start()
    node.get_logger().info(f"live 3D view: http://localhost:{node.port}  (page files: {web_dir()})")
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
