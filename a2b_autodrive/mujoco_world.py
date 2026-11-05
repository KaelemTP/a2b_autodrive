"""
a2b_autodrive.mujoco_world  -  MuJoCo yard with the mini dump truck and a ray-cast 3D lidar.

ROS-free, so the same world runs in the ROS node (mujoco_sim_node) and the offline test
(test/mujoco_lidar_sim.py).

    PWM [steer, bucket, drive] -> planar force model of the truck (MuJoCo physics, collisions)
    lidar on the cab roof      -> mj_multiRay against every geom in the yard except the truck
                                  -> points in the lidar's own frame, like a real driver gives

The truck's body origin is the rear axle (the planner's reference point). World frame: x east,
y north, z up; the truck starts at (start_x, start_y) facing start_yaw.
"""
import math

import numpy as np
import mujoco

from a2b_autodrive.core import NEUTRAL, Params, clamp

# Lidar on the truck, measured from the rear axle at ground level
LIDAR_X = 0.25
LIDAR_Z = 0.40

# Yard layout (world frame, metres). Everything here is visible to the lidar.
# Concrete barrier segments: (x1, y1, x2, y2)
BARRIERS = [(-5, -8, 17, -8), (17, -8, 17, 8), (17, 8, -5, 8), (-5, 8, -5, -8)]
# Boulders: (x, y, rx, ry, rz)
BOULDERS = [(2.5, -4.5, 0.6, 0.5, 0.45), (6.5, -3.2, 0.35, 0.4, 0.3), (9.0, -5.8, 0.7, 0.6, 0.5),
            (-2.5, 4.5, 0.5, 0.6, 0.4), (3.5, 5.8, 0.45, 0.4, 0.35), (13.5, 1.0, 0.4, 0.45, 0.35),
            (-3.0, -5.0, 0.5, 0.4, 0.4), (15.0, -2.5, 0.55, 0.5, 0.4)]
# Light poles: (x, y)
POLES = [(1.0, 7.0), (8.0, -7.2), (15.8, 6.5), (-4.2, -1.0)]
# Site container: (cx, cy, half_x, half_y, half_z)
CONTAINER = (4.0, -6.7, 1.3, 0.6, 0.65)
# Material pile: (cx, cy, rx, ry, rz)
PILE = (13.5, -5.2, 2.0, 1.4, 0.9)
# Excavator: (x, y, yaw_deg)
EXCAVATOR = (11.5, 6.2, -120.0)
# Orange dump pad (flat, only a visual target)
PAD = (8.0, 3.5)


def _barriers_xml():
    out = []
    for i, (x1, y1, x2, y2) in enumerate(BARRIERS):
        length = math.hypot(x2 - x1, y2 - y1)
        yaw = math.degrees(math.atan2(y2 - y1, x2 - x1))
        n = int(length / 2.0)                                  # 2 m barrier blocks, small gaps
        for k in range(n):
            t = (k + 0.5) / n
            cx, cy = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
            out.append(f'<geom name="barrier{i}_{k}" type="box" pos="{cx:.3f} {cy:.3f} 0.4" '
                       f'size="{length / n / 2 - 0.05:.3f} 0.2 0.4" euler="0 0 {yaw:.1f}" material="concrete"/>')
    return "\n".join(out)


def _yard_xml():
    g = [_barriers_xml()]
    for i, (x, y, rx, ry, rz) in enumerate(BOULDERS):
        g.append(f'<geom name="boulder{i}" type="ellipsoid" pos="{x} {y} {rz * 0.7:.2f}" '
                 f'size="{rx} {ry} {rz}" euler="0 0 {i * 37}" material="rock"/>')
    for i, (x, y) in enumerate(POLES):
        g.append(f'<geom name="pole{i}" type="cylinder" pos="{x} {y} 1.5" size="0.08 1.5" material="steel"/>')
        g.append(f'<geom name="lamp{i}" type="box" pos="{x} {y} 3.0" size="0.25 0.12 0.06" material="steel"/>')
    cx, cy, hx, hy, hz = CONTAINER
    g.append(f'<geom name="container" type="box" pos="{cx} {cy} {hz}" size="{hx} {hy} {hz}" material="container"/>')
    px, py, rx, ry, rz = PILE
    g.append(f'<geom name="pile" type="ellipsoid" pos="{px} {py} 0" size="{rx} {ry} {rz}" material="dirt"/>')
    g.append(f'<geom name="pad" type="box" pos="{PAD[0]} {PAD[1]} 0.01" size="0.6 0.6 0.01" '
             f'rgba="0.85 0.5 0.25 1" contype="0" conaffinity="0"/>')
    ex, ey, eyaw = EXCAVATOR
    g.append(f"""
    <body name="excavator" pos="{ex} {ey} 0" euler="0 0 {eyaw}">
      <geom type="box" pos="0  0.42 0.15" size="0.75 0.15 0.15" material="track"/>
      <geom type="box" pos="0 -0.42 0.15" size="0.75 0.15 0.15" material="track"/>
      <geom type="box" pos="-0.1 0 0.45" size="0.55 0.45 0.15" material="yellow"/>
      <geom type="box" pos="0.15 0.22 0.8" size="0.25 0.2 0.22" material="cabin"/>
      <geom type="capsule" fromto="0.35 -0.1 0.55  1.3 -0.1 1.45" size="0.08" material="yellow"/>
      <geom type="capsule" fromto="1.3 -0.1 1.45  2.0 -0.1 0.7" size="0.07" material="yellow"/>
      <geom type="box" pos="2.05 -0.1 0.45" size="0.18 0.25 0.18" material="steel"/>
    </body>""")
    return "\n".join(g)


def build_mjcf(start_x=0.0, start_y=0.0, start_yaw=0.0):
    """MJCF string of the yard with the truck at the given start pose (yaw in rad)."""
    return f"""
<mujoco model="a2b_yard">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="2048" numslices="8" numstacks="8"/>  <!-- low-poly: lidar dots render ~10x faster -->
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.36 0.42 0.50" rgb2="0.30 0.36 0.44"/>
    <material name="floor" texture="grid" texrepeat="22 16" reflectance="0.05"/>
    <material name="teal" rgba="0.13 0.62 0.68 1"/>
    <material name="dark" rgba="0.12 0.13 0.15 1"/>
    <material name="tyre" rgba="0.05 0.05 0.05 1"/>
    <material name="concrete" rgba="0.62 0.62 0.60 1"/>
    <material name="rock" rgba="0.48 0.42 0.36 1"/>
    <material name="dirt" rgba="0.55 0.44 0.32 1"/>
    <material name="steel" rgba="0.45 0.47 0.50 1"/>
    <material name="container" rgba="0.70 0.25 0.20 1"/>
    <material name="yellow" rgba="0.95 0.72 0.10 1"/>
    <material name="track" rgba="0.10 0.10 0.10 1"/>
    <material name="cabin" rgba="0.20 0.25 0.32 1"/>
    <material name="lidar" rgba="0.85 0.85 0.88 1"/>
  </asset>
  <worldbody>
    <light pos="4 -3 12" dir="-0.2 0.3 -1" directional="true" castshadow="true" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" pos="6 0 0" size="11 8 0.1" material="floor"/>
    {_yard_xml()}
    <body name="truck" pos="0 0 0">
      <joint name="tx" type="slide" axis="1 0 0" damping="0"/>
      <joint name="ty" type="slide" axis="0 1 0" damping="0"/>
      <joint name="rz" type="hinge" axis="0 0 1" damping="0"/>
      <inertial pos="0.15 0 0.1" mass="4.0" diaginertia="0.04 0.06 0.06"/>
      <geom name="chassis" type="box" pos="0.15 0 0.08" size="0.27 0.13 0.03" material="dark"/>
      <geom type="box" pos="0.06 0 0.13" size="0.19 0.15 0.01" material="teal" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.06  0.15 0.18" size="0.19 0.01 0.05" material="teal" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.06 -0.15 0.18" size="0.19 0.01 0.05" material="teal" contype="0" conaffinity="0"/>
      <geom type="box" pos="-0.13 0 0.18" size="0.01 0.15 0.05" material="teal" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.25 0 0.18" size="0.01 0.15 0.05" material="teal" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.34 0 0.17" size="0.07 0.13 0.08" material="teal" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0 0.16 0.05" size="0.05 0.025" euler="90 0 0" material="tyre" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0 -0.16 0.05" size="0.05 0.025" euler="90 0 0" material="tyre" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.30 0.16 0.05" size="0.05 0.025" euler="90 0 0" material="tyre" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.30 -0.16 0.05" size="0.05 0.025" euler="90 0 0" material="tyre" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="{LIDAR_X} 0 0.30" size="0.012 0.06" material="steel" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="{LIDAR_X} 0 {LIDAR_Z - 0.02}" size="0.045 0.025" material="lidar" contype="0" conaffinity="0"/>
      <site name="lidar" pos="{LIDAR_X} 0 {LIDAR_Z}" size="0.01"/>
    </body>
    <camera name="chase" mode="trackcom" pos="-3.5 -3.5 3.5" xyaxes="0.7 -0.7 0 0.4 0.4 0.8"/>
  </worldbody>
  <keyframe>
    <key name="start" qpos="{start_x} {start_y} {start_yaw}"/>
  </keyframe>
</mujoco>
"""


def lidar_directions(n_az=720, elevations_deg=None):
    """Unit ray directions (N,3) in the lidar frame, ring by ring. Default: 24 rings -25..60 deg
    (wide vertical view like the Airy)."""
    if elevations_deg is None:
        elevations_deg = np.linspace(-25.0, 60.0, 24)
    el = np.radians(np.asarray(elevations_deg))[:, None]
    az = np.linspace(-math.pi, math.pi, n_az, endpoint=False)[None, :]
    d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el) * np.ones_like(az)], -1)
    return d.reshape(-1, 3)


class MujocoTruckWorld:
    """The yard + truck. step(pwm, dt) drives it, scan() returns one lidar sweep."""

    def __init__(self, params: Params = None, start=(0.0, 0.0, 0.0), top_speed=2.0, tau=0.5,
                 lidar_noise=0.01, max_range=40.0, n_az=720, elevations_deg=None, seed=1):
        self.p = params or Params()
        self.top_speed, self.tau = top_speed, tau
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(*start))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.truck = self.model.body("truck").id
        self.lidar_site = self.model.site("lidar").id
        self.mass = float(self.model.body_mass[self.truck])
        self.izz = float(self.model.body_inertia[self.truck][2])
        self.noise, self.max_range = lidar_noise, max_range
        self.n_az = n_az
        self.dirs = lidar_directions(n_az, elevations_deg)
        self.rng = np.random.default_rng(seed)
        self.steer = 0.0          # current front wheel angle, rad (left +)
        self.last_scan_world = np.zeros((0, 3))

    # -- state --------------------------------------------------------------
    @property
    def pose(self):
        q = self.data.qpos
        return float(q[0]), float(q[1]), float(math.atan2(math.sin(q[2]), math.cos(q[2])))

    @property
    def speed(self):
        """Forward speed of the rear axle, m/s."""
        _, _, yaw = self.pose
        vx, vy = self.data.qvel[0], self.data.qvel[1]
        return float(vx * math.cos(yaw) + vy * math.sin(yaw))

    @property
    def time(self):
        return float(self.data.time)

    # -- physics ------------------------------------------------------------
    def step(self, pwm, dt):
        """Hold the PWM duties for dt seconds of physics. Planar force model:
        drive force from the ESC, stiff lateral tyre force (no side slip), yaw torque that
        follows the bicycle-model yaw rate v * tan(delta) / L."""
        steer_duty, _, drive_duty = pwm
        p = self.p
        n = clamp((steer_duty - NEUTRAL) / 2.5 * p.steer_sign, -1.0, 1.0)   # +1 = left
        delta_cmd = n * math.radians(p.max_steer_deg)
        nsub = max(1, int(round(dt / self.model.opt.timestep)))
        h = self.model.opt.timestep
        m, izz = self.mass, self.izz
        for _ in range(nsub):
            # steering servo, about 0.1 s to full lock
            self.steer += clamp(delta_cmd - self.steer, -6.0 * h, 6.0 * h)
            yaw = self.data.qpos[2]
            c, s = math.cos(yaw), math.sin(yaw)
            vx, vy, wz = self.data.qvel[0], self.data.qvel[1], self.data.qvel[2]
            v_long, v_lat = vx * c + vy * s, -vx * s + vy * c
            if drive_duty >= p.drive_forward_min:
                v_cmd = (drive_duty - NEUTRAL) / 2.5 * self.top_speed
                f_long = m * (v_cmd - v_long) / self.tau
            elif drive_duty < NEUTRAL - 0.5:                                  # brake
                f_long = -m * clamp(v_long / 0.05, -3.0, 3.0)
            else:                                                             # coast
                f_long = -m * clamp(v_long / 0.2, -0.5, 0.5)
            f_lat = -m * v_lat / 0.03
            w_des = v_long * math.tan(self.steer) / p.wheelbase
            tz = izz * (w_des - wz) / 0.03
            self.data.qfrc_applied[0] = f_long * c - f_lat * s
            self.data.qfrc_applied[1] = f_long * s + f_lat * c
            self.data.qfrc_applied[2] = tz
            mujoco.mj_step(self.model, self.data)
        return self.pose + (self.speed,)

    # -- lidar --------------------------------------------------------------
    def lidar_origin(self):
        return self.data.site_xpos[self.lidar_site].copy(), self.data.site_xmat[self.lidar_site].reshape(3, 3).copy()

    def scan(self):
        """One sweep: (K,3) float32 points in the lidar frame (x forward, y left, z up).
        A random azimuth offset each sweep, like a spinning lidar that never fires at the
        exact same angles twice."""
        pos, rot = self.lidar_origin()
        a0 = self.rng.uniform(0, 2 * math.pi / self.n_az)
        ca, sa = math.cos(a0), math.sin(a0)
        rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        local = self.dirs @ rz.T
        world = local @ rot.T
        n = len(world)
        geomid = np.zeros(n, dtype=np.int32)
        dist = np.zeros(n, dtype=np.float64)
        mujoco.mj_multiRay(self.model, self.data, pos, world.reshape(-1), None, 1, self.truck,
                           geomid, dist, None, n, self.max_range)
        hit = (geomid >= 0) & (dist > 0.05) & (dist < self.max_range)
        r = dist[hit] + self.rng.normal(0, self.noise, int(hit.sum()))
        pts = local[hit] * r[:, None]
        self.last_scan_world = pos + world[hit] * r[:, None]
        return pts.astype(np.float32)


# ----------------------------------------------------------------------------
# Drawing helpers: extra geoms in an MjvScene (offscreen renderer or viewer.user_scn)
# ----------------------------------------------------------------------------
def _add(scn, gtype, size, pos, rgba, mat=None):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                        np.eye(3).reshape(-1) if mat is None else np.asarray(mat, float).reshape(-1),
                        np.asarray(rgba, np.float32))
    scn.ngeom += 1
    return g


def draw_polyline(scn, pts, rgba, width=0.025, z=0.03):
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if scn.ngeom >= scn.maxgeom:
            return
        if math.hypot(x1 - x0, y1 - y0) < 1e-4:
            continue
        g = _add(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, [width, 0, 0], [0, 0, 0], rgba)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                             np.array([x0, y0, z], float), np.array([x1, y1, z], float))


def draw_sphere(scn, x, y, z, r, rgba):
    _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [r, r, r], [x, y, z], rgba)


def draw_arrow(scn, x, y, yaw, rgba, length=0.45, z=0.5, width=0.03):
    g = _add(scn, mujoco.mjtGeom.mjGEOM_ARROW, [width, width, length], [0, 0, 0], rgba)
    if g is not None:
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.array([x, y, z], float),
                             np.array([x + length * math.cos(yaw), y + length * math.sin(yaw), z], float))


def draw_scan(scn, pts_world, max_points=1500, r=0.025, rng=None):
    """Lidar returns as small dots, coloured by height (blue low -> red high). Floor hits are
    drawn dimmer so the walls and objects stand out."""
    if len(pts_world) == 0:
        return
    rng = rng or np.random.default_rng(0)
    if len(pts_world) > max_points:
        pts_world = pts_world[rng.choice(len(pts_world), max_points, replace=False)]
    for x, y, z in pts_world:
        if scn.ngeom >= scn.maxgeom:
            return
        t = clamp(z / 2.0, 0.0, 1.0)
        if z < 0.06:
            rgba = (0.55, 0.9, 1.0, 0.35)
        else:
            rgba = (0.3 + 0.7 * t, 1.0 - 0.6 * t, 1.0 - 0.9 * t, 1.0)
        _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [r, r, r], [x, y, max(z, 0.015)], rgba)


def odom_to_world(start, x, y, yaw):
    """Pose in the lidar odometry frame (rear axle starts at 0,0 facing +x) -> world frame."""
    sx, sy, syaw = start
    c, s = math.cos(syaw), math.sin(syaw)
    return sx + c * x - s * y, sy + s * x + c * y, syaw + yaw
