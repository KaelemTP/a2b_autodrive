"""
a2b_autodrive.core  -  ROS-free algorithm for driving the mini dump truck from A to B.

Everything here is plain Python + math so it can be tested without ROS and
re-used unchanged in the SimOne loop and on the real truck.

Pipeline (one call to A2BController.step() per control tick):

    localization (x, y, yaw) + chassis speed + obstacle points
        -> Route (A -> B, densified)
        -> PurePursuit  -> steering angle  -> PWM1 duty
        -> SpeedPlanner + PI -> throttle  -> DriveESC -> PWM3 duty
        -> Safety (timeouts, obstacle stop, arrival)
    output: ControlCmd (normalised steer / throttle / brake + PWM duties)

PWM rules come from the Huzhou test of the real truck (50 Hz):
    PWM1 steering  position: 5 % full left, 7.5 % straight, 10 % full right
    PWM2 bucket    speed   : 7.5 % stop  (not used for A->B, always held at 7.5)
    PWM3 drive ESC speed   : 7.5 % stop, >7.5 forward, 6.0 % = brake
    7.5 % is the only safe value. 5 % is NEVER used as stop.
"""

import math
from dataclasses import dataclass, field

NEUTRAL = 7.5  # % duty, safe value for every channel


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


# ----------------------------------------------------------------------------
# Parameters (all overridable from ROS params / config yaml)
# ----------------------------------------------------------------------------
@dataclass
class Params:
    # vehicle geometry
    wheelbase: float = 0.30            # m, front-to-rear axle (measure the truck!)
    max_steer_deg: float = 30.0        # front wheel angle at 5 % / 10 %
    steer_sign: float = -1.0           # ROS +angle = left; truck: <7.5 % = left -> -1

    # pure pursuit
    lookahead_min: float = 0.6         # m
    lookahead_gain: float = 1.0        # s, Ld = min + gain * v
    lookahead_max: float = 2.5         # m

    # speed
    cruise_speed: float = 0.6          # m/s target on straights
    min_speed: float = 0.25            # m/s, ESC needs some throttle to keep rolling
    max_lat_accel: float = 0.6         # m/s^2, slow down in curves
    decel: float = 0.4                 # m/s^2, used to start slowing before B
    kp: float = 0.8
    ki: float = 0.3
    ff_gain: float = 0.6               # throttle per (m/s) feed-forward

    # PWM mapping (% duty)
    drive_forward_min: float = 7.9     # ESC dead-band: first duty that moves the truck
    drive_forward_max: float = 8.6     # cap for a cautious first trial (10 % = full)
    brake_duty: float = 6.0            # emergency brake value on this ESC
    brake_time: float = 0.6            # s of brake before going back to 7.5 %
    steer_rate_limit: float = 6.0      # % duty per second
    throttle_rate_limit: float = 2.0   # % duty per second (ramping up only)

    # goal / route
    goal_tolerance: float = 0.30       # m, "arrived at B"
    route_step: float = 0.10           # m between route points

    # safety
    loc_timeout: float = 0.5           # s without localization -> stop
    obstacle_stop_dist: float = 1.0    # m, stop if something closer in corridor
    obstacle_slow_dist: float = 2.5    # m, scale speed down
    corridor_half_width: float = 0.35  # m (truck half width + margin)
    obstacle_z_min: float = 0.05       # m above lidar origin  (tune to mounting)
    obstacle_z_max: float = 1.5
    obstacle_min_points: int = 5       # ignore single noisy points
    max_cross_track: float = 1.5       # m off route -> stop (lost)


# ----------------------------------------------------------------------------
# Route
# ----------------------------------------------------------------------------
class Route:
    """Polyline A -> (waypoints) -> B, resampled every `step` metres."""

    def __init__(self, points, step=0.1):
        if len(points) < 2:
            raise ValueError("route needs at least A and B")
        self.pts = []
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            seg = math.hypot(x1 - x0, y1 - y0)
            n = max(1, int(seg / step))
            for i in range(n):
                t = i / n
                self.pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
        self.pts.append(tuple(points[-1]))
        # cumulative distance along route
        self.s = [0.0]
        for (xa, ya), (xb, yb) in zip(self.pts[:-1], self.pts[1:]):
            self.s.append(self.s[-1] + math.hypot(xb - xa, yb - ya))
        self.length = self.s[-1]
        self._idx = 0  # progress index, only moves forward

    @property
    def goal(self):
        return self.pts[-1]

    def nearest(self, x, y, window=60):
        """Nearest route index, searching forward from last progress."""
        best_i, best_d = self._idx, float("inf")
        for i in range(self._idx, min(len(self.pts), self._idx + window)):
            d = math.hypot(self.pts[i][0] - x, self.pts[i][1] - y)
            if d < best_d:
                best_i, best_d = i, d
        self._idx = best_i
        return best_i, best_d

    def lookahead_point(self, i0, ld):
        target_s = self.s[i0] + ld
        for i in range(i0, len(self.pts)):
            if self.s[i] >= target_s:
                return self.pts[i]
        return self.pts[-1]

    def remaining(self, i):
        return self.length - self.s[i]

    def curvature_ahead(self, i, dist=1.5):
        """Max |curvature| over the next `dist` metres (for curve slow-down)."""
        kmax = 0.0
        j = i
        while j + 2 < len(self.pts) and self.s[j] - self.s[i] < dist:
            (x0, y0), (x1, y1), (x2, y2) = self.pts[j], self.pts[j + 1], self.pts[j + 2]
            a = math.hypot(x1 - x0, y1 - y0)
            b = math.hypot(x2 - x1, y2 - y1)
            c = math.hypot(x2 - x0, y2 - y0)
            area2 = abs((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0))
            if a * b * c > 1e-9:
                kmax = max(kmax, 2.0 * area2 / (a * b * c))
            j += 1
        return kmax


# ----------------------------------------------------------------------------
# Obstacle check on a point cloud already in the VEHICLE frame
# (x forward, y left, z up). Points are an iterable of (x, y, z).
# ----------------------------------------------------------------------------
def nearest_obstacle_ahead(points, p: Params, steer_rad=0.0):
    """Distance to the closest obstacle inside the driving corridor, or inf.

    The corridor bends with the current steering so a turn does not see the
    wall beside the road as 'ahead'.
    """
    k = math.tan(steer_rad) / p.wheelbase if p.wheelbase > 0 else 0.0
    hits = []
    for x, y, z in points:
        if x <= 0.0 or x > p.obstacle_slow_dist:
            continue
        if z < p.obstacle_z_min or z > p.obstacle_z_max:
            continue
        y_path = 0.5 * k * x * x          # small-angle arc approximation
        if abs(y - y_path) <= p.corridor_half_width:
            hits.append(x)
    if len(hits) < p.obstacle_min_points:
        return float("inf")
    hits.sort()
    return hits[p.obstacle_min_points - 1]  # robust to a few stray points


# ----------------------------------------------------------------------------
# Drive ESC wrapper: forward + brake only (A->B never needs reverse).
# Encodes the Huzhou rule: stopping from forward = brake (6.0 %) then 7.5 %.
# ----------------------------------------------------------------------------
class DriveESC:
    FORWARD, BRAKING, NEUTRAL_ = "FORWARD", "BRAKING", "NEUTRAL"

    def __init__(self, p: Params):
        self.p = p
        self.state = self.NEUTRAL_
        self.brake_t = 0.0
        self.duty = NEUTRAL

    def update(self, throttle, want_stop, dt):
        """throttle in [0,1]. Returns PWM3 duty (%)."""
        p = self.p
        if want_stop:
            if self.state == self.FORWARD:
                self.state, self.brake_t = self.BRAKING, 0.0
            if self.state == self.BRAKING:
                self.brake_t += dt
                if self.brake_t >= p.brake_time:
                    self.state = self.NEUTRAL_
            self.duty = p.brake_duty if self.state == self.BRAKING else NEUTRAL
            return self.duty

        if throttle <= 1e-3:
            self.state = self.NEUTRAL_ if self.state != self.FORWARD else self.FORWARD
            target = NEUTRAL
        else:
            self.state = self.FORWARD
            target = p.drive_forward_min + throttle * (p.drive_forward_max - p.drive_forward_min)
        # ramp up gently, drop immediately
        if target > self.duty:
            start = max(self.duty, NEUTRAL)
            self.duty = min(target, start + p.throttle_rate_limit * dt)
            if self.duty < p.drive_forward_min and target >= p.drive_forward_min:
                self.duty = p.drive_forward_min  # jump the dead-band
        else:
            self.duty = target
        return self.duty


# ----------------------------------------------------------------------------
# Output of one control step
# ----------------------------------------------------------------------------
@dataclass
class ControlCmd:
    steer: float = 0.0       # normalised [-1, 1], +1 = full LEFT (ROS convention)
    throttle: float = 0.0    # [0, 1]
    brake: float = 0.0       # [0, 1]
    pwm: list = field(default_factory=lambda: [NEUTRAL, NEUTRAL, NEUTRAL])  # PWM1..3 %
    state: str = "IDLE"
    info: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Main controller
# ----------------------------------------------------------------------------
class A2BController:
    IDLE, DRIVING, ARRIVED, STOPPED = "IDLE", "DRIVING", "ARRIVED", "STOPPED"

    def __init__(self, params: Params = None):
        self.p = params or Params()
        self.route = None
        self.esc = DriveESC(self.p)
        self.state = self.IDLE
        self.stop_reason = ""
        self.int_err = 0.0
        self.steer_duty = NEUTRAL

    # -- commands -----------------------------------------------------------
    def set_route(self, points):
        self.route = Route(points, self.p.route_step)
        self.state = self.DRIVING
        self.stop_reason = ""
        self.int_err = 0.0

    def stop(self, reason="stop requested"):
        if self.state != self.IDLE:
            self.state, self.stop_reason = self.STOPPED, reason

    # -- one tick -----------------------------------------------------------
    def step(self, pose, speed, dt, obstacle_dist=float("inf"), loc_age=0.0):
        """
        pose          (x, y, yaw) in the map frame, yaw in rad (or None)
        speed         m/s from chassis (forward positive)
        dt            s since last step
        obstacle_dist m, from nearest_obstacle_ahead()
        loc_age       s since the last localization message
        """
        p = self.p
        cmd = ControlCmd()

        if self.state == self.DRIVING:
            if pose is None or loc_age > p.loc_timeout:
                self.stop("localization timeout")
            elif obstacle_dist <= p.obstacle_stop_dist:
                self.stop(f"obstacle at {obstacle_dist:.2f} m")

        if self.state != self.DRIVING:
            return self._stopped(cmd, dt)

        x, y, yaw = pose
        i, cte = self.route.nearest(x, y)
        remaining = self.route.remaining(i)
        dist_goal = math.hypot(self.route.goal[0] - x, self.route.goal[1] - y)

        if cte > p.max_cross_track:
            self.stop(f"off route ({cte:.2f} m)")
            return self._stopped(cmd, dt)
        # arrival: close to B, or we have passed the end of the route
        if dist_goal < p.goal_tolerance or (i >= len(self.route.pts) - 1 and dist_goal < 1.0):
            self.state, self.stop_reason = self.ARRIVED, "arrived at B"
            return self._stopped(cmd, dt)

        # ---- lateral: pure pursuit ---------------------------------------
        v = max(speed, 0.0)
        ld = clamp(p.lookahead_min + p.lookahead_gain * v, p.lookahead_min, p.lookahead_max)
        ld = min(ld, max(remaining, p.lookahead_min * 0.5))
        tx, ty = self.route.lookahead_point(i, ld)
        alpha = wrap_angle(math.atan2(ty - y, tx - x) - yaw)
        ld_actual = max(math.hypot(tx - x, ty - y), 1e-3)
        delta = math.atan2(2.0 * p.wheelbase * math.sin(alpha), ld_actual)
        dmax = math.radians(p.max_steer_deg)
        delta = clamp(delta, -dmax, dmax)
        steer_n = delta / dmax                                   # +1 = left

        # steer_sign = -1: left (+n) -> duty below 7.5 (towards 5 %)
        target_duty = NEUTRAL + p.steer_sign * steer_n * 2.5
        max_step = p.steer_rate_limit * dt
        self.steer_duty += clamp(target_duty - self.steer_duty, -max_step, max_step)
        self.steer_duty = clamp(self.steer_duty, 5.0, 10.0)

        # ---- longitudinal: speed profile + PI ----------------------------
        kappa = self.route.curvature_ahead(i)
        v_curve = math.sqrt(p.max_lat_accel / kappa) if kappa > 1e-3 else p.cruise_speed
        v_goal = math.sqrt(max(2.0 * p.decel * max(remaining - p.goal_tolerance * 0.5, 0.0), 0.0))
        v_obs = p.cruise_speed
        if obstacle_dist < p.obstacle_slow_dist:
            frac = (obstacle_dist - p.obstacle_stop_dist) / (p.obstacle_slow_dist - p.obstacle_stop_dist)
            v_obs = p.cruise_speed * clamp(frac, 0.0, 1.0)
        v_ref = min(p.cruise_speed, v_curve, v_goal, v_obs)
        v_ref = max(v_ref, p.min_speed) if remaining > p.goal_tolerance else 0.0

        err = v_ref - v
        self.int_err = clamp(self.int_err + err * dt, -1.0, 1.0)
        throttle = clamp(p.ff_gain * v_ref + p.kp * err + p.ki * self.int_err, 0.0, 1.0)

        drive_duty = self.esc.update(throttle, want_stop=False, dt=dt)

        cmd.steer, cmd.throttle, cmd.brake = steer_n, throttle, 0.0
        cmd.pwm = [round(self.steer_duty, 2), NEUTRAL, round(drive_duty, 2)]
        cmd.state = self.state
        cmd.info = dict(idx=i, cte=round(cte, 3), remaining=round(remaining, 2),
                        v_ref=round(v_ref, 2), lookahead=(round(tx, 2), round(ty, 2)),
                        obstacle=obstacle_dist)
        return cmd

    def _stopped(self, cmd, dt):
        drive_duty = self.esc.update(0.0, want_stop=True, dt=dt)
        # straighten steering slowly
        max_step = self.p.steer_rate_limit * dt
        self.steer_duty += clamp(NEUTRAL - self.steer_duty, -max_step, max_step)
        braking = self.esc.state == DriveESC.BRAKING
        cmd.steer = (self.steer_duty - NEUTRAL) / 2.5 * self.p.steer_sign
        cmd.throttle, cmd.brake = 0.0, 1.0 if braking else 0.0
        cmd.pwm = [round(self.steer_duty, 2), NEUTRAL, round(drive_duty, 2)]
        cmd.state = self.state
        cmd.info = dict(reason=self.stop_reason)
        self.int_err = 0.0
        return cmd


# ----------------------------------------------------------------------------
# Simple kinematic bicycle model of the truck, driven by PWM duties.
# Used only for offline testing before SimOne / the real truck.
# ----------------------------------------------------------------------------
class TruckSim:
    def __init__(self, p: Params, x=0.0, y=0.0, yaw=0.0, top_speed=2.0, tau=0.5):
        self.p, self.x, self.y, self.yaw, self.v = p, x, y, yaw, 0.0
        self.top_speed, self.tau = top_speed, tau  # speed at 10 %, motor lag

    def step(self, pwm, dt):
        steer_duty, _, drive_duty = pwm
        n = (steer_duty - NEUTRAL) / 2.5 * self.p.steer_sign       # +1 = left
        delta = clamp(n, -1, 1) * math.radians(self.p.max_steer_deg)
        if drive_duty >= self.p.drive_forward_min:
            v_cmd = (drive_duty - NEUTRAL) / 2.5 * self.top_speed
            self.v += (v_cmd - self.v) * dt / self.tau
        elif drive_duty < NEUTRAL - 0.5:                            # brake
            self.v = max(0.0, self.v - 3.0 * dt)
        else:                                                       # coast
            self.v = max(0.0, self.v - 0.5 * dt)
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw = wrap_angle(self.yaw + self.v / self.p.wheelbase * math.tan(delta) * dt)
        return self.x, self.y, self.yaw, self.v
