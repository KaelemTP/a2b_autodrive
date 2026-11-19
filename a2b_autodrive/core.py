"""
a2b_autodrive.core  -  ROS-free algorithm for driving the mini dump truck from A to B.

Everything here is plain Python + math so it can be tested without ROS and
re-used unchanged in the SimOne loop and on the real truck.

Pipeline (one call to A2BController.step() per control tick):

    localization (x, y, yaw) + chassis speed + obstacle points
        -> Route (A -> B, densified)
        -> PurePursuit  -> steering angle  -> PWM1 duty
        -> SpeedPlanner + PI -> throttle  -> DriveESC -> PWM3 duty
        -> Avoider: picks a sideways detour around obstacles (or none) -> path to follow
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
    obstacle_z_min: float = -0.3       # m in the lidar frame (floor already removed by ground_filter)
    obstacle_z_max: float = 1.5
    obstacle_resume: bool = True       # wait while blocked and carry on when clear (False = stop for good)
    obstacle_clear_margin: float = 0.3  # m beyond obstacle_stop_dist before driving on again
    obstacle_min_points: int = 5       # ignore single noisy points
    max_cross_track: float = 1.5       # m off route -> stop (lost); detours add avoid_max_offset

    # obstacle avoidance (needs the obstacle point cloud; otherwise only the stop above)
    avoid_enable: bool = True          # False = stop and wait at obstacles (v0.3 behaviour)
    avoid_max_offset: float = 2.0      # m, furthest sideways the detour may go from the route
    avoid_step: float = 0.25           # m between the sideways options tried
    avoid_check_dist: float = 5.0      # m ahead along the path that must be free
    avoid_ramp: float = 2.0            # m of forward travel to move sideways onto a detour
    avoid_clearance: float = 0.20      # m extra gap preferred beyond corridor_half_width
    avoid_switch_cost: float = 0.6     # discourages flipping between left and right detours
    avoid_speed: float = 0.4           # m/s while off the route on a detour
    avoid_stop_dist: float = 0.4       # m of driving left before the body would come within
                                       #   corridor_half_width of an obstacle: stop (BLOCKED)
    avoid_slow_dist: float = 1.5       # m, start slowing down before that point
    body_front: float = 0.45           # m, front of the truck body ahead of the rear axle
    body_half_width: float = 0.20      # m, half the truck body width

    # back-up recovery: when BLOCKED with no way forward, reverse a little and try again.
    # OFF by default: the reverse PWM value of the real ESC has NOT been measured yet.
    reverse_enable: bool = False
    reverse_duty: float = 7.0          # % PWM3 for slow reverse (sim assumption; measure on the truck!)
    reverse_dist: float = 0.6          # m to back up each time
    reverse_after: float = 0.8         # s of BLOCKED with no free option before backing up
                                       #   (must be >= brake_time so the brake has finished)
    reverse_neutral_time: float = 0.2  # s at 7.5 % between brake and reverse (ESC needs neutral)
    reverse_max_tries: int = 5         # then just wait (counter resets after 1.5 m of progress)
    goal_blocked_arrive: float = 2.5   # m: if B itself is blocked and the truck is this close, it
                                       #   finishes (ARRIVED, "as close as it can get")


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
        # unit LEFT normal of each point, averaged over about +-0.5 m so detours offset from
        # the route bend smoothly round waypoint corners instead of jumping
        raw = []
        for k in range(len(self.pts)):
            a, b = (self.pts[k], self.pts[k + 1]) if k + 1 < len(self.pts) else (self.pts[k - 1], self.pts[k])
            L = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
            raw.append((-(b[1] - a[1]) / L, (b[0] - a[0]) / L))
        w = max(1, int(round(0.5 / step)))
        self.nrm = []
        for k in range(len(raw)):
            nx = sum(r[0] for r in raw[max(0, k - w):k + w + 1])
            ny = sum(r[1] for r in raw[max(0, k - w):k + w + 1])
            L = math.hypot(nx, ny)
            self.nrm.append((nx / L, ny / L) if L > 1e-6 else raw[k])

    @property
    def goal(self):
        return self.pts[-1]

    def nearest(self, x, y, window=60):
        """Nearest route index, searching forward from last progress."""
        best_i, best_d = self._idx, float("inf")
        # mostly forward; a little backward so backing up a bit does not confuse the progress
        for i in range(max(0, self._idx - 10), min(len(self.pts), self._idx + window)):
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

    def lateral(self, i, x, y):
        """Signed sideways distance of (x, y) from route point i, + = left of the route."""
        (px, py), (nx, ny) = self.pts[i], self.nrm[i]
        return (x - px) * nx + (y - py) * ny

    def sample(self, s_values):
        """Route points and left normals at the given distances along the route (numpy arrays)."""
        import numpy as np
        s_arr = np.asarray(self.s)
        idx = np.clip(np.searchsorted(s_arr, s_values, side="right") - 1, 0, len(self.pts) - 1)
        pts, nrm = np.asarray(self.pts), np.asarray(self.nrm)
        nxt = np.minimum(idx + 1, len(self.pts) - 1)
        seg = np.maximum(s_arr[nxt] - s_arr[idx], 1e-9)
        t = np.clip((np.asarray(s_values) - s_arr[idx]) / seg, 0.0, 1.0)[:, None]
        return pts[idx] + t * (pts[nxt] - pts[idx]), nrm[idx]

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
# Obstacle avoidance: try paths shifted sideways from the route, keep the best free one.
#
# Each option d is "move to d metres left (+) / right (-) of the route over avoid_ramp metres,
# then stay there". It is re-planned every tick from where the truck is now, so once the
# obstacle is passed the d = 0 option (back on the route) becomes free and is preferred.
# Near B every option bends back onto the route, so an obstacle ON B still blocks.
# ----------------------------------------------------------------------------
class Avoider:
    """Obstacle avoidance for one control tick.

    Options are reference paths starting at the truck:
      * "offset" paths: the route shifted 0, +-0.25 ... +-avoid_max_offset m sideways (smooth
        ramp from where the truck is now, back onto the route near B);
      * a "search" path: when none of those is free, a shortest way round on a grid built from
        the lidar points (A*), to B or to a free point further along the route.
    Every option is checked by simulating the drive along it: same Pure Pursuit steering, turning
    limit and steering speed, from the truck's real position and heading. The truck body (three
    circles along it) must keep corridor_half_width to every point (a little less in the first
    metre, so a truck that is already close can steer away). The controller then follows the
    chosen reference path."""

    GRID = 0.10            # m, search grid cell
    SEARCH_EVERY = 0.5     # s, at most this often (search is the slow part)

    def __init__(self, p: Params):
        self.p = p
        self.reset()

    def reset(self):
        self.d = 0.0            # chosen sideways offset, m (+ left); 0 for a search path
        self.d_now = 0.0        # where the truck is sideways from the route right now, m
        self.kind = "offset"
        self.free = True
        self.path = []          # simulated drive of the chosen option [(x, y), ...] for display
        self.ref = None         # chosen reference path, (M, 2) numpy array
        self.clearance = float("inf")
        self.goal_blocked = False
        self.search_path = None
        self.search_t = -1e9
        self.clock = 0.0
        self.steer_now = 0.0

    # ---- reference paths -----------------------------------------------------------
    def _offset_paths(self, route, options, ld):
        import numpy as np
        p = self.p
        s_end = min(self.s0 + p.avoid_check_dist + ld + 0.5, route.length)
        s = np.arange(self.s0, s_end + 1e-9, 0.1)
        if len(s) < 2:
            s = np.array([self.s0, route.length])
        base, nrm = route.sample(s)
        ds = s - self.s0
        out = []
        for d in options:
            ramp = max(p.avoid_ramp, 1.5 * abs(d - self.d_now))
            u = np.clip(ds / ramp, 0.0, 1.0)
            taper = np.clip((route.length - s) / p.avoid_ramp, 0.0, 1.0)
            off = (self.d_now + (d - self.d_now) * u * u * (3.0 - 2.0 * u)) * taper
            out.append(base + off[:, None] * nrm)
        return out

    def _grids(self, pts, x0, y0, nx, ny):
        """Hard grid (body would touch) and soft grid (closer than corridor_half_width)."""
        import numpy as np
        p, g = self.p, self.GRID
        occ = np.zeros((nx, ny), bool)
        if len(pts):
            ix = np.floor((pts[:, 0] - x0) / g).astype(int)
            iy = np.floor((pts[:, 1] - y0) / g).astype(int)
            ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            cnt = np.zeros((nx, ny), int)
            np.add.at(cnt, (ix[ok], iy[ok]), 1)
            occ = cnt >= 1

        def dilate(r):
            out = np.zeros_like(occ)
            R = int(math.ceil(r / g))
            for dx in range(-R, R + 1):
                for dy in range(-R, R + 1):
                    if (dx * dx + dy * dy) * g * g > r * r:
                        continue
                    src = occ[max(0, -dx):nx - max(0, dx), max(0, -dy):ny - max(0, dy)]
                    out[max(0, dx):nx - max(0, -dx), max(0, dy):ny - max(0, -dy)] |= src
            return out
        return (dilate(p.body_half_width + 0.08), dilate(p.corridor_half_width + 0.05),
                dilate(p.corridor_half_width + 0.40))

    def _search(self, route, x, y, pts, yaw=0.0):
        """Hybrid A*: a drivable way round (forward arcs within the truck's turning circle) on a
        grid made from the lidar points, to B or to a free point further along the route.
        Returns an (M, 2) path or None, and whether B itself is blocked."""
        import heapq
        import time as _time
        import numpy as np
        p, g = self.p, self.GRID
        # ---- grid around the truck and the target ------------------------------------------
        s_t = min(self.s0 + p.avoid_check_dist + 2.0, route.length)
        cand_s = np.arange(s_t, route.length + 1e-9, 0.25) if s_t < route.length else np.array([route.length])
        cand = route.sample(np.append(cand_s, route.length))[0]
        lo = np.minimum(np.array([x, y]) - 4.0, cand[0] - 4.0)
        hi = np.maximum(np.array([x, y]) + 4.0, cand[0] + 4.0)
        hi = np.minimum(hi, lo + 24.0)
        nx, ny = int((hi[0] - lo[0]) / g) + 1, int((hi[1] - lo[1]) / g) + 1
        hard, soft, wide = self._grids(pts, lo[0], lo[1], nx, ny)

        def cell(q):
            return int((q[0] - lo[0]) / g), int((q[1] - lo[1]) / g)

        def inside(c):
            return 0 <= c[0] < nx and 0 <= c[1] < ny

        bx, by = cell(route.goal)
        goal_blocked = inside((bx, by)) and bool(soft[bx, by])
        goal = None
        for q in cand:
            c = cell(q)
            if inside(c) and not soft[c]:
                goal = (float(q[0]), float(q[1]))
                break
        if goal is None:
            return None, goal_blocked

        # ---- heuristic: shortest grid distance to the goal around obstacles (Dijkstra) ---------
        gc = cell(goal)
        dist = np.full((nx, ny), np.inf)
        dist[gc] = 0.0
        heap = [(0.0, gc)]
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        while heap:
            d0, c = heapq.heappop(heap)
            if d0 > dist[c]:
                continue
            for dx, dy, w in moves:
                nb = (c[0] + dx, c[1] + dy)
                if inside(nb) and not hard[nb] and d0 + w < dist[nb]:
                    dist[nb] = d0 + w
                    heapq.heappush(heap, (d0 + w, nb))
        dist *= g

        # ---- search over (x, y, heading) with forward arcs --------------------------------
        body = np.array([0.0, 0.5 * p.body_front, p.body_front])
        kmax = math.tan(math.radians(p.max_steer_deg)) / p.wheelbase * 0.8   # keep some steering spare
        curvs = [-kmax, -0.5 * kmax, 0.0, 0.5 * kmax, kmax]
        L, sub = 0.3, 3                                       # arc length per step, checks per arc
        n_head = 72

        def blocked_pose(px, py, th, near_start):
            c, s = math.cos(th), math.sin(th)
            pen = 0.0
            for b in body:
                ci = cell((px + c * b, py + s * b))
                if not inside(ci):
                    return True, 0.0
                if hard[ci] and not near_start:
                    return True, 0.0
                pen += 0.6 if soft[ci] else 0.15 if wide[ci] else 0.0
            return False, pen

        def key(px, py, th):
            return (int(px / 0.15), int(py / 0.15), int(round(th / (2 * math.pi) * n_head)) % n_head)

        sx, sy = x, y
        start = (sx, sy, yaw)
        h0 = dist[cell((sx, sy))] if inside(cell((sx, sy))) else np.inf
        if not np.isfinite(h0):
            h0 = math.hypot(goal[0] - sx, goal[1] - sy)
        openl = [(h0, 0.0, start, None, 0.0)]
        seen = {}
        parents = {}
        t_end = _time.monotonic() + 0.3                       # the planner waits for this: keep it short
        found = None
        n = 0
        while openl and n < 20000 and _time.monotonic() < t_end:
            f, cst, st, par, k_prev = heapq.heappop(openl)
            k = key(*st)
            if k in seen:
                continue
            seen[k] = True
            parents[k] = (par, st)
            n += 1
            if math.hypot(st[0] - goal[0], st[1] - goal[1]) < 0.35:
                found = k
                break
            near_start = math.hypot(st[0] - sx, st[1] - sy) < 0.4
            for kv in curvs:
                px, py, th = st
                pen, ok = 0.0, True
                for _ in range(sub):
                    ds = L / sub
                    if abs(kv) < 1e-6:
                        px += ds * math.cos(th)
                        py += ds * math.sin(th)
                    else:
                        nth = th + kv * ds
                        px += (math.sin(nth) - math.sin(th)) / kv
                        py += -(math.cos(nth) - math.cos(th)) / kv
                        th = nth
                    bad, pp = blocked_pose(px, py, th, near_start)
                    if bad:
                        ok = False
                        break
                    pen += pp
                if not ok:
                    continue
                nk = key(px, py, th)
                if nk in seen:
                    continue
                c2 = cell((px, py))
                h = dist[c2] if inside(c2) else np.inf
                if not np.isfinite(h):
                    continue
                ncost = cst + L * (1.0 + 0.3 * abs(kv) / kmax + 0.4 * abs(kv - k_prev) / kmax) + pen * L
                heapq.heappush(openl, (ncost + 1.1 * h, ncost, (px, py, th), k, kv))
        if found is None:
            return None, goal_blocked
        states = []
        k = found
        while k is not None:
            par, st = parents[k]
            states.append(st)
            k = par
        states.reverse()
        path = np.array([(s_[0], s_[1]) for s_ in states])
        # densify to 0.1 m for path following
        dense = [path[0]]
        for a, b in zip(path[:-1], path[1:]):
            m = max(1, int(math.hypot(*(b - a)) / 0.1))
            for t in range(1, m + 1):
                dense.append(a + (b - a) * t / m)
        path = np.array(dense)
        gi = int(np.argmin(np.hypot(*(np.asarray(route.pts) - path[-1]).T)))
        tail = np.asarray(route.pts[gi + 1:gi + 30]) if gi + 1 < len(route.pts) else np.zeros((0, 2))
        return self._smooth(np.vstack([path, tail]), len(path), soft, wide, lo), goal_blocked

    def _smooth(self, path, n_search, soft, wide, lo, iters=80, w=0.3):
        """Take the zig-zag out of a Hybrid A* path. The search only knows 5 fixed turning rates,
        so its path wiggles, and following it swung the steering from full lock one way to full
        lock the other. Each point is pulled towards the middle of its neighbours, but never closer
        to an obstacle than it was (measured in the same margin bands the search uses: within
        corridor_half_width + 5 cm, within + 40 cm, further), so it does not cut corners. The
        first 0.3 m (the truck's heading) and the route tail after the way round stay fixed."""
        import numpy as np
        g = self.GRID
        nx, ny = soft.shape
        n = len(path)
        if n < 7:
            return path
        movable = np.zeros(n, bool)
        movable[3:min(n_search + 5, n - 2)] = True

        def band(q):   # 2 = within the soft margin, 1 = within the wide margin, 0 = clear
            ix = np.floor((q[:, 0] - lo[0]) / g).astype(int)
            iy = np.floor((q[:, 1] - lo[1]) / g).astype(int)
            ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            res = np.full(len(q), 2)
            res[ok] = 2 * soft[ix[ok], iy[ok]] + (wide[ix[ok], iy[ok]] & ~soft[ix[ok], iy[ok]])
            return res

        was = band(path)
        cur = path.copy()
        for _ in range(iters):
            new = cur.copy()
            new[1:-1] += w * (cur[:-2] + cur[2:] - 2.0 * cur[1:-1])
            new[~movable] = cur[~movable]
            bad = band(new) > was                # never move a point closer to an obstacle
            new[bad] = cur[bad]
            cur = new
        return cur

    # ---- simulate the drive along each reference ------------------------------------
    def _simulate(self, refs, x, y, yaw, ld, n_steps, step, goal=None):
        import numpy as np
        p = self.p
        K = len(refs)
        M = max(len(r) for r in refs)
        P = np.zeros((K, M, 2))
        for k, r in enumerate(refs):
            P[k, :len(r)] = r
            P[k, len(r):] = r[-1]
        seg = np.hypot(*np.diff(P, axis=1).transpose(2, 0, 1))                  # (K, M-1)
        A = np.concatenate([np.zeros((K, 1)), np.cumsum(seg, axis=1)], axis=1)  # arc length
        dmax = math.radians(p.max_steer_deg)
        rate = p.steer_rate_limit / 2.5 * dmax * step / max(p.avoid_speed, 0.1)
        X, Y, YAW = np.full(K, x, float), np.full(K, y, float), np.full(K, yaw, float)
        DEL = np.full(K, float(self.steer_now))
        idx = np.zeros(K, int)
        rows = np.arange(K)
        poses = np.zeros((n_steps, K, 3))
        # the truck stops at B: a simulated drive that reaches B stays there (before, it drove on
        # past B and circled round it, which drew a loop and checked for obstacles beyond B)
        done = np.zeros(K, bool)
        gx, gy = goal if goal is not None else (math.inf, math.inf)
        for s in range(n_steps):
            done |= np.hypot(X - gx, Y - gy) < p.goal_tolerance
            if done.all():
                poses[s:] = poses[s - 1] if s > 0 else np.stack([X, Y, YAW], axis=1)
                break
            win = np.clip(idx[:, None] + np.arange(0, 12)[None, :], 0, M - 1)    # nearest, forward only
            dd = np.hypot(P[rows[:, None], win, 0] - X[:, None], P[rows[:, None], win, 1] - Y[:, None])
            idx = win[rows, np.argmin(dd, axis=1)]
            thr = A[rows, idx] + ld
            ti = np.where((A >= thr[:, None]).any(1), (A >= thr[:, None]).argmax(1), M - 1)
            tx, ty = P[rows, ti, 0], P[rows, ti, 1]
            alpha = np.arctan2(ty - Y, tx - X) - YAW
            alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
            dist = np.maximum(np.hypot(tx - X, ty - Y), 1e-3)
            want = np.clip(np.arctan2(2.0 * p.wheelbase * np.sin(alpha), dist), -dmax, dmax)
            DEL += np.clip(want - DEL, -rate, rate)
            mv = ~done
            X[mv] += step * np.cos(YAW[mv])
            Y[mv] += step * np.sin(YAW[mv])
            YAW[mv] += step / p.wheelbase * np.tan(DEL[mv])
            poses[s, :, 0], poses[s, :, 1], poses[s, :, 2] = X, Y, YAW
        return poses

    def _collisions(self, poses, pts, x, y, step, relief=None):
        """First step at which each simulated drive gets too close, and its smallest gap."""
        import numpy as np
        p = self.p
        n_steps, K, _ = poses.shape
        moved = (np.arange(n_steps) + 1) * step
        first, gap = np.full(K, -1), np.full(K, np.inf)
        if not len(pts):
            return first, gap, moved
        cells, counts = np.unique(np.floor(pts / 0.05).astype(np.int64), axis=0, return_counts=True)
        q_all, w_all = (cells + 0.5) * 0.05, counts
        offsets = np.array([0.0, 0.5 * p.body_front, p.body_front])
        tight = p.body_half_width + 0.05
        for k in range(K):
            px = poses[:, k, 0][:, None] + np.cos(poses[:, k, 2])[:, None] * offsets[None, :]
            py = poses[:, k, 1][:, None] + np.sin(poses[:, k, 2])[:, None] * offsets[None, :]
            m = (q_all[:, 0] > px.min() - 0.5) & (q_all[:, 0] < px.max() + 0.5) & \
                (q_all[:, 1] > py.min() - 0.5) & (q_all[:, 1] < py.max() + 0.5)
            if not m.any():
                continue
            q, w = q_all[m], w_all[m]
            d2 = (px.reshape(-1, 1) - q[None, :, 0]) ** 2 + (py.reshape(-1, 1) - q[None, :, 1]) ** 2
            dmin = np.sqrt(d2.reshape(n_steps, 3, -1).min(axis=1))
            # In the first metre the margin may shrink, so a truck that is already close to
            # something can still steer away - but only down to the room it has now. Before
            # v0.4.5 it always shrank to body + 5 cm, which let the truck cut corners.
            floor = clamp(float(dmin[0].min()) - 0.03, tight, p.corridor_half_width)
            radius = np.minimum(p.corridor_half_width, floor + (p.corridor_half_width - floor) * moved)
            r_k = radius - (relief[k] if relief is not None else 0.0)
            hits = ((dmin <= r_k[:, None]) * w[None, :]).sum(1) >= 2
            if hits.any():
                first[k] = int(np.argmax(hits))
            gap[k] = float((dmin.min(1) - radius + p.corridor_half_width).min())
        return first, gap, moved

    # ---- one planning tick ------------------------------------------------------------
    def plan(self, route, i, x, y, pts, yaw=None, dt=0.05):
        """pts: (N, 2) obstacle points in the map frame (floor removed). Returns the distance
        (m of driving) before the truck would get too close on the chosen option, inf if free."""
        import numpy as np
        p = self.p
        self.clock += dt
        self.s0 = route.s[i]
        self.d_now = route.lateral(i, x, y)
        if yaw is None:
            yaw = math.atan2(-route.nrm[i][0], route.nrm[i][1])
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        if len(pts):
            pts = pts[np.hypot(pts[:, 0] - x, pts[:, 1] - y) < p.avoid_check_dist + 6.0]
        step = 0.1
        n_steps = int(p.avoid_check_dist / step)
        ld = clamp(p.lookahead_min + p.lookahead_gain * p.avoid_speed, p.lookahead_min, p.lookahead_max)

        n_opt = int(round(p.avoid_max_offset / p.avoid_step))
        options = list(np.arange(-n_opt, n_opt + 1) * p.avoid_step)
        refs = self._offset_paths(route, options, ld)
        kinds = ["offset"] * len(refs)
        labels = [float(d) for d in options]

        poses = self._simulate(refs, x, y, yaw, ld, n_steps, step, route.goal)
        # hysteresis: the option chosen last time gets 3 cm of slack so it does not flicker
        relief = np.array([0.03 if (self.kind == "offset" and abs(d - self.d) < 1e-6) else 0.0 for d in labels])
        first, gap, moved = self._collisions(poses, pts, x, y, step, relief)
        any_free = (first < 0).any() and not (self.kind == "search" and self.search_path is not None
                                              and not (first[np.abs(np.array(labels) - 0.0) < 1e-6] < 0).all())

        # no shifted route is free: search a way round (or re-check the last one found)
        goal_blocked = False
        if not any_free:
            goal_blocked = getattr(self, "search_goal_blocked", False)
            f2 = None
            if self.search_path is not None:   # keep the way round found earlier while it is free
                j = int(np.argmin(np.hypot(*(self.search_path - [x, y]).T)))
                self.search_path = np.vstack([[x, y], self.search_path[j + 1:]]) \
                    if j + 1 < len(self.search_path) else None
            if self.search_path is not None and len(self.search_path) >= 2:
                sp = self._simulate([self.search_path], x, y, yaw, ld, n_steps, step, route.goal)
                f2, g2, _ = self._collisions(sp, pts, x, y, step, np.array([0.03]))
            if (f2 is None or f2[0] >= 0) and self.clock - self.search_t > self.SEARCH_EVERY:
                self.search_path, goal_blocked = self._search(route, x, y, pts, yaw)
                self.search_t = self.clock
                self.search_goal_blocked = goal_blocked
                f2 = None
                if self.search_path is not None and len(self.search_path) >= 2:
                    sp = self._simulate([self.search_path], x, y, yaw, ld, n_steps, step, route.goal)
                    f2, g2, _ = self._collisions(sp, pts, x, y, step)
            if f2 is not None:
                refs.append(self.search_path)
                kinds.append("search")
                labels.append(0.0)
                poses = np.concatenate([poses, sp], axis=1)
                first, gap = np.append(first, f2), np.append(gap, g2)
        else:
            self.search_path = None

        results = []
        for k in range(len(refs)):
            coll = float("inf") if first[k] < 0 else float(moved[first[k]] - step)
            if kinds[k] == "offset":
                cost = abs(labels[k]) + p.avoid_switch_cost * abs(labels[k] - self.d)
            else:
                cost = 4.0 if self.kind != "search" else -1.0     # stick with a way round once taken
            if gap[k] < p.corridor_half_width + p.avoid_clearance:
                cost += 1.0 + 2.0 * (p.corridor_half_width + p.avoid_clearance - gap[k])
            end = poses[first[k], k, :2] if first[k] >= 0 else poses[-1, k, :2]
            near_b = first[k] >= 0 and math.hypot(end[0] - route.goal[0], end[1] - route.goal[1]) < 1.2
            results.append((k, coll, cost, float(gap[k]), near_b))

        free = [r for r in results if math.isinf(r[1])]
        if free:
            best = min(free, key=lambda r: r[2])
        else:   # nothing gets through: take the option that gets furthest, then stop before it
            reach = max(r[1] for r in results)
            best = min((r for r in results if r[1] >= reach - 0.6), key=lambda r: r[2])
        k, coll, _, gap_b, _ = best
        self.kind, self.d, self.ref = kinds[k], labels[k], refs[k]
        self.free = math.isinf(coll)
        self.clearance = gap_b
        # B counts as blocked when nothing gets through and B is right next to an obstacle (the
        # lidar only sees the near side of things, so "right next to" is generous: 0.7 m)
        b_tight = len(pts) > 0 and bool((np.hypot(pts[:, 0] - route.goal[0], pts[:, 1] - route.goal[1])
                                          < p.corridor_half_width + 0.35).sum() >= 3)
        self.goal_blocked = (not free) and (goal_blocked or b_tight or all(r[4] for r in results))
        self.path = [(round(x, 2), round(y, 2))] + \
            [(round(float(a), 2), round(float(b), 2)) for a, b in poses[2::3, k, :2]]
        return coll

    def target(self, route, i, ld):
        """Pure Pursuit target ld metres ahead along the chosen reference path."""
        import numpy as np
        r = self.ref
        if r is None or len(r) < 2:
            return route.lookahead_point(i, ld)
        a = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(r, axis=0).T))])
        j = int(np.searchsorted(a, ld))
        return (float(r[-1, 0]), float(r[-1, 1])) if j >= len(r) else (float(r[j, 0]), float(r[j, 1]))


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
    IDLE, DRIVING, BLOCKED, BACKING, ARRIVED, STOPPED = "IDLE", "DRIVING", "BLOCKED", "BACKING", "ARRIVED", "STOPPED"

    def __init__(self, params: Params = None):
        self.p = params or Params()
        self.route = None
        self.esc = DriveESC(self.p)
        self.avoider = Avoider(self.p)
        self.avoiding = False       # avoidance active this tick (points given and enabled)
        self.state = self.IDLE
        self.stop_reason = ""
        self.int_err = 0.0
        self.steer_duty = NEUTRAL
        self.blocked_t = 0.0        # s spent BLOCKED
        self.back_tries = 0
        self.back_start = None      # (x, y) where backing up started
        self.back_t = 0.0
        self.back_steer = 0.0       # normalised steering while backing (+1 = left)
        self.rear_pts = None

    # -- commands -----------------------------------------------------------
    def set_route(self, points):
        self.route = Route(points, self.p.route_step)
        self.avoider.reset()
        self.back_tries, self.blocked_t = 0, 0.0
        self.state = self.DRIVING
        self.stop_reason = ""
        self.int_err = 0.0

    def stop(self, reason="stop requested"):
        if self.state != self.IDLE:
            self.state, self.stop_reason = self.STOPPED, reason

    # -- one tick -----------------------------------------------------------
    def step(self, pose, speed, dt, obstacle_dist=float("inf"), loc_age=0.0, obstacle_points=None):
        """
        pose            (x, y, yaw) in the map frame, yaw in rad (or None)
        speed           m/s from chassis (forward positive)
        dt              s since last step
        obstacle_dist   m, from nearest_obstacle_ahead() (used when no obstacle_points are given),
                        -1 = obstacle detection on but no data
        loc_age         s since the last localization message
        obstacle_points (N, 2) obstacle points in the MAP frame, floor removed. With these and
                        avoid_enable the truck steers around obstacles; obstacle_dist is then
                        worked out along the chosen path.
        """
        p = self.p
        cmd = ControlCmd()

        self.avoiding = (p.avoid_enable and obstacle_points is not None and self.route is not None
                         and pose is not None and obstacle_dist >= 0
                         and self.state in (self.DRIVING, self.BLOCKED))
        if self.avoiding:
            i_av, _ = self.route.nearest(pose[0], pose[1])
            self.avoider.steer_now = ((self.steer_duty - NEUTRAL) / 2.5 * p.steer_sign) * math.radians(p.max_steer_deg)
            obstacle_dist = self.avoider.plan(self.route, i_av, pose[0], pose[1], obstacle_points, yaw=pose[2], dt=dt)
        stop_d = p.avoid_stop_dist if self.avoiding else p.obstacle_stop_dist
        slow_d = p.avoid_slow_dist if self.avoiding else p.obstacle_slow_dist

        # obstacle_dist < 0 means "obstacle detection is on but has no data"
        def obstacle_reason():
            return "no obstacle data (point cloud lost)" if obstacle_dist < 0 else f"obstacle at {obstacle_dist:.2f} m"

        if self.state == self.BLOCKED:
            if pose is None or loc_age > p.loc_timeout:
                self.stop("localization timeout")
            elif (obstacle_dist > stop_d + p.obstacle_clear_margin and not self.avoiding) or \
                    (self.avoiding and (self.avoider.free or obstacle_dist > 2.5)):
                # path clear: carry on. With avoidance: only when a whole option is free (or the
                # obstacle is still far off), so it does not creep back into the same spot.
                self.state, self.stop_reason = self.DRIVING, ""
            elif self.avoiding and self.avoider.goal_blocked:
                dist_goal = math.hypot(self.route.goal[0] - pose[0], self.route.goal[1] - pose[1])
                if dist_goal <= p.goal_blocked_arrive:
                    self.state = self.ARRIVED
                    self.stop_reason = (f"as close as it can get: {dist_goal:.2f} m from B "
                                        f"(B is blocked by an obstacle)")
                else:
                    self.stop_reason = "B is blocked by an obstacle - waiting (move B or the obstacle)"
            else:
                self.stop_reason = obstacle_reason() + " - waiting"
                self.blocked_t += dt
                # (no speed check: the lidar-based speed jitters 0.05-0.2 m/s even when standing;
                # BLOCKED has braked for brake_time already)
                if (p.reverse_enable and self.avoiding and not self.avoider.free
                        and self.blocked_t > max(p.reverse_after, p.brake_time) and self.back_tries < p.reverse_max_tries
                        and self._rear_clear(pose, obstacle_points)):
                    self._start_backing(pose, obstacle_points)

        if self.state == self.BACKING:
            if pose is None or loc_age > p.loc_timeout:
                self.stop("localization timeout")
            else:
                return self._backing(cmd, dt, pose, obstacle_points)

        if self.state == self.DRIVING and self.avoiding:
            # the way ahead stays blocked: stop now, while there is still room to back out,
            # instead of creeping in to stop_d
            self.notfree_t = 0.0 if self.avoider.free else getattr(self, "notfree_t", 0.0) + dt
            if self.notfree_t > 0.5 and obstacle_dist < 2.0 and p.reverse_enable:
                obstacle_dist = min(obstacle_dist, stop_d)

        if self.state == self.DRIVING:
            if pose is None or loc_age > p.loc_timeout:
                self.stop("localization timeout")
            elif obstacle_dist <= stop_d:
                if p.obstacle_resume:
                    self.state, self.stop_reason = self.BLOCKED, obstacle_reason() + " - waiting"
                    self.blocked_t = 0.0
                else:
                    self.stop(obstacle_reason())

        if self.state != self.DRIVING:
            return self._stopped(cmd, dt)

        x, y, yaw = pose
        i, cte = self.route.nearest(x, y)
        if self.back_tries and self.route.s[i] > getattr(self, "back_s", 0.0) + 1.5:
            self.back_tries = 0                     # got well past that spot: allow backing up again later
        remaining = self.route.remaining(i)
        dist_goal = math.hypot(self.route.goal[0] - x, self.route.goal[1] - y)

        if cte > p.max_cross_track + (p.avoid_max_offset if self.avoiding else 0.0):
            self.stop(f"off route ({cte:.2f} m)")
            return self._stopped(cmd, dt)
        # arrival: close to B, or we have passed the end of the route
        if dist_goal < p.goal_tolerance or ((i >= len(self.route.pts) - 1 or remaining <= p.goal_tolerance)
                                            and dist_goal < 1.0):
            self.state = self.ARRIVED
            self.stop_reason = "arrived at B" if dist_goal < p.goal_tolerance + 0.15 else \
                f"arrived at B ({dist_goal:.2f} m away, came in from the side)"
            return self._stopped(cmd, dt)

        # ---- lateral: pure pursuit ---------------------------------------
        v = max(speed, 0.0)
        # with avoidance: the same look-ahead the planner used when it checked the drive
        v_ld = p.avoid_speed if self.avoiding else v
        ld = clamp(p.lookahead_min + p.lookahead_gain * v_ld, p.lookahead_min, p.lookahead_max)
        ld = min(ld, max(remaining, p.lookahead_min * 0.5))
        if self.avoiding:
            tx, ty = self.avoider.target(self.route, i, ld)
        else:
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
        if obstacle_dist < slow_d:
            frac = (obstacle_dist - stop_d) / (slow_d - stop_d)
            v_obs = p.cruise_speed * clamp(frac, 0.0, 1.0)
        v_ref = min(p.cruise_speed, v_curve, v_goal, v_obs)
        if self.avoiding and max(abs(self.avoider.d), abs(self.avoider.d_now)) > 0.15:
            v_ref = min(v_ref, p.avoid_speed)               # careful while on a detour
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
        if self.avoiding:
            cmd.info["avoid"] = round(float(self.avoider.d), 2) if self.avoider.kind == "offset" else "search"
        return cmd

    # -- back-up recovery ---------------------------------------------------
    def _rear_clear(self, pose, pts):
        """Nothing behind the truck within reverse_dist (+0.25 m) and the body width (+0.1 m)."""
        import numpy as np
        if pts is None or len(pts) == 0:
            return True
        p = self.p
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        q = np.asarray(pts, dtype=float).reshape(-1, 2) - [x, y]
        fx, fy = q[:, 0] * c + q[:, 1] * s, -q[:, 0] * s + q[:, 1] * c
        behind = (fx < -0.05) & (fx > -0.10 - p.reverse_dist - 0.25) & (np.abs(fy) < p.body_half_width + 0.10)
        return int(behind.sum()) < 2

    def _start_backing(self, pose, pts):
        import numpy as np
        p = self.p
        self.state, self.back_start, self.back_t = self.BACKING, (pose[0], pose[1]), 0.0
        self.back_tries += 1
        self.back_s = self.route.s[self.route._idx]
        # Reversing with the wheels turned RIGHT swings the nose LEFT (and the other way round).
        # Turn the nose away from what is in front: the side with more obstacle points loses.
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        q = np.asarray(pts, dtype=float).reshape(-1, 2) - [x, y]
        fx, fy = q[:, 0] * c + q[:, 1] * s, -q[:, 0] * s + q[:, 1] * c
        ahead = (fx > 0) & (fx < p.body_front + 1.0) & (np.abs(fy) < 1.0)
        side = float(np.mean(fy[ahead])) if ahead.any() else 0.0
        if abs(side) < 0.05:               # dead ahead: go round on the side the planner prefers
            side = -self.avoider.d if abs(self.avoider.d) > 0.1 else -1.0
        steer = -0.4 if side < 0 else 0.4             # obstacle right -> wheels right -> nose left
        # never swing the nose more than ~35 deg away from the route direction
        i = self.route._idx
        route_yaw = math.atan2(-self.route.nrm[i][0], self.route.nrm[i][1])
        dev = wrap_angle(yaw - route_yaw)                 # + = nose left of the route
        if (steer < 0 and dev > math.radians(35)) or (steer > 0 and dev < -math.radians(35)):
            steer = 0.0
        self.back_steer = steer
        self.stop_reason = f"no way forward - backing up {p.reverse_dist:.1f} m (try {self.back_tries}/{p.reverse_max_tries})"

    def _backing(self, cmd, dt, pose, pts):
        p = self.p
        self.back_t += dt
        moved = math.hypot(pose[0] - self.back_start[0], pose[1] - self.back_start[1])
        if moved >= p.reverse_dist or self.back_t > 6.0 or not self._rear_clear(pose, pts):
            self.state, self.blocked_t = self.BLOCKED, 0.0       # stop and plan again
            self.esc.state, self.esc.duty = DriveESC.NEUTRAL_, NEUTRAL
            self.stop_reason = "backed up - planning again"
            return self._stopped(cmd, dt)
        target = NEUTRAL + p.steer_sign * self.back_steer * 2.5
        max_step = p.steer_rate_limit * dt
        self.steer_duty += clamp(target - self.steer_duty, -max_step, max_step)
        # neutral briefly first (many ESCs need neutral between brake and reverse), then reverse
        drive = NEUTRAL if self.back_t < p.reverse_neutral_time else p.reverse_duty
        self.esc.state, self.esc.duty = DriveESC.NEUTRAL_, drive
        cmd.steer = (self.steer_duty - NEUTRAL) / 2.5 * p.steer_sign
        cmd.throttle, cmd.brake = 0.0, 0.0
        cmd.pwm = [round(self.steer_duty, 2), NEUTRAL, round(drive, 2)]
        cmd.state = self.state
        cmd.info = dict(reason=self.stop_reason)
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
        elif drive_duty <= self.p.brake_duty + 0.3:                 # brake
            self.v = max(0.0, self.v - 3.0 * dt) if self.v >= 0 else min(0.0, self.v + 3.0 * dt)
        elif drive_duty <= NEUTRAL - 0.2 and self.v <= 0.05:        # reverse (sim assumption)
            v_cmd = -(NEUTRAL - drive_duty) / 2.5 * self.top_speed
            self.v += (v_cmd - self.v) * dt / self.tau
        else:                                                       # coast
            self.v = max(0.0, self.v - 0.5 * dt) if self.v >= 0 else min(0.0, self.v + 0.5 * dt)
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw = wrap_angle(self.yaw + self.v / self.p.wheelbase * math.tan(delta) * dt)
        return self.x, self.y, self.yaw, self.v
