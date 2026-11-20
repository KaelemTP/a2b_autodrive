#!/usr/bin/env python3
"""8 obstacle scenarios in the MuJoCo yard with the 3D lidar and KISS-ICP, no ROS needed.

Same chain as a2b_lidar_mujoco.launch.py, in one process:

    MuJoCo truck + test obstacles -> ray-cast lidar sweep (10 Hz)
      -> ground filter (z > -0.3 m, range >= 0.3 m) -> KISS-ICP -> rear-axle pose (lidar_loc_bridge)
      -> obstacle points in the map frame (like a2b_planner) -> A2BController (20 Hz) -> PWM

The controller only sees the lidar: its position comes from KISS-ICP and its obstacles from the
lidar points. "gap" = smallest distance between the truck body (rear 0.10 m, front 0.45 m,
half width 0.20 m) and the real obstacle outlines; "contacts" = times the MuJoCo chassis hit
something.

    pip install mujoco kiss-icp imageio imageio-ffmpeg pillow
    python3 test/mujoco_scenarios.py                         # checks only (about 2 min)
    MUJOCO_GL=egl python3 test/mujoco_scenarios.py --video test/mujoco_scenarios.mp4
    python3 test/mujoco_scenarios.py --only 4 --video s4.mp4  # one scenario
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from a2b_autodrive.core import A2BController, Params  # noqa: E402
from a2b_autodrive import mujoco_world as mw  # noqa: E402
from mujoco_lidar_sim import Bridge, make_kiss  # noqa: E402
from offline_sim import box_pts, circle_pts, clearance  # noqa: E402

DT = 0.05            # 20 Hz control loop, like a2b_planner
LIDAR_EVERY = 2      # lidar at 10 Hz
GROUND_MIN_Z = -0.3  # ground_filter min_z
MIN_RANGE = 0.3      # ground_filter min_range
VERSION = "v0.4.7"
TRACE = os.environ.get("A2B_TRACE") == "1"


def box(x0, y0, x1, y1, h=0.8):
    return ("box", x0, y0, x1, y1, h)


def cyl(cx, cy, r, h=1.0):
    return ("cyl", cx, cy, r, h)


# name, subtitle, route (start frame = world, truck starts at 0,0 facing +x), obstacles, options, expected
SCENARIOS = [
    ("Waypoint Turn", "No obstacles: drive to a waypoint, turn left onto the orange pad (B)",
     [(0, 0), (6, 0), (8.5, 3.5)], [], {}, "ARRIVED"),
    ("Single Crate", "0.8 m crate on the route -> sideways option round it, back onto the route",
     [(0, 0), (10, 0)], [box(4.6, -0.4, 5.4, 0.4)], {}, "ARRIVED"),
    ("Slalom (3 obstacles)", "Box left of the route, pillar right, box left -> weave through",
     [(0, 0), (12.5, 0)], [box(3.5, -0.6, 4.1, 0.2), cyl(7.5, 0.3, 0.3), box(10.3, -0.3, 10.9, 0.5)],
     {}, "ARRIVED"),
    ("Hybrid A* Way-Round Search", "Wide 4.6 m barrier blocks all +-2.0 m offset paths -> Hybrid A* finds a drivable way round",
     [(0, 0), (10, 0)], [box(4.6, -2.0, 4.9, 2.6, h=0.9)], {}, "ARRIVED"),
    ("Pillar Gate (0.75 m)", "Two pillars 0.75 m apart on the route: needs 0.70 m (2 x corridor 0.35 m)",
     [(0, 0), (9, 0)], [cyl(4.8, 0.625, 0.25), cyl(4.8, -0.625, 0.25)], {}, "ARRIVED"),
    ("Boxed In -> Back Up", "Starts with its nose 0.45 m from a pillar: backs up, then drives round",
     [(0, 0), (6, 0)], [cyl(1.05, 0.0, 0.3)], {"reverse": True}, "ARRIVED"),
    ("B Blocked", "A crate sits on B -> stop as close as it can get (within 2.5 m) with a clear message",
     [(0, 0), (7, 0)], [box(6.5, -0.5, 7.5, 0.5)], {}, "ARRIVED"),
    ("LiDAR Dropout", "Lidar stops at t = 6 s -> localization timeout -> brake and stop",
     [(0, 0), (10, 0)], [], {"dropout_at": 6.0}, "STOPPED"),
]


def obstacles_xml(obs):
    out = []
    for i, o in enumerate(obs):
        if o[0] == "box":
            _, x0, y0, x1, y1, h = o
            out.append(f'<geom name="obs{i}" type="box" pos="{(x0 + x1) / 2} {(y0 + y1) / 2} {h / 2}" '
                       f'size="{(x1 - x0) / 2} {(y1 - y0) / 2} {h / 2}" material="obstacle"/>')
        else:
            _, cx, cy, r, h = o
            out.append(f'<geom name="obs{i}" type="cylinder" pos="{cx} {cy} {h / 2}" size="{r} {h / 2}" '
                       f'material="obstacle"/>')
    return "\n".join(out)


def obstacles_outline(obs):
    pts = []
    for o in obs:
        pts += box_pts(*o[1:5], step=0.02) if o[0] == "box" else circle_pts(*o[1:4], n=120)
    return np.array(pts) if pts else np.zeros((0, 2))


def nearest_obstacle(p, xyz, steer_duty):
    """a2b_planner.nearest_obstacle: nearest point in the (steering-bent) corridor ahead, m."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    steer = (steer_duty - 7.5) / 2.5 * p.steer_sign * math.radians(p.max_steer_deg)
    k = math.tan(steer) / p.wheelbase
    hit = (x > 0.0) & (x <= p.obstacle_slow_dist) & (z >= p.obstacle_z_min) & (z <= p.obstacle_z_max) \
        & (np.abs(y - 0.5 * k * x * x) <= p.corridor_half_width)
    xs = np.sort(x[hit])
    return float(xs[p.obstacle_min_points - 1]) if len(xs) >= p.obstacle_min_points else float("inf")


class Pose:
    def __init__(self, x, y, yaw):
        self.x, self.y, self.yaw = x, y, yaw


STATE_COLORS = {"DRIVING": (46, 204, 113), "BACKING": (243, 156, 18), "ARRIVED": (52, 152, 219),
                "BLOCKED": (231, 76, 60), "STOPPED": (231, 76, 60), "IDLE": (127, 140, 141)}


class Video:
    """Offscreen MuJoCo render with a header (scenario, state) and a footer (live numbers)."""

    def __init__(self, path, w=1280, h=624, fps=40):
        import mujoco
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont
        self.mujoco, self.Image, self.ImageDraw = mujoco, Image, ImageDraw
        self.w, self.h = w, h
        self.renderer = None
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 9.5, -36.0, 62.0
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1)
        mono = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
        bold = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
        self.f_title, self.f_sub = ImageFont.truetype(bold, 24), ImageFont.truetype(mono, 15)
        self.f_foot, self.f_small = ImageFont.truetype(mono, 15), ImageFont.truetype(mono, 13)
        self.rng = np.random.default_rng(0)
        self._look = None

    def reset(self, world):
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = self.mujoco.Renderer(world.model, self.h, self.w, max_geom=8000)
        self._look = None

    def frame(self, world, head, sub, state, route_w, goal_w, detour_w, est_trail, scan, left, right):
        x, y, _ = world.pose
        tgt = np.array([0.6 * x + 0.4 * goal_w[0], 0.6 * y + 0.4 * goal_w[1], 0.0])
        self._look = tgt if self._look is None else self._look + 0.08 * (tgt - self._look)
        self.cam.lookat[:] = self._look
        r = self.renderer
        r.update_scene(world.data, self.cam)
        scn = r.scene
        mw.draw_polyline(scn, route_w, (0.2, 0.95, 1.0, 1.0), width=0.02)
        mw.draw_sphere(scn, goal_w[0], goal_w[1], 0.15, 0.15, (0.3, 0.95, 0.45, 1.0))
        if len(detour_w) > 1:
            mw.draw_polyline(scn, detour_w, (0.45, 1.0, 0.25, 1.0), width=0.022, z=0.05)
        if len(est_trail) > 1:
            mw.draw_polyline(scn, est_trail[::2] + [est_trail[-1]], (1.0, 0.72, 0.25, 1.0), width=0.015, z=0.035)
        mw.draw_scan(scn, scan, max_points=1600, rng=self.rng)
        img = r.render()
        top, bot = 66, 70
        cv = self.Image.new("RGB", (self.w, self.h + top + bot), (17, 24, 32))
        cv.paste(self.Image.fromarray(img), (0, top))
        d = self.ImageDraw.Draw(cv)
        d.text((18, 8), head, fill=(235, 240, 245), font=self.f_title)
        d.text((18, 40), sub, fill=(170, 180, 190), font=self.f_sub)
        col = STATE_COLORS.get(state, (127, 140, 141))
        d.rounded_rectangle((self.w - 166, 13, self.w - 18, 51), radius=6, fill=col)
        tw = d.textlength(state, font=self.f_foot)
        d.text((self.w - 92 - tw / 2, 23), state, fill=(20, 25, 30), font=self.f_foot)
        yb = self.h + top
        d.text((18, yb + 12), left, fill=(235, 240, 245), font=self.f_foot)
        rw = d.textlength(right, font=self.f_foot)
        d.text((self.w - 18 - rw, yb + 12), right, fill=(255, 196, 90), font=self.f_foot)
        d.text((18, yb + 42), f"a2b_autodrive {VERSION}  |  MuJoCo {self.mujoco.__version__} + 3D LiDAR (10 Hz) -> KISS-ICP",
               fill=(120, 130, 140), font=self.f_small)
        legend = "Cyan: route  |  Green: detour  |  Amber: KISS-ICP  |  Red dots: lidar  |  2x speed"
        lw = d.textlength(legend, font=self.f_small)
        d.text((self.w - 18 - lw, yb + 42), legend, fill=(120, 130, 140), font=self.f_small)
        self.writer.append_data(np.asarray(cv))

    def close(self):
        self.writer.close()
        if self.renderer is not None:
            self.renderer.close()


def run(idx, video=None, t_max=60.0):
    name, sub, route, obs, opts, _ = SCENARIOS[idx]
    p = Params(avoid_enable=True, reverse_enable=opts.get("reverse", False))
    world = mw.MujocoTruckWorld(p, extra_xml=obstacles_xml(obs))
    outline = obstacles_outline(obs)
    chassis = world.model.geom("chassis").id
    obs_ids = {world.model.geom(f"obs{i}").id for i in range(len(obs))}
    kiss, bridge, ctl = make_kiss(), Bridge(mw.LIDAR_X), A2BController(p)
    ctl.set_route(route)
    dropout = opts.get("dropout_at")
    if video:
        video.reset(world)
    head = f"[{idx + 1}/{len(SCENARIOS)}]  {idx + 1}. {name}"
    xyz = np.zeros((0, 3))
    est_trail, errs, gaps = [], [], []
    contacts, touching, backs = 0, False, 0
    t, k, cmd, prev_state, blocked_for = 0.0, 0, None, None, 0.0
    left = right = ""
    while t < t_max:
        if k % LIDAR_EVERY == 0 and not (dropout is not None and t >= dropout):
            pts = world.scan().astype(np.float64)
            rng_ = np.hypot(pts[:, 0], pts[:, 1])
            pts = pts[(pts[:, 2] > GROUND_MIN_Z) & (rng_ >= MIN_RANGE)]
            kiss.register_frame(pts, np.empty((0,)))
            bridge.update(kiss.last_pose, t)
            xyz = pts.copy()
            xyz[:, 0] += mw.LIDAR_X                  # lidar frame -> rear-axle frame (a2b_planner)
        loc_age = t - bridge.t
        obst_pts, obst_dist = None, float("inf")
        if bridge.pose is not None and len(xyz):
            sel = xyz[(xyz[:, 2] >= p.obstacle_z_min) & (xyz[:, 2] <= p.obstacle_z_max)]
            bx, by, byaw = bridge.pose
            c, s = math.cos(byaw), math.sin(byaw)
            obst_pts = np.stack([bx + c * sel[:, 0] - s * sel[:, 1], by + s * sel[:, 0] + c * sel[:, 1]], 1)
            obst_dist = nearest_obstacle(p, xyz, ctl.steer_duty)
        cmd = ctl.step(bridge.pose, bridge.speed, DT, obst_dist, loc_age, obstacle_points=obst_pts)
        world.step(cmd.pwm, DT)
        t += DT
        k += 1
        if cmd.state == "BACKING" and prev_state != "BACKING":
            backs += 1
        prev_state = cmd.state
        if TRACE and k % 10 == 0:
            print(f'{t:5.1f} {cmd.state:8s} pose={world.pose[0]:.2f},{world.pose[1]:.2f} v={world.speed:+.2f} pwm={cmd.pwm[0]:.1f},{cmd.pwm[2]:.1f} kind={ctl.avoider.kind} d={ctl.avoider.d:+.2f} reason={cmd.info.get("reason","")}')

        x, y, yaw = world.pose
        hit = any((world.data.contact[i].geom1 == chassis and world.data.contact[i].geom2 in obs_ids) or
                  (world.data.contact[i].geom2 == chassis and world.data.contact[i].geom1 in obs_ids)
                  for i in range(world.data.ncon))
        contacts += int(hit and not touching)
        touching = hit
        gap = clearance(Pose(x, y, yaw), outline) if len(outline) else float("inf")
        gaps.append(gap)
        if bridge.pose is not None:
            est_trail.append(bridge.pose[:2])
            if loc_age < 0.15:
                errs.append(math.hypot(bridge.pose[0] - x, bridge.pose[1] - y))

        if video:
            gx, gy = route[-1]
            av = ctl.avoider
            if not ctl.avoiding or cmd.state not in ("DRIVING", "BACKING"):
                detour, det_txt = [], "none"
            else:
                detour = list(av.path)
                det_txt = "Hybrid A*" if av.kind == "search" else (
                    "none" if abs(av.d) < 1e-6 else f"offset {av.d:+.2f} m")
            lost = " (lidar lost)" if loc_age > p.loc_timeout else ""
            left = (f"t={t:4.1f}s  |  v={world.speed:+.2f}m/s  |  PWM=[{cmd.pwm[0]:.1f},{cmd.pwm[2]:.1f}]%  |  "
                    f"dist_B={math.hypot(gx - x, gy - y):.2f}m  |  detour={det_txt}{lost}")
            g_txt = "-" if not np.isfinite(gap) else f"{gap:.2f}m (min {min(gaps):.2f})"
            right = (f"ICP_err={errs[-1]:.3f}m" if errs else "ICP_err=-") + f"  |  gap={g_txt}  |  contacts={contacts}"
            video.frame(world, head, sub, cmd.state, route, (gx, gy), detour, est_trail,
                        world.last_scan_world, left, right)
            if k % 40 == 0:
                print(f"  [{idx + 1}] {t:4.1f} s simulated, {cmd.state}", flush=True)
        blocked_for = blocked_for + DT if cmd.state == "BLOCKED" else 0.0
        if (cmd.state in ("ARRIVED", "STOPPED") or blocked_for > 8.0) and abs(world.speed) < 0.01:
            break
    if video:   # hold the last frame 1.5 s, with the result in the subtitle
        reason = cmd.info.get("reason", "")
        for _ in range(60):
            video.frame(world, head, f"Result: {cmd.state} - {reason}", cmd.state, route, route[-1], [],
                        est_trail, world.last_scan_world, left, right)

    x, y, _ = world.pose
    gx, gy = route[-1]
    e = np.array(errs) if errs else np.array([0.0])
    res = dict(state=cmd.state, reason=cmd.info.get("reason", ""), t=t, dist_b=math.hypot(gx - x, gy - y),
               err_max=float(e.max()), gap=min(gaps) if gaps else float("inf"), contacts=contacts, backs=backs)
    g = "-" if not np.isfinite(res["gap"]) else f"{res['gap']:.2f}"
    print(f"{idx + 1} {name:28s} {res['state']:8s} t={t:5.1f}s dist_B={res['dist_b']:.2f} m "
          f"ICP max={res['err_max']:.3f} m min_gap={g} m contacts={contacts} back-ups={backs}  '{res['reason']}'",
          flush=True)
    return res


def check(idx, res):
    expected = SCENARIOS[idx][5]
    ok = res["state"] == expected and res["contacts"] == 0 and res["err_max"] < 0.3
    if idx == 6:                                   # B blocked: finish near B, not at B
        ok &= "as close as it can get" in res["reason"] and res["dist_b"] < 2.6
    elif expected == "ARRIVED":
        ok &= res["dist_b"] < 0.5
    if idx == 5:
        ok &= res["backs"] >= 1
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="write an mp4 of the scenarios to this path")
    ap.add_argument("--only", type=int, nargs="*", help="scenario numbers (1-8), default all")
    args = ap.parse_args()
    which = [i - 1 for i in args.only] if args.only else list(range(len(SCENARIOS)))
    vid = None
    if args.video:
        os.environ.setdefault("MUJOCO_GL", "egl")
        vid = Video(args.video)
    fails = []
    try:
        for i in which:
            if not check(i, run(i, vid)):
                fails.append(i + 1)
    finally:
        if vid:
            vid.close()
            print("video saved to", args.video, flush=True)
    print("all MuJoCo scenarios passed" if not fails else f"FAILED scenarios: {fails}")
    sys.exit(1 if fails else 0)
