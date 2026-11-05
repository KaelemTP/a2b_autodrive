#!/usr/bin/env python3
"""Offline MuJoCo + lidar closed-loop test, no ROS needed.

Same chain as a2b_lidar_mujoco.launch.py, in one process:

    MuJoCo truck (planar force model) -> ray-cast lidar sweep (10 Hz)
      -> ground filter (z > -0.3 m) -> KISS-ICP (python) -> rear-axle pose, like lidar_loc_bridge
      -> A2BController (20 Hz) -> PWM -> MuJoCo truck

The controller only sees the lidar-based position. Prints true vs lidar error per scenario and
checks the outcome. With --video it renders an mp4 (needs an OpenGL backend: on a headless
machine set MUJOCO_GL=osmesa or egl).

    pip install mujoco kiss-icp imageio imageio-ffmpeg pillow
    python3 test/mujoco_lidar_sim.py                      # checks only
    MUJOCO_GL=egl python3 test/mujoco_lidar_sim.py --video test/mujoco_lidar_sim.mp4
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from a2b_autodrive.core import A2BController, Params, wrap_angle  # noqa: E402
from a2b_autodrive import mujoco_world as mw  # noqa: E402

DT = 0.05            # 20 Hz control loop, like a2b_planner
LIDAR_EVERY = 2      # lidar at 10 Hz
GROUND_MIN_Z = -0.3  # ground_filter min_z in the launch file
MIN_RANGE = 0.3


def make_kiss():
    from kiss_icp.config import load_config
    from kiss_icp.kiss_icp import KissICP
    cfg = load_config(None)            # same defaults as the KISS-ICP ROS launch (max_range 100)
    cfg.data.deskew = False            # the simulated sweep is instantaneous
    return KissICP(cfg)


class Bridge:
    """lidar_loc_bridge in plain python: lidar pose in KISS odom -> rear axle, start at (0, 0)."""

    def __init__(self, dx, alpha=0.4):
        self.dx, self.alpha = dx, alpha
        self.pose, self.speed, self.last, self.t = None, 0.0, None, -1e9

    def update(self, T, t):
        yaw = math.atan2(T[1, 0], T[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        x = T[0, 3] - c * self.dx + self.dx
        y = T[1, 3] - s * self.dx
        if self.last is not None and t - self.last[0] > 1e-3:
            v = math.hypot(x - self.last[1], y - self.last[2]) / (t - self.last[0])
            self.speed += self.alpha * (v - self.speed)
        self.last = (t, x, y)
        self.pose, self.t = (x, y, yaw), t


SCENARIOS = [
    # name, route in the start frame (metres ahead, left), lidar dropout time, expected state
    ("Straight arrival", [(0, 0), (8, 0)], None, "ARRIVED"),
    ("Turning arrival", [(0, 0), (5, 0), (8, 3.5)], None, "ARRIVED"),
    ("Diagonal 10 m / 3 m", [(0, 0), (10, 3)], None, "ARRIVED"),
    ("Lidar dropout", [(0, 0), (8, 0)], 5.0, "STOPPED"),
]


class Video:
    """Offscreen MuJoCo render + header/footer text bars, like the earlier MuJoCo test video."""

    def __init__(self, world, path, w=960, h=544, fps=40):
        import mujoco
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont
        self.mujoco, self.Image, self.ImageDraw = mujoco, Image, ImageDraw
        self.w, self.h = w, h
        self.renderer = mujoco.Renderer(world.model, h, w, max_geom=8000)
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 8.5, -38.0, 60.0
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1)
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        try:
            self.f1, self.f2 = ImageFont.truetype(font, 20), ImageFont.truetype(font, 17)
        except OSError:
            self.f1 = self.f2 = ImageFont.load_default()
        self.rng = np.random.default_rng(0)

    def reset(self, world):
        self.renderer.close()
        self.renderer = self.mujoco.Renderer(world.model, self.h, self.w, max_geom=8000)

    def frame(self, world, name, start, route_odom, truth_trail, est_trail, est_pose, scan, footer, err_txt):
        gx, gy = mw.odom_to_world(start, *route_odom[-1], 0.0)[:2]
        x, y, _ = world.pose
        # camera between the truck and B, eased
        tgt = np.array([0.6 * x + 0.4 * gx, 0.6 * y + 0.4 * gy, 0.0])
        if not hasattr(self, "_look") or self._look is None:
            self._look = tgt
        self._look = self._look + 0.08 * (tgt - self._look)
        self.cam.lookat[:] = self._look
        r = self.renderer
        r.update_scene(world.data, self.cam)
        scn = r.scene
        route_w = [mw.odom_to_world(start, px, py, 0.0)[:2] for px, py in route_odom]
        mw.draw_polyline(scn, route_w, (0.2, 0.95, 1.0, 1.0), width=0.022)
        mw.draw_sphere(scn, gx, gy, 0.15, 0.15, (0.3, 0.95, 0.45, 1.0))
        if len(truth_trail) > 1:
            mw.draw_polyline(scn, truth_trail[::3] + [truth_trail[-1]], (0.55, 1.0, 0.8, 1.0), width=0.015, z=0.02)
        if len(est_trail) > 1:
            mw.draw_polyline(scn, est_trail[::2] + [est_trail[-1]], (1.0, 0.72, 0.25, 1.0), width=0.015, z=0.045)
        if est_pose is not None:
            ex, ey, eyaw = est_pose
            mw.draw_arrow(scn, ex, ey, eyaw, (1.0, 0.72, 0.25, 1.0), length=0.5, z=0.55)
        mw.draw_scan(scn, scan, max_points=1800, rng=self.rng)
        img = r.render()
        bar_t, bar_b = 62, 34
        canvas = self.Image.new("RGB", (self.w, self.h + bar_t + bar_b), (17, 24, 32))
        canvas.paste(self.Image.fromarray(img), (0, bar_t))
        d = self.ImageDraw.Draw(canvas)
        d.text((12, 8), f"A2B controller + MuJoCo + LiDAR  |  {name}", fill=(235, 240, 245), font=self.f1)
        d.text((12, 36), "Planar force model  |  ray-cast lidar -> ground filter -> KISS-ICP  |  "
               "Playback 2x", fill=(222, 190, 120), font=self.f2)
        d.text((12, self.h + bar_t + 7), footer, fill=(235, 240, 245), font=self.f2)
        d.text((self.w - 330, self.h + bar_t + 7), err_txt, fill=(255, 190, 90), font=self.f2)
        self.writer.append_data(np.asarray(canvas))

    def close(self):
        self.writer.close()
        self.renderer.close()


def run(name, route, dropout_at, video=None, start=(0.0, 0.0, 0.0), t_max=45.0):
    p = Params()
    world = mw.MujocoTruckWorld(p, start=start)
    kiss = make_kiss()
    bridge = Bridge(mw.LIDAR_X)
    ctl = A2BController(p)
    ctl.set_route(route)
    if video:
        video.reset(world)
        video._look = None
    truth_trail, est_trail, errs = [], [], []
    cmd, k, t = None, 0, 0.0
    pwm = [7.5, 7.5, 7.5]
    while t < t_max:
        if k % LIDAR_EVERY == 0 and not (dropout_at is not None and t >= dropout_at):
            pts = world.scan().astype(np.float64)
            r = np.hypot(pts[:, 0], pts[:, 1])
            pts = pts[(pts[:, 2] > GROUND_MIN_Z) & (r >= MIN_RANGE)]
            kiss.register_frame(pts, np.empty((0,)))
            bridge.update(kiss.last_pose, t)
        loc_age = t - bridge.t
        cmd = ctl.step(bridge.pose, bridge.speed, DT, loc_age=loc_age)
        pwm = cmd.pwm
        world.step(pwm, DT)
        t += DT
        k += 1

        x, y, yaw = world.pose
        truth_trail.append((x, y))
        est_w = None
        if bridge.pose is not None:
            est_w = mw.odom_to_world(start, *bridge.pose)
            est_trail.append(est_w[:2])
            if loc_age < 0.15:    # only while the lidar is running (not the frozen dropout pose)
                errs.append(math.hypot(est_w[0] - x, est_w[1] - y))
        if video and k % 1 == 0:
            gx, gy = mw.odom_to_world(start, *route[-1], 0.0)[:2]
            lost = " (lidar lost)" if loc_age > p.loc_timeout else ""
            footer = (f"{t:4.1f}s  {cmd.state}{lost}   speed {world.speed:.2f} m/s   "
                      f"goal {math.hypot(gx - x, gy - y):.2f} m")
            err_txt = f"lidar position error {errs[-1]:.3f} m" if errs else "lidar: waiting"
            video.frame(world, name, start, route, truth_trail, est_trail, est_w,
                        world.last_scan_world, footer, err_txt)
        if video and k % 40 == 0:     # rendering is slow: show it is still working
            print(f"  {name}: {t:4.1f} s simulated, {cmd.state}", flush=True)
        if cmd.state != "DRIVING" and abs(world.speed) < 0.01 and pwm[2] == 7.5:
            break
    if video:   # hold the last frame for a second
        for _ in range(20):
            video.frame(world, name, start, route, truth_trail, est_trail, est_w, world.last_scan_world,
                        footer, err_txt)

    gx, gy = mw.odom_to_world(start, *route[-1], 0.0)[:2]
    x, y, yaw = world.pose
    dist_b = math.hypot(gx - x, gy - y)
    e = np.array(errs)
    print(f"{name:22s} state={cmd.state:8s} reason='{cmd.info.get('reason', '')}' t={t:5.1f}s "
          f"true dist to B={dist_b:.2f} m  lidar err mean={e.mean():.3f} max={e.max():.3f} m  "
          f"yaw err={math.degrees(abs(wrap_angle(est_w[2] - yaw))):.1f} deg")
    return cmd.state, dist_b, e


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="write an mp4 of all scenarios to this path")
    args = ap.parse_args()
    vid = None
    if args.video:
        os.environ.setdefault("MUJOCO_GL", "egl")
        world0 = mw.MujocoTruckWorld()
        vid = Video(world0, args.video)
    results = []
    try:
        for name, route, drop, expected in SCENARIOS:
            if vid:
                print(f"rendering '{name}' ...", flush=True)
            state, dist_b, errs = run(name, route, drop, vid)
            results.append((name, state, expected, dist_b, errs))
    finally:   # also on Ctrl+C or an error: finish the mp4 so it plays
        if vid:
            vid.close()
            print("video saved to", args.video, "" if len(results) == len(SCENARIOS) else
                  f"(incomplete: {len(results)} of {len(SCENARIOS)} scenarios)", flush=True)
    for name, state, expected, dist_b, errs in results:
        assert state == expected, f"{name}: {state} != {expected}"
        assert errs.max() < 0.3, f"{name}: lidar error {errs.max():.2f} m"
        if expected == "ARRIVED":
            assert dist_b < 0.5, f"{name}: stopped {dist_b:.2f} m from B"
    print("all MuJoCo lidar checks passed")
