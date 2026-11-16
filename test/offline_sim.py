#!/usr/bin/env python3
"""Offline closed-loop test of the A->B algorithm, no ROS needed.

    python3 test/offline_sim.py            # runs the scenarios, saves offline_sim.png

Scenarios 6-9 test obstacle avoidance: the controller gets obstacle points in the map frame
(what the planner node builds from the lidar) and must drive around them without touching.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from a2b_autodrive.core import A2BController, Params, TruckSim, nearest_obstacle_ahead  # noqa: E402

DT = 0.05  # 20 Hz control loop


def to_vehicle_frame(pts_world, x, y, yaw):
    c, s = math.cos(-yaw), math.sin(-yaw)
    out = []
    for px, py, pz in pts_world:
        dx, dy = px - x, py - y
        out.append((c * dx - s * dy, s * dx + c * dy, pz))
    return out


TRUCK_HALF_WIDTH = 0.20   # m, assumed body half width for the touch check
TRUCK_REAR, TRUCK_FRONT = -0.10, 0.45   # m from the rear axle


def box_pts(x0, y0, x1, y1, step=0.05):
    """Outline points of a box obstacle (what a lidar would hit)."""
    out = []
    for t in np.arange(0, 1.0001, step / max(x1 - x0, y1 - y0)):
        out += [(x0 + t * (x1 - x0), y0), (x0 + t * (x1 - x0), y1), (x0, y0 + t * (y1 - y0)), (x1, y0 + t * (y1 - y0))]
    return out


def circle_pts(cx, cy, r, n=40):
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, n, endpoint=False)]


def clearance(sim, obs):
    """Smallest gap between the truck body and the obstacle points (negative = touching)."""
    c, s = math.cos(sim.yaw), math.sin(sim.yaw)
    rel = obs - [sim.x, sim.y]
    fx, fy = rel[:, 0] * c + rel[:, 1] * s, -rel[:, 0] * s + rel[:, 1] * c   # truck frame
    dx = np.maximum(np.maximum(TRUCK_REAR - fx, fx - TRUCK_FRONT), 0)
    dy = np.maximum(np.abs(fy) - TRUCK_HALF_WIDTH, 0)
    inside = (dx == 0) & (dy == 0)
    gap = np.hypot(dx, dy)
    return -1.0 if inside.any() else float(gap.min())


def run(name, route, start, obstacle=None, drop_loc_at=None, t_max=60.0, avoid_pts=None, avoid=True,
        reverse=False):
    p = Params(avoid_enable=avoid, reverse_enable=reverse)
    ctl = A2BController(p)
    sim = TruckSim(p, *start)
    ctl.set_route(route)
    log = []
    t = 0.0
    cmd = None
    while t < t_max:
        pose = (sim.x, sim.y, sim.yaw)
        loc_age = 0.0
        if drop_loc_at and t >= drop_loc_at:
            loc_age = t - drop_loc_at
        obs = float("inf")
        if obstacle:
            pts = to_vehicle_frame(obstacle, sim.x, sim.y, sim.yaw)
            steer = (cmd.steer if cmd else 0.0) * math.radians(p.max_steer_deg)
            obs = nearest_obstacle_ahead(pts, p, steer)
        if avoid_pts is not None:
            cmd = ctl.step(pose, sim.v, DT, float("inf"), loc_age, obstacle_points=avoid_pts)
        else:
            cmd = ctl.step(pose, sim.v, DT, obs, loc_age)
        for d in cmd.pwm:  # hard safety checks on every output
            assert 5.0 <= d <= 10.0, d
        assert cmd.pwm[1] == 7.5, "bucket must stay at 7.5 %"
        assert cmd.pwm[2] >= 6.0, "drive must never go below the brake value"
        if not reverse:
            assert sim.v >= -1e-6, "reversed although reverse_enable is off"
        sim.step(cmd.pwm, DT)
        gap = clearance(sim, avoid_pts) if avoid_pts is not None and len(avoid_pts) else float("inf")
        log.append((t, sim.x, sim.y, sim.v, *cmd.pwm, cmd.state, gap))
        if cmd.state not in ("DRIVING", "BACKING") and abs(sim.v) < 0.01 and cmd.pwm[2] == 7.5 \
                and not (reverse and cmd.state == "BLOCKED" and t < 25.0):
            break
        t += DT
    gx, gy = route[-1]
    err = math.hypot(sim.x - gx, sim.y - gy)
    print(f"{name:28s} state={cmd.state:8s} reason='{cmd.info.get('reason','')}' "
          f"t={t:5.1f}s  final=({sim.x:.2f},{sim.y:.2f})  dist_to_B={err:.2f} m  "
          f"max_v={max(r[3] for r in log):.2f} m/s"
          + (f"  min_gap={min(r[8] for r in log):.2f} m" if avoid_pts is not None else ""))
    return log, cmd


if __name__ == "__main__":
    scenarios = [
        ("1 straight A->B 10 m", [(0, 0), (10, 0)], (0, 0, 0.0), {}),
        ("2 start 30 deg off, offset", [(0, 0), (8, 4)], (0, -0.5, math.radians(-30)), {}),
        ("3 L-turn via waypoint", [(0, 0), (6, 0), (6, 6)], (0, 0, 0.0), {}),
        ("4 box obstacle in path", [(0, 0), (10, 0)], (0, 0, 0.0),
         {"obstacle": [(5.0 + dx * 0.05, dy * 0.05, 0.3) for dx in range(4) for dy in range(-4, 5)]}),
        ("5 localization lost at 4 s", [(0, 0), (10, 0)], (0, 0, 0.0), {"drop_loc_at": 4.0}),
        ("6 avoid box in path", [(0, 0), (10, 0)], (0, 0, 0.0),
         {"avoid_pts": np.array(box_pts(4.5, -0.4, 5.3, 0.4))}),
        ("7 avoid slalom (3 obstacles)", [(0, 0), (14, 0)], (0, 0, 0.0),
         {"avoid_pts": np.array(box_pts(3.5, -0.6, 4.1, 0.2) + circle_pts(7.5, 0.3, 0.3)
                                + box_pts(10.5, -0.3, 11.1, 0.5))}),
        ("8 wall across road -> wait", [(0, 0), (10, 0)], (0, 0, 0.0),
         {"avoid_pts": np.array(box_pts(5.0, -12.0, 5.3, 12.0))}),
        ("9 obstacle on B -> stop near B", [(0, 0), (8, 0)], (0, 0, 0.0),
         {"avoid_pts": np.array(circle_pts(8.0, 0.0, 0.3))}),
        ("10 narrow gate (0.7 m)", [(0, 0), (9, 0)], (0, 0, 0.0),
         {"avoid_pts": np.array(circle_pts(4.5, 0.65, 0.3) + circle_pts(4.5, -0.65, 0.3))}),
        ("11 starts nose-to-pillar, backs up", [(0, 0), (6, 0)], (0, 0, 0.0),
         {"avoid_pts": np.array(circle_pts(1.05, 0.0, 0.3)), "reverse": True}),
    ]
    logs = []
    for name, route, start, kw in scenarios:
        log, cmd = run(name, route, start, **kw)
        logs.append((name, route, log, cmd.info.get("reason", ""), kw))

    # expected outcomes
    assert logs[1][2][-1][7] == "ARRIVED"
    assert logs[2][2][-1][7] == "ARRIVED"
    S = lambda k: logs[k][2][-1][7]  # noqa: E731  final state
    assert S(0) == "ARRIVED"
    assert S(3) == "BLOCKED" and logs[3][2][-1][1] < 5.0   # (avoidance off) waits in front of the box
    assert S(4) == "STOPPED"
    for k in (5, 6):                                       # drives around, never touches
        assert S(k) == "ARRIVED", (k, S(k))
        assert min(r[8] for r in logs[k][2]) > 0.05, (k, "too close")
    for k in (9, 10):                                      # squeezes through / backs up, never touches
        assert S(k) == "ARRIVED", (k, S(k))
        assert min(r[8] for r in logs[k][2]) > 0.05, (k, "too close")
    assert min(r[3] for r in logs[10][2]) < -0.05, "scenario 11 should have reversed"
    assert S(7) == "BLOCKED"                               # wall across: waits
    assert S(8) == "ARRIVED" and "B is blocked" in logs[8][3]   # obstacle on B: as close as it can
    for k in (7, 8):                                       # never touches
        assert min(r[8] for r in logs[k][2]) > 0.3, (k, "too close")
    print("all scenario checks passed")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        for ax, (name, route, log, _, kw) in zip(axes.flat, logs):
            rx, ry = zip(*route)
            ax.plot(rx, ry, "k--", label="route A->B")
            ax.plot([r[1] for r in log], [r[2] for r in log], "b-", label="truck")
            ax.plot(rx[0], ry[0], "go", ms=10, label="A")
            ax.plot(rx[-1], ry[-1], "r*", ms=14, label="B")
            if kw.get("obstacle"):
                ox, oy, _ = zip(*kw["obstacle"])
                ax.plot(ox, oy, "s", color="orange", ms=3, label="obstacle")
            if kw.get("avoid_pts") is not None:
                ax.plot(kw["avoid_pts"][:, 0], kw["avoid_pts"][:, 1], ".", color="orange", ms=3, label="obstacle")
            ax.set_title(name)
            ax.axis("equal")
            ax.grid(True)
        ax = axes.flat[11]
        log = logs[1][2]
        ax.plot([r[0] for r in log], [r[4] for r in log], label="PWM1 steer %")
        ax.plot([r[0] for r in log], [r[6] for r in log], label="PWM3 drive %")
        ax.plot([r[0] for r in log], [r[3] * 10 for r in log], label="speed x10 (m/s)")
        ax.axhline(7.5, color="grey", lw=0.8)
        ax.set_title("scenario 2: PWM outputs")
        ax.set_xlabel("t (s)")
        ax.legend(fontsize=8)
        ax.grid(True)
        axes.flat[0].legend(fontsize=8)
        for ax in axes.flat[12:]:
            ax.axis("off")
        fig.tight_layout()
        out = os.path.join(os.path.dirname(__file__), "offline_sim.png")
        fig.savefig(out, dpi=90)
        print("plot saved to", out)
    except ImportError:
        pass
