#!/usr/bin/env python3
"""Checks for the three things seen in the coworker's MuJoCo test video (2026-11-19).

    python3 test/video_checks.py

1. Way round a wide barrier: the steering must not slam from full lock one way to full lock
   the other (it did 10.0 -> 5.0 -> 10.0 -> 5.2 % in the video).
2. The simulated drive ("green line") must stop at B instead of carrying on and looping round it.
3. With a bigger real-truck margin (corridor_half_width 0.50) the truck still gets round the
   usual obstacles, and keeps more room.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from a2b_autodrive.core import A2BController, Params, TruckSim  # noqa: E402
from offline_sim import box_pts, circle_pts, clearance  # noqa: E402

DT = 0.05


def drive(route, obstacles, t_max=60.0, **params):
    p = Params(avoid_enable=True, **params)
    ctl = A2BController(p)
    sim = TruckSim(p, 0.0, 0.0, 0.0)
    ctl.set_route(route)
    log, paths = [], []
    t = 0.0
    while t < t_max:
        cmd = ctl.step((sim.x, sim.y, sim.yaw), sim.v, DT, float("inf"), 0.0, obstacle_points=obstacles)
        sim.step(cmd.pwm, DT)
        log.append((t, sim.x, sim.y, cmd.pwm[0], cmd.state, clearance(sim, obstacles)))
        paths.append(list(ctl.avoider.path))
        if cmd.state not in ("DRIVING", "BACKING") and abs(sim.v) < 0.01 and cmd.pwm[2] == 7.5:
            break
        t += DT
    return log, paths, cmd


def full_lock_flips(log, lo=5.6, hi=9.4):
    """How often the steering goes from near full lock one way to near full lock the other."""
    flips, last = 0, None
    for r in log:
        side = "L" if r[3] <= lo else "R" if r[3] >= hi else None
        if side and last and side != last:
            flips += 1
        if side:
            last = side
    return flips


def overshoot(paths, goal, direction):
    """Furthest any displayed simulated drive goes past B (metres, along the route direction)."""
    worst = 0.0
    for path in paths:
        for x, y in path:
            worst = max(worst, (x - goal[0]) * direction[0] + (y - goal[1]) * direction[1])
    return worst


if __name__ == "__main__":
    ok = True

    # 1. wide barrier across the route (wider than the +-2 m sideways options)
    barrier = np.array(box_pts(5.0, -2.3, 5.3, 2.3))
    log, paths, cmd = drive([(0, 0), (12, 0)], barrier)
    flips = full_lock_flips(log)
    travel = sum(abs(b[3] - a[3]) for a, b in zip(log[:-1], log[1:]))
    gap = min(r[5] for r in log)
    print(f"1 wide barrier      state={cmd.state:8s} full-lock flips={flips}  steering travel={travel:.1f} %"
          f"  min_gap={gap:.2f} m")
    ok &= cmd.state == "ARRIVED" and gap > 0.12      # keeps (nearly) the full 0.15 m sim margin
    ok &= flips <= 1

    # 2. green line must stop at B (straight route along +x, box on the route)
    box = np.array(box_pts(4.5, -0.4, 5.3, 0.4))
    log, paths, cmd = drive([(0, 0), (10, 0)], box)
    over = overshoot(paths, (10.0, 0.0), (1.0, 0.0))
    print(f"2 path stops at B   state={cmd.state:8s} drawn path goes {over:.2f} m past B")
    ok &= cmd.state == "ARRIVED" and over <= 0.35

    # 3. real-truck margin: corridor_half_width 0.50
    for name, route, obs in [
            ("box", [(0, 0), (10, 0)], box),
            ("slalom", [(0, 0), (14, 0)], np.array(box_pts(3.5, -0.6, 4.1, 0.2) + circle_pts(7.5, 0.3, 0.3)
                                                    + box_pts(10.5, -0.3, 11.1, 0.5))),
            ("wide barrier", [(0, 0), (12, 0)], barrier)]:
        log, paths, cmd = drive(route, obs, corridor_half_width=0.50)
        gap = min(r[5] for r in log)
        print(f"3 margin 0.50 {name:12s} state={cmd.state:8s} min_gap={gap:.2f} m")
        ok &= cmd.state == "ARRIVED" and gap > 0.30

    print("all video checks passed" if ok else "SOME VIDEO CHECKS FAILED")
    sys.exit(0 if ok else 1)
