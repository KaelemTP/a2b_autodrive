#!/usr/bin/env python3
"""Offline closed-loop test of the A->B algorithm, no ROS needed.

    python3 test/offline_sim.py            # runs 4 scenarios, saves offline_sim.png
"""
import math
import os
import sys

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


def run(name, route, start, obstacle=None, drop_loc_at=None, t_max=60.0):
    p = Params()
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
        cmd = ctl.step(pose, sim.v, DT, obs, loc_age)
        for d in cmd.pwm:  # hard safety checks on every output
            assert 5.0 <= d <= 10.0, d
        assert cmd.pwm[1] == 7.5, "bucket must stay at 7.5 %"
        assert cmd.pwm[2] >= 6.0, "drive must never go below brake value (no reverse)"
        sim.step(cmd.pwm, DT)
        log.append((t, sim.x, sim.y, sim.v, *cmd.pwm, cmd.state))
        if cmd.state != "DRIVING" and sim.v < 0.01 and cmd.pwm[2] == 7.5:
            break
        t += DT
    gx, gy = route[-1]
    err = math.hypot(sim.x - gx, sim.y - gy)
    print(f"{name:28s} state={cmd.state:8s} reason='{cmd.info.get('reason','')}' "
          f"t={t:5.1f}s  final=({sim.x:.2f},{sim.y:.2f})  dist_to_B={err:.2f} m  "
          f"max_v={max(r[3] for r in log):.2f} m/s")
    return log, cmd


if __name__ == "__main__":
    scenarios = [
        ("1 straight A->B 10 m", [(0, 0), (10, 0)], (0, 0, 0.0), {}),
        ("2 start 30 deg off, offset", [(0, 0), (8, 4)], (0, -0.5, math.radians(-30)), {}),
        ("3 L-turn via waypoint", [(0, 0), (6, 0), (6, 6)], (0, 0, 0.0), {}),
        ("4 box obstacle in path", [(0, 0), (10, 0)], (0, 0, 0.0),
         {"obstacle": [(5.0 + dx * 0.05, dy * 0.05, 0.3) for dx in range(4) for dy in range(-4, 5)]}),
        ("5 localization lost at 4 s", [(0, 0), (10, 0)], (0, 0, 0.0), {"drop_loc_at": 4.0}),
    ]
    logs = []
    for name, route, start, kw in scenarios:
        log, cmd = run(name, route, start, **kw)
        logs.append((name, route, log, kw))

    # expected outcomes
    assert logs[0][2][-1][-1] == "ARRIVED"
    assert logs[1][2][-1][-1] == "ARRIVED"
    assert logs[2][2][-1][-1] == "ARRIVED"
    assert logs[3][2][-1][-1] == "STOPPED" and logs[3][2][-1][1] < 5.0
    assert logs[4][2][-1][-1] == "STOPPED"
    print("all scenario checks passed")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, (name, route, log, kw) in zip(axes.flat, logs):
            rx, ry = zip(*route)
            ax.plot(rx, ry, "k--", label="route A->B")
            ax.plot([r[1] for r in log], [r[2] for r in log], "b-", label="truck")
            ax.plot(rx[0], ry[0], "go", ms=10, label="A")
            ax.plot(rx[-1], ry[-1], "r*", ms=14, label="B")
            if kw.get("obstacle"):
                ox, oy, _ = zip(*kw["obstacle"])
                ax.plot(ox, oy, "s", color="orange", ms=3, label="obstacle")
            ax.set_title(name)
            ax.axis("equal")
            ax.grid(True)
        ax = axes.flat[5]
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
        fig.tight_layout()
        out = os.path.join(os.path.dirname(__file__), "offline_sim.png")
        fig.savefig(out, dpi=90)
        print("plot saved to", out)
    except ImportError:
        pass
