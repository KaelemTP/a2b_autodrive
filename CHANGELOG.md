# Changelog

## 0.4.4 (2026-11-16)

- Way-round search is now Hybrid A*: forward arcs within the truck's turning circle, so the
  paths it finds can really be driven (grid A* paths had corners the truck could not follow).
- Fixed: BLOCKED with the obstacle far away (e.g. "obstacle at 4.60 m - waiting") after a
  back-up. It now drives on when the nearest obstacle on the best option is over 2.5 m away.

## 0.4.3 (2026-11-16)

- Fixed: backing up never started in the lidar sim, because it waited for speed < 0.05 m/s and
  the lidar-based speed jitters 0.05-0.2 m/s even when standing still.
- Backing up starts sooner: 0.8 s after stopping (was 2 s), 0.2 s neutral before reverse (was 0.4).
- Stops early (room to back out) if the chosen way stays blocked for 0.5 s; up to 5 back-ups,
  count resets after 1.5 m of progress.
- Way-round search keeps a wider berth (cells within corridor + 0.4 m cost double).

## 0.4.2 (2026-11-16)

- Way-round search: when no sideways option is free, A* on a grid from the lidar points finds
  a way round (to B or further along the route); checked with the same simulated drive and
  kept while free. Hysteresis so options do not flicker.
- B blocked: if B is next to an obstacle and nothing gets through, it finishes within 2.5 m of B
  ("as close as it can get") or waits with a clear message, instead of an unexplained BLOCKED.
- Fixed: reaching the end of the route beside B (after a detour) left the truck at 0 m/s
  without ever counting as arrived.
- Viewer: **Reset truck to start** button (`/sim/reset`): truck_sim jumps back to the start,
  lidar_loc_bridge resets KISS-ICP (`/kiss/reset`), planner goes back to IDLE.

## 0.4.1 (2026-11-16)

- Fixed: the truck could end up wedged next to an obstacle (BLOCKED at 0.2 m) in tight layouts.
  Each avoidance option is now checked by simulating the drive from the truck's real position
  and heading (same steering, turning limit and steering speed) with the truck body as three
  circles, instead of checking a sideways-shifted copy of the route.
- After BLOCKED it only drives on when a whole option is free (no creeping back into the same spot).
- New: backing up when boxed in (state BACKING), `reverse_enable`, on in the lidar sim
  (`reverse:=true`), off on the real truck (reverse PWM not measured yet).
- Offline tests: narrow 0.7 m gap, and starting nose-to-pillar (backs up).

## 0.4.0 (2026-11-16)

- Obstacle avoidance: tries sideways-shifted paths (up to ±2 m) every cycle, takes the free one
  closest to the route, slows to 0.4 m/s on a detour and returns to the route after. Falls back
  to stop-and-wait (BLOCKED) when nothing gets through. `avoid_enable` / launch arg `avoid`.
  On by default in the lidar sim, **off by default on the real truck** (`a2b_lidar.launch.py`).
- `/a2b/local_path`: the path actually being followed.
- fake_lidar: obstacles can be added while running (`/sim/obstacles`, JSON).
- Viewer is now 3D (three.js, bundled): truck model, 3D lidar scan, follow / top / free camera,
  click tools to set B, waypoints, place / remove boxes and cylinders, contact counter.
- Offline tests: 4 new avoidance scenarios (box, slalom, wall across, obstacle on B).

## 0.3.0 (2026-11-16)

- Obstacle stop from the lidar point cloud: slows from 2.5 m, stops and waits at 1.0 m
  (state BLOCKED), drives on when the path is clear.
- Waypoints: from the `waypoints` setting or launch argument (relative to the start), or
  Shift+click in the viewer (`/a2b/waypoints`).
- Viewer: waypoints, obstacle distance, BLOCKED state, Clear waypoints button.
- README rewritten as a setup and test guide.

## 0.2.0 (2026-11-05)

- Lidar localization: KISS-ICP odometry → `lidar_loc_bridge` → `LocalizationData`.
- Ground filter before KISS-ICP (fixes distance being measured 5–22 % short on flat floors).
- Jump check: impossible lidar position jumps mark the position invalid so the truck stops.
- Fake lidar in a virtual room and a lidar simulation launch.
- Live browser view (Flask).

## 0.1.x (2026-10-27 to 2026-11-03)

- A→B algorithm (Pure Pursuit, speed PI, safety stops) using `robominer_msgs`.
- `/auto/control_debug` for Simmoulde; full-size SimOne truck settings.
- Huzhou PWM mapping for the real truck.
