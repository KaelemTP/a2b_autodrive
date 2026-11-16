# a2b_autodrive

Drives the mini dump truck from where it is now (A) to a goal point (B), optionally through
waypoints, steering around obstacles on the way, and stops there. ROS 2 Humble, Python. Version **0.4.4**.

It runs in three setups with the same algorithm:

| Setup | Truck | Position comes from | Launch file |
|---|---|---|---|
| **Quick sim** | Built-in fake truck | The fake truck (perfect) | `a2b_sim.launch.py` |
| **Lidar sim** | Built-in fake truck + fake lidar in a virtual room | Fake lidar → KISS-ICP (realistic) | `a2b_lidar_sim.launch.py` |
| **WorldSim / SimOne** | Company simulator | Simmoulde | `a2b_simone.launch.py` |
| **Real mini truck** | Truck via BT-PW112 PWM module | Airy lidar → KISS-ICP | `a2b_lidar.launch.py` |

The quick sim and lidar sim come with a live **3D view** in the browser at **http://localhost:5000**.
In the lidar sim you can drop boxes and cylinders in front of the truck there and watch it drive round them.

---

## 1. What it does

Every 0.05 s:

1. **Reads the position** (`/localization/localization_data`: x, y in metres, yaw in degrees, speed).
2. **Follows the route** A → waypoints → B with **Pure Pursuit**: it aims at a point on the route
   0.6 m + 1.0 s × speed ahead and steers onto it.
3. **Picks a speed**: cruise 0.6 m/s, slower in curves, slower near B, slower near obstacles.
   A PI controller turns that into throttle.
4. **Checks safety** and stops (brakes) if any of these happen:
   - position missing or older than 0.5 s, or the lidar position jumps impossibly fast
   - more than 1.5 m off the route
   - a stop command (`/a2b/enable false`, or Guardian `task_state` stop/pause/error)
5. **Obstacles** (lidar setups): every cycle it considers 17 options: stay on the route, or move
   0.25, 0.5 … 2.0 m to the left or right of it. For each option it **simulates the drive** for
   the next 5 m from where the truck is and the way it points, with the same steering, turning
   limit and steering speed as the real controller. The truck body (three circles along it) must
   keep 0.35 m (`corridor_half_width`) from the centre line to every lidar point (a bit less in
   the first metre, so a truck that is already close can still steer away). It takes the free
   option closest to the route, prefers keeping the same side, slows to 0.4 m/s while off the
   route and comes back once past.
   **Way-round search**: if none of those options is free (e.g. B behind a cluster of
   obstacles), it searches a way round with **Hybrid A\***: only forward arcs the truck can
   actually drive (within 80 % of its tightest turn), on a 10 cm grid made from the lidar points,
   to B or to a free point further along the route. It checks that path with the same simulated
   drive and sticks with it while it stays free. A search takes ~20 ms, at most 0.3 s. The viewer then shows **Detour: way round**.
   If nothing is free it slows and stops 0.4 m before it would get too close (state
   **BLOCKED**), and drives on when an option is free again or the nearest obstacle on the best
   option is more than 2.5 m away (then it drives closer and plans again).
   **B blocked**: if nothing gets through and B is within 0.7 m of an obstacle (the lidar only
   sees the near side of things), it treats B as blocked. Within 2.5 m of B it finishes:
   `ARRIVED as close as it can get: 1.96 m from B (B is blocked by an obstacle)`. Further away it
   waits with `B is blocked by an obstacle - waiting (move B or the obstacle)`.
   **Backing up** (lidar sim only, `reverse:=true`): if it is BLOCKED for 0.8 s with no way
   forward, it reverses 0.6 m (state **BACKING**), turning its nose away from the obstacle (at
   most 35° off the route), and plans again. Up to 5 times (the count resets after 1.5 m of
   progress), then it waits. If the way it picked stays blocked for 0.5 s while driving, it
   stops early so there is still room to back out.
   With `avoid:=false` it only stops and waits (v0.3 behaviour). **The real-truck launch has
   avoidance and backing up off by default.**
6. **Arrives** within 0.3 m of B (state **ARRIVED**).

States: `IDLE` → `DRIVING` ⇄ `BLOCKED` ⇄ `BACKING` → `ARRIVED` or `STOPPED`.

Outputs, both every cycle:

- `/auto/control_cmd` (`robominer_msgs/ControlCmd`) and `/auto/control_debug` (`ControlDebug`)
  for the company simulator: AUTO mode, gear D/N, steering in degrees (left +), throttle and
  brake 0–100 %. Simmoulde needs both messages.
- `/pwm_duty` (`[PWM1, PWM2, PWM3]` %) for the real truck's Modbus driver, using the Huzhou
  values at 50 Hz: steering 5 / 7.5 / 10 %, bucket held at 7.5 %, drive 7.9–8.6 % forward,
  stop = 6.0 % brake for 0.6 s then 7.5 %. Never sends 5 %. Reverses only with
  `reverse_enable` (off by default): 7.5 % for 0.2 s, then `reverse_duty` (7.0 %, a guess:
  **the real ESC's reverse value has not been measured**).

## 2. Install (once)

Ubuntu 22.04 (or WSL) with ROS 2 Humble.

**a) Company messages (`robominer_msgs`).** Either build them from the `robotruck/common` repo
(branch `v_20261025`), or use the compiled install package:

```bash
source <install package>/install/robominer_msgs/share/robominer_msgs/local_setup.bash
# the install package does not set the Python path, add it:
export PYTHONPATH=<install package>/install/robominer_msgs/local/lib/python3.10/dist-packages:$PYTHONPATH
ros2 interface show robominer_msgs/msg/ControlCmd      # should print the fields
```

Put both lines in `~/.bashrc` so every terminal has them.

**b) This package**

```bash
sudo apt install python3-flask
cd ~/ros2_ws/src && git clone https://github.com/KaelemTP/a2b_autodrive.git
cd ~/ros2_ws && colcon build --packages-select a2b_autodrive && source install/setup.bash
```

**c) KISS-ICP** (lidar setups only). Not in apt for Humble, build from source. It needs
CMake ≥ 3.24 (Ubuntu 22.04 has 3.22):

```bash
sudo apt install -y python3-pip libeigen3-dev libtbb-dev
python3 -m pip install --user "cmake==3.28.3" && export PATH=$HOME/.local/bin:$PATH
cd ~/ros2_ws/src && git clone https://github.com/PRBonn/kiss-icp.git
cd ~/ros2_ws && colcon build --packages-select kiss_icp --cmake-args -DCMAKE_BUILD_TYPE=Release
```

**d) Middleware.** Point clouds are large; use Fast DDS in **every** terminal of the lidar setups:

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

All terminals must use the same `RMW_IMPLEMENTATION` and `ROS_DOMAIN_ID`, or they cannot see
each other.

## 3. Run a test

### Quick sim

```bash
ros2 launch a2b_autodrive a2b_sim.launch.py              # B = 10 m ahead
```

Open http://localhost:5000 and press **Start** (or `ros2 topic pub --once /a2b/enable std_msgs/Bool "data: true"`).

### Lidar sim (recommended)

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch a2b_autodrive a2b_lidar_sim.launch.py        # B = 10 m ahead, 3 m left
```

Open http://localhost:5000 (3D view):

- **Reset truck to start**: stops the truck and puts it back where it started (the fake truck
  jumps back and KISS-ICP is reset, so the lidar position starts again at 0, 0). Obstacles, B
  and waypoints stay; press Start to run again. Needs a KISS-ICP with the `/kiss/reset` service
  (any version from May 2026 on); with an older one, restart the launch instead.
- Mouse: **drag** to rotate, **right-drag** to move, **scroll** to zoom. Camera buttons:
  **Follow** (behind the truck), **Top**, **Free**.
- Pick a **click tool**, then click the ground: **Set B**, **Waypoint** (also Shift+click),
  **Box**, **Cylinder** (size slider), **Remove** (nearest obstacle you placed). **Start** / **Stop**.
- Yellow truck = true position, amber box outline = lidar estimate, coloured dots = lidar scan
  (blue low → yellow high), dashed blue = route, green = path it is actually following.
- **Detour** shows how far left/right of the route it is heading. **Contacts** counts the truck
  body touching anything; it should stay 0 (the truck turns red and a CONTACT banner shows).
- Things to try: a box on the route (drives round it); several cylinders in a slalom or a 0.7 m
  gap; a cylinder dropped right in front of the moving truck (BLOCKED, then BACKING, then round
  it); a row of boxes across the room (BLOCKED; remove one and it drives on); a box on B.

Placed obstacles go out on `/sim/obstacles` and fake_lidar adds them to the scan, so the planner
only knows about them through the lidar, as on the real truck. The quick sim has no lidar, so
the obstacle tools are greyed out there.

Waypoints can also be given at launch, in metres ahead/left of the start, as decimals:

```bash
ros2 launch a2b_autodrive a2b_lidar_sim.launch.py goal_x:=12.0 goal_y:=0.0 waypoints:="[4.0, 3.0, 8.0, -2.0]"
```

### WorldSim / SimOne

```bash
ros2 launch a2b_autodrive a2b_simone.launch.py
```

Uses `config/a2b_params_simone.yaml` (full-size truck). In Simmoulde's `config.ini` set
`[Version] Version=PncSim`, `[TaskState] State=1` and `[DriverMode] driver_mode=1`; Simmoulde then
sends B (the test case's `end_point`) and the start command itself. Turn off Pnc so only one
node publishes `/auto/control_cmd`. B should be on a straight road for now (see limits).

### Real mini truck

1. Airy driver: `ros2 launch rslidar_sdk start.py` (publishes `/rslidar_points`)
2. Odometry: `ros2 launch kiss_icp odometry.launch.py topic:=/points_no_ground visualize:=false use_sim_time:=false`
3. The Modbus PWM driver (listens to `/pwm_duty`)
4. `ros2 launch a2b_autodrive a2b_lidar.launch.py goal_x:=3.0 lidar_x_offset:=<lidar ahead of rear axle, m> min_z:=<-(lidar height - 0.1)>`
5. `ros2 topic pub --once /a2b/enable std_msgs/Bool "data: true"`

First push the truck by hand with the PWM driver off: `x` should go up by about 1.0 per metre.
Then lift the wheels, then a 3 m run. Keep the GameSir **B** (all channels 7.5 %) ready.

## 4. What "working" looks like

| Check | Expected |
|---|---|
| Log after start | `driving A(...) -> B(...)`, then `IDLE -> DRIVING` |
| Arrival | `DRIVING -> ARRIVED arrived at B`, truck within about 0.3 m of B |
| Lidar sim, viewer | Amber outline on the white truck; "Lidar error" under about 0.2 m |
| Obstacle in the way (lidar sim) | Green path bends round it, **Detour** shows e.g. `1.00 m right`, then `on route`; Contacts 0 |
| No way past | `DRIVING -> BLOCKED obstacle at 0.40 m - waiting`, then `BLOCKED -> DRIVING` when clear |
| Boxed in (lidar sim) | `BLOCKED -> BACKING no way forward - backing up 0.6 m (try 1/3)`, then `BLOCKED -> DRIVING` |
| Stop button / stop task | `DRIVING -> STOPPED disabled by /a2b/enable` (or `task_state 26`) |
| `/auto/control_cmd` while driving | `mode: 2`, `gear: 3`, `target_throttle` above 0, `target_brake: 0.0` |

## 5. Settings you may need to change

All in `config/a2b_params.yaml` (small truck) or `config/a2b_params_simone.yaml` (SimOne truck).
Rebuild after editing (`colcon build --packages-select a2b_autodrive`).

| Setting | Default (small) | What it is |
|---|---|---|
| `wheelbase` | 0.30 m | Front to rear axle. **Estimated, measure it** |
| `max_steer_deg` | 30° | Front-wheel angle at full lock. **Estimated** |
| `cruise_speed` | 0.6 m/s | Normal speed |
| `drive_forward_min` / `max` | 7.9 / 8.6 % | Lowest PWM that moves the truck / cautious top limit |
| `lookahead_min` | 0.6 m | Raise if it weaves, lower if it cuts corners |
| `goal_tolerance` | 0.3 m | Counted as arrived |
| `max_cross_track` | 1.5 m | Off route → stop |
| `obstacle_stop_dist` / `slow_dist` | 1.0 / 2.5 m | From the rear axle |
| `corridor_half_width` | 0.35 m | Half the truck width + margin |
| `obstacle_resume` | true | false = stop for good at an obstacle |
| `avoid_enable` | true (sim) | Steer round obstacles. Launch argument `avoid:=true/false`; real-truck launch defaults to false |
| `avoid_max_offset` | 2.0 m | Furthest a detour goes from the route. Lower it on narrow roads |
| `avoid_ramp` | 2.0 m | Travel used to move sideways. Larger = gentler swerve |
| `avoid_clearance` | 0.2 m | Extra gap it prefers beyond `corridor_half_width` |
| `avoid_speed` | 0.4 m/s | Speed while on a detour |
| `avoid_stop_dist` | 0.4 m | Stop this much driving before it would get too close |
| `body_front` / `body_half_width` | 0.45 / 0.20 m | Truck size used for the clearance check. **Measure** |
| `reverse_enable` | false (sim: true) | Back up when boxed in. Launch argument `reverse:=true/false` in the lidar sim |
| `goal_blocked_arrive` | 2.5 m | If B is blocked, finish when this close |
| `reverse_duty` | 7.0 % | PWM3 for slow reverse. **Not measured on the real ESC**: find it with the GameSir before turning this on |

## 6. If something goes wrong, send back

1. The launch terminal output from the start (screenshot or `> log.txt`).
2. A recording of the run:
   ```bash
   ros2 bag record -o a2b_run /localization/localization_data /auto/control_cmd /pwm_duty /a2b/status /a2b/path /a2b/local_path /sim/obstacles
   ```
   (add `/sim/ground_truth /kiss/odometry` in the lidar sim)
3. A screenshot of the viewer if you used it.

Common problems:

| Symptom | Cause |
|---|---|
| `ros2 node list` empty in a second terminal | Different `RMW_IMPLEMENTATION` / `ROS_DOMAIN_ID` between terminals |
| `ModuleNotFoundError: robominer_msgs` | `PYTHONPATH` line from section 2a missing |
| Lidar estimate spins in circles | Lidar sim launched without Fast DDS, or two copies running (`ros2 node list`) |
| Same node listed twice | An old launch still running (Ctrl+Z instead of Ctrl+C): `pkill -9 -f a2b_autodrive` |
| Planner `STOPPED off route` | Steering settings wrong for the vehicle (`wheelbase`, `max_steer_deg`) |
| SimOne truck does not move | Pnc still running, or Simmoulde not getting `/auto/control_debug` |
| Viewer page blank / "could not start 3D" | Browser without WebGL: use Chrome or Edge with hardware acceleration on |
| Viewer works in WSL (`curl -s localhost:5000/state`) but not in Windows | Windows firewall; try `http://<WSL IP>:5000` (`hostname -I`), else ask IT |
| Truck drives through a placed obstacle | Quick sim (no lidar), or `obstacle_stop:=off` / `avoid:=false` |

## 7. Known limits

- **Route = straight lines** between A, the waypoints and B. It does not follow roads from the
  map yet; give waypoints along the road. Following the company router's reference line
  (`/routing/navigation`) is the next step.
- **Forward only**, no reversing or three-point turns. B behind the truck gives a tight U-turn.
- **Lidar position is relative to the start** and slowly drifts (lidar sim: under 0.1 m over
  10 m; real world will be worse, especially in open areas with few features nearby).
- **Truck dimensions are estimates** until measured on the real truck.
- **Obstacles** are only what the lidar sees right now: no map, so it cannot plan round something
  hidden behind another obstacle, and it only looks 5 m ahead and 2 m to each side.
- **Detours stay near the route** (±2 m). It does not search for a completely different way, and
  it does not know where the road edge is: on a narrow road set `avoid_max_offset` smaller.
- **Boxed in**: on the real truck (no backing up) it waits in BLOCKED if it is pointing at an
  obstacle closer than about 0.3 m. In the sim it backs up, but after 3 tries it also waits, e.g.
  when a waypoint makes it turn back right beside a box. Press Stop and move B / the obstacle.
- **Moving obstacles** (people walking) are treated as still; it re-plans 20 times a second,
  so it reacts, but does not predict where they are going.
- The SimOne big-truck settings (`a2b_params_simone.yaml`) have not been tested in a full run.

## 8. Files

| File | What it is |
|---|---|
| `a2b_autodrive/core.py` | The algorithm (route, Pure Pursuit, speed, safety, PWM). No ROS |
| `a2b_autodrive/a2b_planner_node.py` | ROS 2 node around the algorithm |
| `a2b_autodrive/lidar_loc_bridge_node.py` | KISS-ICP odometry → `LocalizationData`, with jump check |
| `a2b_autodrive/ground_filter_node.py` | Removes floor and truck-body points before KISS-ICP |
| `a2b_autodrive/truck_sim_node.py` | Fake truck driven by `/pwm_duty` |
| `a2b_autodrive/fake_lidar_node.py` | Fake lidar in a virtual room |
| `a2b_autodrive/sim_viewer_node.py` | Live 3D view in the browser (Flask), obstacle placing, contact check |
| `web/` | The 3D page (`index.html`) and three.js (MIT licence), served locally, no internet needed |
| `test/offline_sim.py` | Offline checks of the algorithm without ROS: `python3 test/offline_sim.py` |

See `CHANGELOG.md` for what changed in each version.
