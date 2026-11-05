# a2b_autodrive — A→B algorithm for the mini dump truck

The A→B algorithm sits in the **Pnc** slot of the Guardian / WorldSim / Pnc simulation
architecture and uses the company's `robominer_msgs`, on the same topics Pnc uses.

```
 Guardian ── /guardian/plan_target (PlanTarget: B) ──────────┐
          ── /guardian/task_state  (TaskState: start/stop) ──┤
 Simmoulde ─ /localization/localization_data (LocalizationData) ─► a2b_planner ─► /auto/control_cmd (ControlCmd)
                                                                            └──► /pwm_duty (real truck)
```

## How the algorithm works

| Step | Method |
|---|---|
| Route | Current pose (A) → optional waypoints → B, resampled every 0.1 m |
| Steering | Pure Pursuit, look-ahead 0.6 m + 1.0 s × speed |
| Speed | min(cruise, curve limit, stopping distance to B), then PI + feed-forward |
| Safety | Stop if localization is invalid or older than 0.5 s, or > 1.5 m off route |
| Arrive | Within 0.3 m of B → brake, state ARRIVED |

## Interfaces (robominer_msgs)

| Direction | Topic | Message | Fields used |
|---|---|---|---|
| In | `/localization/localization_data` | `LocalizationData` | `x`, `y` (ENU m), `yaw` (ENU deg), `speed` (m/s), `is_valid` |
| In | `/guardian/plan_target` | `PlanTarget` | `goal_x`, `goal_y` (ENU m) = B |
| In | `/guardian/task_state` | `TaskState` | `START_TASK` (1) → drive; `PAUSING`/`PAUSE_FINISH`/`STOP_TASK`/`ERROR` → stop |
| Out | `/auto/control_cmd` | `ControlCmd` | AUTO mode, gear D (N when stopped), `target_steering` deg left +, `target_throttle`/`target_brake` 0–100 %, `target_speed` |
| Out | `/pwm_duty` | `Float32MultiArray` | [PWM1, PWM2, PWM3] % for the real truck's Modbus driver |

Guardian publishes `task_state` and `plan_target` **best effort**, so the planner subscribes best
effort (a reliable subscriber would receive nothing). For local tests, `/a2b/enable`
(`std_msgs/Bool`) and RViz `/goal_pose` also start/stop and set B.

### Truck PWM (Huzhou test, 50 Hz)
- PWM1 steering: 5 % full left, 7.5 % straight, 10 % full right.
- PWM2 bucket: always 7.5 % for A→B.
- PWM3 drive: forward 7.9–8.6 % (cautious cap), stop = 6.0 % brake for 0.6 s then 7.5 %. No reverse, never 5 %.

## Setup (WSL, ROS 2 Humble)

The messages come from the company install package (already compiled, no build needed):

```bash
cp -r /mnt/c/Users/Administrator/Downloads/test_install20261010_x86_ros2_3.0 ~/
echo 'source ~/test_install20261010_x86_ros2_3.0/install/robominer_msgs/share/robominer_msgs/local_setup.bash' >> ~/.bashrc
source ~/.bashrc
ros2 interface show robominer_msgs/msg/ControlCmd     # should print the fields

rm -rf ~/ros2_ws/src/a2b_autodrive
cp -r /mnt/c/Users/Administrator/Downloads/a2b_autodrive ~/ros2_ws/src/
cd ~/ros2_ws && colcon build --packages-select a2b_autodrive && source install/setup.bash
```

## 1. Offline test (no ROS)

```bash
python3 ~/ros2_ws/src/a2b_autodrive/test/offline_sim.py
```

## 2. Local ROS trial with the built-in truck sim

Starts the truck at the test case start point (327, −481), B = 10 m ahead:

```bash
ros2 launch a2b_autodrive a2b_sim.launch.py
# second terminal (source the same two setup files first):
ros2 topic pub --once /a2b/enable std_msgs/Bool "data: true"
ros2 topic echo /auto/control_cmd
```

Test the Guardian path instead of `/a2b/enable` (best effort, like Guardian):

```bash
ros2 topic pub --once --qos-reliability best_effort /guardian/plan_target robominer_msgs/msg/PlanTarget "{goal_x: 335.0, goal_y: -475.0}"
ros2 topic pub --once --qos-reliability best_effort /guardian/task_state robominer_msgs/msg/TaskState "{state: 1}"
# stop:
ros2 topic pub --once --qos-reliability best_effort /guardian/task_state robominer_msgs/msg/TaskState "{state: 26}"
```

## 3. WorldSim / SimOne trial (with 汪杰's Simmoulde running)

```bash
ros2 launch a2b_autodrive a2b_simone.launch.py
```
B comes from Guardian's `plan_target`, and the run starts on `task_state` = START_TASK.
`publish_pwm` is off in this launch so nothing goes to the real truck.

## Tuning notes
- `wheelbase`, `max_steer_deg`: measure on the truck.
- `sim_max_steer_deg`: the sim vehicle's full steering angle in `target_steering` degrees.
- Weaving → raise `lookahead_min`. Cutting corners → lower it.
- Check in the first sim run that yaw 0° = east (the truck should head towards B, not sideways).

## 4. Real mini truck with lidar localization

Position comes from the Airy lidar instead of a simulator:

```
Airy (rslidar_sdk) -> /rslidar_points -> ground_filter -> /points_no_ground -> KISS-ICP -> /kiss/odometry
   -> lidar_loc_bridge -> /localization/localization_data -> a2b_planner -> /pwm_duty -> Modbus driver
```

The odometry frame starts where the truck is when KISS-ICP starts (x = straight ahead, yaw 0 =
starting direction), so B is given as "metres ahead". Use Fast DDS for the point cloud:
`export RMW_IMPLEMENTATION=rmw_fastrtps_cpp` in every terminal.

```bash
# T1 lidar driver (rslidar_sdk, lidar_type RSAIRY)
ros2 launch rslidar_sdk start.py
# T2 lidar odometry
# once: build KISS-ICP from source (github.com/PRBonn/kiss-icp, needs CMake >= 3.24)
ros2 launch kiss_icp odometry.launch.py topic:=/points_no_ground visualize:=false use_sim_time:=false
# T3 Modbus PWM driver (existing package)
# T4 bridge + planner, B 3 m ahead; lidar_x_offset = lidar distance in front of rear axle
ros2 launch a2b_autodrive a2b_lidar.launch.py goal_x:=3.0 lidar_x_offset:=0.25 min_z:=-0.3   # min_z = -(lidar height - 0.1)
# T5 start
ros2 topic pub --once /a2b/enable std_msgs/Bool "data: true"
```

Check first by hand, with the truck's PWM driver off: carry or push the truck 1 m forward and
watch `ros2 topic echo /localization/localization_data --qos-reliability best_effort` - x should
go up by about 1.0 and yaw should stay near 0.

## 5. Lidar localization test without hardware

A fake lidar ray-casts a virtual room (walls, pillars, boxes) from the fake truck's true pose.
KISS-ICP works out the position from those clouds and the planner drives on that alone.

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch a2b_autodrive a2b_lidar_sim.launch.py          # B = 10 m ahead, 3 m left
ros2 topic pub --once /a2b/enable std_msgs/Bool "data: true"   # second terminal
```
`fake_lidar` prints `TRUE ... | LIDAR ... | error x.xx m` every 2 s.

## 6. Live view in the browser (Flask)

`a2b_sim.launch.py` and `a2b_lidar_sim.launch.py` also start `sim_viewer`, a small Flask page:
open **http://localhost:5000** in Chrome/Edge on Windows. It shows the truck, its trail, the
route and B, the lidar estimate and scan (lidar test), state, speed and PWM. Buttons start/stop
the run; click the map to set a new B. Install Flask once: `sudo apt install python3-flask`.
Turn it off with `viewer:=false` (a2b_sim) and change the port with `port:=5001`.

## 7. Lidar test in MuJoCo (3D yard, physics truck)

Same lidar chain as section 5, but the truck and the lidar live in a MuJoCo 3D yard
(checker floor, concrete barriers, boulders, light poles, container, material pile, excavator,
orange dump pad). The truck is driven by `/pwm_duty` through a planar force model with
collisions; the lidar on the cab roof (0.25 m ahead of the rear axle, 0.4 m high, 24 rings
-25..60 deg x 720) ray-casts every geom in the yard except the truck itself.

```
mujoco_sim -> /mujoco/points -> ground_filter -> KISS-ICP -> lidar_loc_bridge
           -> /localization/localization_data -> a2b_planner -> /pwm_duty -> mujoco_sim
```

```bash
pip install mujoco                                       # once
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch a2b_autodrive a2b_lidar_mujoco.launch.py     # B = 10 m ahead, 3 m left
ros2 topic pub --once /a2b/enable std_msgs/Bool "data: true"   # second terminal
```

The MuJoCo window shows the route (cyan), B (green), the true trail (mint), the lidar estimate
(amber arrow + trail) and the lidar returns (dots, coloured by height). It needs a display
(WSLg works); `gui:=false` runs headless. The browser view at http://localhost:5000 runs too
(`viewer:=false` to skip). `mujoco_sim` prints `TRUE ... | LIDAR ... | error x.xx m` every 2 s.

Offline, without ROS (same chain with the KISS-ICP python package, 4 scenarios incl. a lidar
dropout that must stop the truck):

```bash
pip install mujoco kiss-icp imageio imageio-ffmpeg pillow
python3 ~/ros2_ws/src/a2b_autodrive/test/mujoco_lidar_sim.py
python3 ~/ros2_ws/src/a2b_autodrive/test/mujoco_lidar_sim.py --video mujoco_lidar_sim.mp4   # also render a video
```
On a machine without a screen, add `MUJOCO_GL=egl` (or `osmesa`) in front of the video command.
