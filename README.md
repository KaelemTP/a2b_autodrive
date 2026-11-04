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
