"""Lidar localization test in MuJoCo (3D yard, physics truck) - no hardware.

mujoco_sim (truck + ray-cast lidar, /mujoco/points; true pose on /sim/ground_truth)
-> ground_filter -> KISS-ICP -> lidar_loc_bridge -> /localization/localization_data -> a2b_planner
-> /pwm_duty -> mujoco_sim. The planner only sees the lidar-based position.

The MuJoCo window (gui:=true) shows the route, B, the lidar returns and the lidar estimate in 3D.
The browser view is also started: http://localhost:5000 (viewer:=false to skip it).

Needs: pip install mujoco   (and KISS-ICP built, like a2b_lidar_sim.launch.py)
Run with RMW_IMPLEMENTATION=rmw_fastrtps_cpp. Start the run with:
  ros2 topic pub --once /a2b/enable std_msgs/Bool "data: true"
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("a2b_autodrive"), "config", "a2b_params.yaml")
    kiss = os.path.join(get_package_share_directory("kiss_icp"), "launch", "odometry.launch.py")
    L = LaunchConfiguration
    offset = 0.25   # lidar ahead of the rear axle (mujoco_world.LIDAR_X)
    return LaunchDescription([
        DeclareLaunchArgument("goal_x", default_value="10.0"),   # metres ahead
        DeclareLaunchArgument("goal_y", default_value="3.0"),    # metres left (makes it turn)
        DeclareLaunchArgument("gui", default_value="true"),      # MuJoCo 3D window
        DeclareLaunchArgument("shadows", default_value="false"), # MuJoCo window shadows (slow on WSL)
        DeclareLaunchArgument("viewer", default_value="true"),   # browser view
        DeclareLaunchArgument("port", default_value="5000"),
        Node(package="a2b_autodrive", executable="mujoco_sim", name="mujoco_sim", output="screen",
             parameters=[{"truth_topic": "/sim/ground_truth", "points_topic": "/mujoco/points",
                          "gui": L("gui"), "gui_shadows": L("shadows"), "start_x": 0.0, "start_y": 0.0, "start_yaw_deg": 0.0}]),
        # remove the floor before odometry (lidar 0.4 m above ground -> floor at z = -0.4)
        Node(package="a2b_autodrive", executable="ground_filter", name="ground_filter", output="screen",
             parameters=[{"input_topic": "/mujoco/points", "output_topic": "/mujoco/points_no_ground",
                          "min_z": -0.3, "min_range": 0.3}]),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(kiss),
                                 launch_arguments={"topic": "/mujoco/points_no_ground", "visualize": "false",
                                                   "use_sim_time": "false"}.items()),
        Node(package="a2b_autodrive", executable="sim_viewer", name="sim_viewer", output="screen",
             condition=IfCondition(L("viewer")),
             parameters=[{"port": L("port"), "truth_topic": "/sim/ground_truth",
                          "estimate_topic": "/localization/localization_data",
                          "points_topic": "/mujoco/points", "show_room": False, "lidar_x_offset": offset}]),
        Node(package="a2b_autodrive", executable="lidar_loc_bridge", name="lidar_loc_bridge", output="screen",
             parameters=[{"lidar_x_offset": offset}]),
        # give KISS-ICP a few scans before the planner starts reading the position
        TimerAction(period=2.0, actions=[
            Node(package="a2b_autodrive", executable="a2b_planner", name="a2b_planner", output="screen",
                 parameters=[cfg, {"goal_x": L("goal_x"), "goal_y": L("goal_y"), "goal_relative": True,
                                   "publish_pwm": True, "map_frame": "odom_lidar"}])]),
    ])
