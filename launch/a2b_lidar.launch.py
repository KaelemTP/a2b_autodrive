"""Real mini truck with lidar localization.

Start these first, each in its own terminal (RMW_IMPLEMENTATION=rmw_fastrtps_cpp for point clouds):
  1. Airy driver (rslidar_sdk)      -> /rslidar_points
  2. KISS-ICP lidar odometry on the floor-free cloud:
       ros2 launch kiss_icp odometry.launch.py topic:=/points_no_ground visualize:=false use_sim_time:=false
  3. Modbus PWM driver              <- /pwm_duty
Then this launch: ground_filter + lidar_loc_bridge + a2b_planner (small-truck config, B = goal_x m ahead).
Start the run with:  ros2 topic pub --once /a2b/enable std_msgs/Bool "data: true"
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("a2b_autodrive"), "config", "a2b_params.yaml")
    L = LaunchConfiguration
    return LaunchDescription([
        DeclareLaunchArgument("goal_x", default_value="3.0"),          # metres ahead (start short)
        DeclareLaunchArgument("goal_y", default_value="0.0"),          # metres left
        DeclareLaunchArgument("odom_topic", default_value="/kiss/odometry"),
        DeclareLaunchArgument("lidar_x_offset", default_value="0.0"),  # lidar ahead of rear axle, m
        DeclareLaunchArgument("lidar_topic", default_value="/rslidar_points"),
        DeclareLaunchArgument("min_z", default_value="-0.3"),       # about -(lidar height - 0.1 m)
        DeclareLaunchArgument("min_range", default_value="0.5"),    # m, ignore the truck body
        DeclareLaunchArgument("waypoints", default_value="[0.0]"),  # e.g. "[3.0, 0.0, 6.0, 2.0]" (x1,y1,x2,y2 metres ahead/left); [0.0] = none
        DeclareLaunchArgument("obstacle_stop", default_value="/points_no_ground"),  # "off" = no obstacle stop
        # Steering round obstacles is OFF on the real truck until it has been tried in the sim and
        # on a clear test area: it drives up to 2 m off the route. Off = stop and wait (v0.3).
        DeclareLaunchArgument("avoid", default_value="false"),
        Node(package="a2b_autodrive", executable="ground_filter", name="ground_filter", output="screen",
             parameters=[{"input_topic": L("lidar_topic"), "output_topic": "/points_no_ground",
                          "min_z": L("min_z"), "min_range": L("min_range")}]),
        Node(package="a2b_autodrive", executable="lidar_loc_bridge", name="lidar_loc_bridge", output="screen",
             parameters=[{"odom_topic": L("odom_topic"), "lidar_x_offset": L("lidar_x_offset")}]),
        Node(package="a2b_autodrive", executable="a2b_planner", name="a2b_planner", output="screen",
             parameters=[cfg, {"goal_x": L("goal_x"), "goal_y": L("goal_y"), "goal_relative": True,
                               "waypoints": L("waypoints"), "publish_pwm": True, "map_frame": "odom",
                               "obstacle_cloud_topic": L("obstacle_stop"), "lidar_x_offset": L("lidar_x_offset"),
                               "avoid_enable": PythonExpression(["'", L("avoid"), "'.lower() == 'true'"])}]),
    ])
