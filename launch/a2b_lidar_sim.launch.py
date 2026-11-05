"""Lidar localization test without hardware.

truck_sim (true pose on /sim/ground_truth) -> fake_lidar (PointCloud2 of a virtual room)
-> ground_filter -> KISS-ICP -> lidar_loc_bridge -> /localization/localization_data -> a2b_planner -> /pwm_duty
-> truck_sim. The planner only sees the lidar-based position. fake_lidar prints TRUE vs LIDAR. Live view: http://localhost:5000

Run with RMW_IMPLEMENTATION=rmw_fastrtps_cpp (point clouds are large). Start the run with:
  ros2 topic pub --once /a2b/enable std_msgs/Bool "data: true"
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("a2b_autodrive"), "config", "a2b_params.yaml")
    kiss = os.path.join(get_package_share_directory("kiss_icp"), "launch", "odometry.launch.py")
    L = LaunchConfiguration
    offset = 0.25
    return LaunchDescription([
        DeclareLaunchArgument("goal_x", default_value="10.0"),   # metres ahead
        DeclareLaunchArgument("goal_y", default_value="3.0"),    # metres left (makes it turn)
        DeclareLaunchArgument("port", default_value="5000"),     # live view: http://localhost:5000
        Node(package="a2b_autodrive", executable="truck_sim", name="truck_sim", output="screen",
             parameters=[{"localization_topic": "/sim/ground_truth", "start_x": 0.0, "start_y": 0.0,
                          "start_yaw_deg": 0.0}]),
        Node(package="a2b_autodrive", executable="fake_lidar", name="fake_lidar", output="screen",
             parameters=[{"lidar_x_offset": offset}]),
        # remove the floor before odometry (flat floors make KISS-ICP under-measure distance)
        Node(package="a2b_autodrive", executable="ground_filter", name="ground_filter", output="screen",
             parameters=[{"input_topic": "/fake_lidar/points", "output_topic": "/fake_lidar/points_no_ground",
                          "min_z": -0.3, "min_range": 0.3}]),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(kiss),
                                 launch_arguments={"topic": "/fake_lidar/points_no_ground", "visualize": "false",
                                                   "use_sim_time": "false"}.items()),
        Node(package="a2b_autodrive", executable="sim_viewer", name="sim_viewer", output="screen",
             parameters=[{"port": L("port"), "truth_topic": "/sim/ground_truth",
                          "estimate_topic": "/localization/localization_data",
                          "points_topic": "/fake_lidar/points", "show_room": True, "lidar_x_offset": offset}]),
        Node(package="a2b_autodrive", executable="lidar_loc_bridge", name="lidar_loc_bridge", output="screen",
             parameters=[{"lidar_x_offset": offset}]),
        # give KISS-ICP a few scans before the planner starts reading the position
        TimerAction(period=2.0, actions=[
            Node(package="a2b_autodrive", executable="a2b_planner", name="a2b_planner", output="screen",
                 parameters=[cfg, {"goal_x": L("goal_x"), "goal_y": L("goal_y"), "goal_relative": True,
                                   "publish_pwm": True, "map_frame": "odom_lidar"}])]),
    ])
