"""Local trial: kinematic truck sim (publishes LocalizationData) + A->B planner (+ RViz with rviz:=true).
Starts the truck at the WorldSim test case start point; B defaults to 10 m straight ahead."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("a2b_autodrive"), "config", "a2b_params.yaml")
    L = LaunchConfiguration
    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("start_x", default_value="327.0"),
        DeclareLaunchArgument("start_y", default_value="-481.0"),
        DeclareLaunchArgument("start_yaw_deg", default_value="30.0"),
        DeclareLaunchArgument("goal_x", default_value="10.0"),   # relative: metres ahead
        DeclareLaunchArgument("goal_y", default_value="0.0"),    # relative: metres left
        DeclareLaunchArgument("viewer", default_value="true"),   # live view: http://localhost:5000
        DeclareLaunchArgument("port", default_value="5000"),
        Node(package="a2b_autodrive", executable="truck_sim", name="truck_sim", output="screen",
             parameters=[{"start_x": L("start_x"), "start_y": L("start_y"),
                          "start_yaw_deg": L("start_yaw_deg")}]),
        Node(package="a2b_autodrive", executable="a2b_planner", name="a2b_planner", output="screen",
             parameters=[cfg, {"goal_x": L("goal_x"), "goal_y": L("goal_y"), "goal_relative": True}]),
        Node(package="a2b_autodrive", executable="sim_viewer", name="sim_viewer", output="screen",
             condition=IfCondition(L("viewer")),
             parameters=[{"port": L("port"), "truth_topic": "/localization/localization_data",
                          "estimate_topic": "/localization/localization_data", "points_topic": "",
                          "show_room": False}]),
        Node(package="rviz2", executable="rviz2", name="rviz2", condition=IfCondition(L("rviz"))),
    ])
