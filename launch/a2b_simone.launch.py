"""WorldSim / SimOne run: planner only, in the Pnc slot, with the full-size sim truck settings
(config/a2b_params_simone.yaml). Simmoulde provides localization, plan_target (B) and task_state."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("a2b_autodrive"), "config", "a2b_params_simone.yaml")
    return LaunchDescription([
        Node(package="a2b_autodrive", executable="a2b_planner", name="a2b_planner",
             output="screen", parameters=[cfg]),
    ])
