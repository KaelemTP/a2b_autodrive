from glob import glob
from setuptools import setup

package_name = "a2b_autodrive"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "a2b_planner = a2b_autodrive.a2b_planner_node:main",
            "truck_sim = a2b_autodrive.truck_sim_node:main",
            "lidar_loc_bridge = a2b_autodrive.lidar_loc_bridge_node:main",
            "fake_lidar = a2b_autodrive.fake_lidar_node:main",
            "sim_viewer = a2b_autodrive.sim_viewer_node:main",
            "ground_filter = a2b_autodrive.ground_filter_node:main",
            "mujoco_sim = a2b_autodrive.mujoco_sim_node:main",
        ],
    },
)
