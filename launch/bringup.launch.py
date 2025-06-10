from agimus_demos_common.launch_utils import (
    generate_default_franka_args,
    generate_include_launch,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchContext, LaunchDescription
from launch.actions import OpaqueFunction
from launch.launch_description_entity import LaunchDescriptionEntity
from launch.substitutions import PathJoinSubstitution


def launch_setup(
    context: LaunchContext, *args, **kwargs
) -> list[LaunchDescriptionEntity]:
    rviz_config_path = PathJoinSubstitution(
        [
            FindPackageShare("panda_deburring"),
            "rviz",
            "config.rviz",
        ]
    )

    pytroller_params = (
        PathJoinSubstitution(
            [
                FindPackageShare("panda_deburring"),
                "config",
                "pytroller_params.yaml",
            ]
        ),
    )

    agimus_pytroller_names = ["agimus_pytroller", "ft_calibration_filter"]

    franka_robot_launch = generate_include_launch(
        "franka_common.launch.py",
        extra_launch_arguments={
            "external_controllers_names": str(agimus_pytroller_names),
            "external_controllers_params": pytroller_params,
            "rviz_config_path": rviz_config_path,
            "use_ft_sensor": "true",
        },
    )

    trajectory_publisher_params = (
        PathJoinSubstitution(
            [
                FindPackageShare("panda_deburring"),
                "config",
                "trajectory_publisher_params.yaml",
            ]
        ),
    )

    trajectory_publisher_node = Node(
        package="panda_deburring",
        executable="trajectory_publisher",
        name="trajectory_publisher_node",
        output="screen",
        parameters=[trajectory_publisher_params],
    )

    return [franka_robot_launch, trajectory_publisher_node]


def generate_launch_description():
    return LaunchDescription(
        generate_default_franka_args() + [OpaqueFunction(function=launch_setup)]
    )
