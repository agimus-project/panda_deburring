from agimus_demos_common.launch_utils import (
    generate_default_franka_args,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchContext, LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_entity import LaunchDescriptionEntity
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)


def launch_setup(
    context: LaunchContext, *args, **kwargs
) -> list[LaunchDescriptionEntity]:
    pytroller_params = (
        PathJoinSubstitution(
            [
                FindPackageShare("panda_deburring"),
                "config",
                "pytroller_params.yaml",
            ]
        ),
    )

    agimus_pytroller_names = ["agimus_pytroller"]

    franka_robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("agimus_demos_common"),
                        "launch",
                        "franka_common.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments={
            "external_controllers_names": str(agimus_pytroller_names),
            "external_controllers_params": pytroller_params,
            "arm_id": LaunchConfiguration("arm_id"),
            "aux_computer_ip": LaunchConfiguration("aux_computer_ip"),
            "aux_computer_user": LaunchConfiguration("aux_computer_user"),
            "on_aux_computer": LaunchConfiguration("on_aux_computer"),
            "robot_ip": LaunchConfiguration("robot_ip"),
            "use_gazebo": LaunchConfiguration("use_gazebo"),
            "use_rviz": LaunchConfiguration("use_rviz"),
            "gz_verbose": LaunchConfiguration("gz_verbose"),
            "gz_headless": LaunchConfiguration("gz_headless"),
        }.items(),
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
