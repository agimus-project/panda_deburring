from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.launch_description_entity import LaunchDescriptionEntity
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)


def launch_setup(
    context: LaunchContext, *args, **kwargs
) -> list[LaunchDescriptionEntity]:
    trajectory_weights_yaml = PathJoinSubstitution(
        [
            FindPackageShare("panda_deburring"),
            "config",
            "trajectory_pub_params.yaml",
        ]
    )

    simple_trajectory_publisher_node = Node(
        package="panda_deburring",
        executable="test_trajectroy_publisher",
        parameters=[trajectory_weights_yaml],
        arguments=[
            "-T",
            "4",
            "-A",
            "0.2",
            "fer_joint3",
            "fer_joint5",
        ],
        output="screen",
    )

    return [simple_trajectory_publisher_node]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=launch_setup)])
