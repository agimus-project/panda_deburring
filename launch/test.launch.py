from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription
from launch.actions import Shutdown
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution


def generate_launch_description():
    franka_controllers_params = PathJoinSubstitution(
        [
            FindPackageShare("panda_deburring"),
            "config",
            "controller_manager_pytroller_params.yaml",
        ]
    )

    xacro_args = {
        "robot_ip": "",
        "arm_id": "fer",
        "ros2_control": "true",
        "hand": "true",
        "use_fake_hardware": "true",
        "fake_sensor_commands": "true",
        "gazebo": "false",
        "ee_id": "franka_hand",
        "gazebo_effort": "true",
        "with_sc": "false",
        "franka_controllers_params": franka_controllers_params,
    }
    robot_description_file_substitution = PathJoinSubstitution(
        [
            FindPackageShare("franka_description"),
            "robots",
            "fer",
            "fer.urdf.xacro",
        ]
    )
    robot_description = ParameterValue(
        Command(
            [
                PathJoinSubstitution([FindExecutable(name="xacro")]),
                " ",
                robot_description_file_substitution,
                # Convert dict to list of parameters
                *[arg for key, val in xacro_args.items() for arg in (f" {key}:=", val)],
            ]
        ),
        value_type=str,
    )

    return LaunchDescription(
        [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="controller_manager",
                executable="ros2_control_node",
                parameters=[
                    franka_controllers_params,
                    {"arm_id": "fer", "load_gripper": "false"},
                ],
                remappings=[
                    ("/controller_manager/robot_description", "/robot_description"),
                ],
                output={
                    "stdout": "screen",
                    "stderr": "screen",
                },
                on_exit=Shutdown(),
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["joint_state_broadcaster"],
                output="screen",
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["agimus_pytroller"],
                output="screen",
            ),
        ]
    )
