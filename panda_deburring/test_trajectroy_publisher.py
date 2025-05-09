from typing import List

import numpy as np
import rclpy
from agimus_controller.factory.robot_model import (
    RobotModelParameters,
    RobotModels,
)
from agimus_controller.trajectories.generic_trajectory import GenericTrajectory
from agimus_controller.trajectories.sine_wave_cartesian_space import (
    SinusWaveCartesianSpace,
)
from agimus_controller.trajectories.sine_wave_configuration_space import (
    SinusWaveConfigurationSpace,
)
from agimus_controller.trajectories.sine_wave_params import SinWaveParams
from agimus_controller.trajectories.trajectory_base import TrajectoryBase
from agimus_controller_ros.ros_utils import weighted_traj_point_to_mpc_msg
from agimus_msgs.msg import MpcInput
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.task import Future
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from panda_deburring.test_trajectroy_publisher_parameters import (
    trajectory_weights_params,
)


class SimpleTrajectoryPublisher(Node):
    """This is a simple trajectory publisher for a Panda robot."""

    def __init__(self):
        super().__init__("simple_trajectory_publisher")

        self.param_listener = trajectory_weights_params.ParamListener(self)
        self.params = self.param_listener.get_params()
        self.ee_frame_name = self.params.ee_frame_name

        self.q0 = None
        self.current_q = None
        self.robot_description_msg = None
        self.t = 0.0
        self.dt = 0.01
        self.croco_nq = 7
        self.future_init_done = Future()
        self.future_trajectory_done = Future()

        self.moving_joint_names = self.params.moving_joint_names

        self.subscriber_robot_description_ = self.create_subscription(
            String,
            "/robot_description",
            self.robot_description_callback,
            qos_profile=QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        self.state_subscriber = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_states_callback,
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )
        self.publisher_ = self.create_publisher(
            MpcInput,
            "mpc_input",
            qos_profile=QoSProfile(
                depth=1000,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )
        self.trajectory = self.get_trajectory(self.params.trajectory_name)
        self.timer = self.create_timer(
            self.dt, self.publish_mpc_input
        )  # Publish at 100 Hz
        self.get_logger().info("Simple trajectory publisher node started.")

    def get_param_from_node(self, node_name: str, param_name: str) -> ParameterValue:
        """Returns parameter from the node"""
        param_client = self.create_client(GetParameters, f"/{node_name}/get_parameters")
        while not param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service not available, waiting again...")
        request = GetParameters.Request()
        request.names = [param_name]

        future = param_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            return future.result().values[0]
        else:
            raise ValueError("Failed to load moving joint names from LFC")

    def add_trajectory(self, trajectory):
        if self.params.trajectory_name == "generic_trajectory":
            self.trajectory.add_trajectory(trajectory)
            self.future_trajectory_done = Future()
        else:
            raise RuntimeError(
                f"the function add_trajectory can't be used with trajectory type {self.params.trajectory_name}"
            )

    def joint_states_callback(self, msg: JointState) -> None:
        """Set joint state reference."""
        joint_map = [
            msg.name.index(joint_name) for joint_name in self.params.moving_joint_names
        ]

        self.current_q = np.array(msg.position)[joint_map]
        self.current_dq = np.array(msg.velocity)[joint_map]

        if self.q0 is None and np.linalg.norm(self.current_q) > 1e-2:
            self.q0 = self.current_q
            self.get_logger().warn(f"Received q0 = {[round(el, 2) for el in self.q0]}.")

    def robot_description_callback(self, msg: String) -> None:
        """Create the models of the robot from the urdf string."""
        self.get_logger().warn("Received robot description.")
        self.robot_description_msg = msg
        self.destroy_subscription(self.subscriber_robot_description_)

    def get_sine_wave_parameters(self) -> SinWaveParams:
        """Get sine wave parameters."""
        sine_wave_amplitude = self.params.sine_wave.amplitude
        if len(sine_wave_amplitude) == 1:
            self.params.sine_wave.amplitude = (sine_wave_amplitude[0],) * len(
                self.moving_joint_names
            )
        sine_wave_period = self.params.sine_wave.period
        if len(sine_wave_period) == 1:
            self.params.sine_wave.period = (sine_wave_period[0],) * len(
                self.moving_joint_names
            )
        sine_wave_scale_duration = self.params.sine_wave.scale_duration
        if len(sine_wave_scale_duration) == 1:
            self.params.sine_wave.scale_duration = (sine_wave_scale_duration[0],) * len(
                self.moving_joint_names
            )
        self.sine_wave_parameters = SinWaveParams(
            amplitude=sine_wave_amplitude,
            period=sine_wave_period,
            scale_duration=sine_wave_scale_duration,
        )
        return self.sine_wave_parameters

    def get_trajectory(self, trajectory_name: String) -> TrajectoryBase:
        """Build chosen trajectory."""
        self.sine_wave_parameters = self.get_sine_wave_parameters()
        if trajectory_name == "sine_wave_configuration_space":
            assert len(self.sine_wave_parameters.amplitude) == len(
                self.moving_joint_names
            ), "sine_wave_amplitude and moving_joint_names must have the same length"
            assert len(self.sine_wave_parameters.period) == len(
                self.moving_joint_names
            ), "sine_wave_period and moving_joint_names must have the same length"
            assert len(self.sine_wave_parameters.scale_duration) == len(
                self.moving_joint_names
            ), (
                "sine_wave_scale_duration and moving_joint_names must have the same length"
            )
            return SinusWaveConfigurationSpace(
                sine_wave_params=self.sine_wave_parameters,
                ee_frame_name=self.ee_frame_name,
                w_q=self.get_weights(self.params.w_q, self.croco_nq),
                w_qdot=self.get_weights(self.params.w_qdot, self.croco_nq),
                w_qddot=self.get_weights(self.params.w_qddot, self.croco_nq),
                w_robot_effort=self.get_weights(
                    self.params.w_robot_effort, self.croco_nq
                ),
                w_pose=self.get_weights(self.params.w_pose, 6),
            )
        elif trajectory_name == "sine_wave_cartesian_space":
            assert len(self.sine_wave_parameters.amplitude) == 3, (
                "sine_wave_amplitude length must be 3"
            )
            assert len(self.sine_wave_parameters.period) == 3, (
                "sine_wave_period length must be 3"
            )
            assert len(self.sine_wave_parameters.scale_duration) == 3, (
                "sine_wave_scale_duration length must be 3"
            )
            return SinusWaveCartesianSpace(
                sine_wave_params=self.sine_wave_parameters,
                ee_frame_name=self.ee_frame_name,
                w_q=self.get_weights(self.params.w_q, self.croco_nq),
                w_qdot=self.get_weights(self.params.w_qdot, self.croco_nq),
                w_qddot=self.get_weights(self.params.w_qddot, self.croco_nq),
                w_robot_effort=self.get_weights(
                    self.params.w_robot_effort, self.croco_nq
                ),
                w_pose=self.get_weights(self.params.w_pose, 6),
                mask=self.params.mask,
            )
        elif trajectory_name == "generic_trajectory":
            return GenericTrajectory(
                ee_frame_name=self.ee_frame_name,
                w_q=self.get_weights(self.params.w_q, self.croco_nq),
                w_qdot=self.get_weights(self.params.w_qdot, self.croco_nq),
                w_qddot=self.get_weights(self.params.w_qddot, self.croco_nq),
                w_robot_effort=self.get_weights(
                    self.params.w_robot_effort, self.croco_nq
                ),
                w_pose=self.get_weights(self.params.w_pose, 6),
            )
        else:
            raise ValueError("Unknown Trajectory.")

    def load_models(self):
        """Callback to get robot description and store to object"""
        self.robot_models = RobotModels(
            param=RobotModelParameters(
                robot_urdf=self.robot_description_msg.data,
                free_flyer=False,
                moving_joint_names=self.moving_joint_names,
            )
        )
        self.get_logger().warn(
            f"Model loaded, pin_model.nq = {self.robot_models.robot_model.nq}"
        )
        self.get_logger().warn(f"Model loaded, reduced self.q0 = {self.q0}")

    def get_weights(
        self, weights: List[np.float64], size: np.float64
    ) -> List[np.float64]:
        """
        Return weights with right size if user sent only one value, otherwise
        directly returns weights.
        """
        if len(weights) == 1:
            return weights * size
        else:
            return weights

    def publish_mpc_input(self):
        """
        Main function to create a dummy mpc input
        Modifies each joint in sin manner with 0.2 rad amplitude
        """

        if self.robot_description_msg is None or self.q0 is None:
            return

        if not self.trajectory.is_initialized:
            self.load_models()
            self.trajectory.initialize(self.robot_models.robot_model, self.q0)
            self.future_init_done.set_result(True)
            return
        if (
            self.params.trajectory_name == "generic_trajectory"
            and self.trajectory.trajectory is None
        ):
            self.get_logger().warn(
                "Waiting for trajectory to be initialized.",
                throttle_duration_sec=5.0,
            )
            return
        w_traj_point = self.trajectory.get_traj_point_at_t(self.t)
        msg = weighted_traj_point_to_mpc_msg(w_traj_point)

        self.publisher_.publish(msg)
        if self.trajectory.trajectory_is_done:
            self.future_trajectory_done.set_result(True)
        self.t += self.dt


def main(args=None):
    rclpy.init(args=args)
    node = SimpleTrajectoryPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
