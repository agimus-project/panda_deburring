import copy
import time

import numpy as np
import pinocchio as pin
import rclpy
from agimus_controller.trajectory import (
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)
from agimus_controller_ros.ros_utils import weighted_traj_point_to_mpc_msg
from agimus_msgs.msg import MpcInput, MpcInputArray
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from panda_deburring.trajectory_publisher_parameters import trajectory_publisher


class TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__("trajectory_publisher")

        self._param_listener = trajectory_publisher.ParamListener(self)
        self._params = self._param_listener.get_params()

        self._mpc_input_publisher = self.create_publisher(
            MpcInputArray,
            "mpc_input",
            qos_profile=QoSProfile(
                depth=1000,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )

        self._base_trajectory_point: WeightedTrajectoryPoint | None = None
        self._update_params(first_call=True)
        self._sequence_cnt = 0

        self._generator_cb = {"sanding_generator": self._circle_generator}[
            self._params.trajectory_generator_name
        ]

        self._timer = self.create_timer(
            1.0 / self._params.update_frequency, self._publish_mpc_input_cb
        )
        self.get_logger().info("Node started.")

    def _update_params(self, first_call: bool = False) -> None:
        """Updates values of dynamic parameters and updated dependent objects.

        Args:
            first_call (bool, optional): If the function is called for the first time from __init__. Defaults to False.
        """
        if self._param_listener.is_old(self._params) or first_call:
            self._param_listener.refresh_dynamic_parameters()
            self._params = self._param_listener.get_params()

            frame_of_interest = self._params.frame_of_interest

            configuration = self._params.initial_targets
            traj_point = TrajectoryPoint(
                time_ns=time.time_ns(),
                robot_configuration=np.asarray(configuration.robot_configuration),
                robot_velocity=np.asarray(configuration.robot_velocity),
                robot_acceleration=np.zeros(len(configuration.robot_velocity)),
                robot_effort=np.zeros(len(configuration.robot_velocity)),
                forces={
                    frame_of_interest: pin.Force(np.array(configuration.desired_force))
                },
                end_effector_poses={
                    frame_of_interest: np.concatenate(
                        (
                            np.asarray(configuration.frame_translation),
                            np.asarray(configuration.frame_rotation),
                        ),
                    )
                },
                end_effector_velocities={frame_of_interest: pin.Motion.Zero()},
            )

            weights = self._params.weights
            traj_weights = TrajectoryPointWeights(
                w_robot_configuration=weights.robot_configuration,
                w_robot_velocity=weights.robot_velocity,
                w_robot_acceleration=[0.0] * len(weights.robot_configuration),
                w_robot_effort=weights.robot_effort,
                w_forces={
                    frame_of_interest: np.asarray(weights.desired_force),
                },
                w_end_effector_poses={
                    frame_of_interest: np.concatenate(
                        (
                            np.asarray(weights.frame_translation),
                            np.asarray(weights.frame_rotation),
                        )
                    )
                },
                w_end_effector_velocities={
                    frame_of_interest: np.concatenate(
                        (
                            np.asarray(weights.frame_linear_velocity),
                            np.asarray(weights.frame_angular_velocity),
                        )
                    )
                },
            )

            self._base_trajectory_point = WeightedTrajectoryPoint(
                point=traj_point, weights=traj_weights
            )

    def _circle_generator(self, seq: int) -> MpcInput:
        """Generates circular motion around specified point.

        Args:
            seq (int): Sequence number.

        Returns:
            agimus_msgs.msg.MpcInput: ROS message with MPC input.
        """
        point = copy.deepcopy(self._base_trajectory_point)
        point.point.id = seq

        r = self._params.sanding_generator.circle.radius
        frequency = self._params.sanding_generator.circle.frequency

        t = seq / self._params.update_frequency

        omega = frequency * 2.0 * np.pi
        x = np.cos(t * omega) * r
        y = np.sin(t * omega) * r
        circle = np.array([x, y, 0.0])

        dx = -r * omega * np.sin(t * omega)
        dy = r * omega * np.cos(t * omega)
        dcircle = np.array([dx, dy, 0.0])

        point.point.end_effector_poses[self._params.frame_of_interest][:3] = (
            np.array(self._params.initial_targets.frame_translation) + circle
        )

        point.point.end_effector_velocities[
            self._params.frame_of_interest
        ].linear = dcircle

        return weighted_traj_point_to_mpc_msg(point)

    def _publish_mpc_input_cb(self):
        """Callback publishing messages with trajectory to follow by MPC."""
        self._update_params()

        end_seq = self._sequence_cnt + self._params.n_buffer_points
        mpc_input_array = MpcInputArray(
            inputs=[
                self._generator_cb(seq) for seq in range(self._sequence_cnt, end_seq)
            ]
        )
        self._sequence_cnt = end_seq

        mpc_input_array.header.stamp = self.get_clock().now().to_msg()
        self._mpc_input_publisher.publish(mpc_input_array)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
