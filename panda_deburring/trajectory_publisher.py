import copy
import time
from enum import Enum

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
from std_msgs.msg import Int64
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from panda_deburring.trajectory_publisher_parameters import trajectory_publisher


class MotionPhases(Enum):
    wait_for_data = 0
    initialize = 1
    perform_motion = 2


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

        self._buffer_size_subscriber = self.create_subscription(
            Int64,
            "buffer_size",
            self._buffer_size_cb,
            10,
        )

        # Transform buffers
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, self)

        self._base_trajectory_point: WeightedTrajectoryPoint | None = None
        self._update_params(first_call=True)
        self._sequence_cnt = 0
        self._buffer_size = self._params.n_buffer_points

        self._initial_configuration = pin.SE3()
        self._trajectory_offset = 0
        self._start_point = None
        self._motion_phase = MotionPhases.wait_for_data

        self._generator_cb = {"sanding_generator": self._circle_generator}[
            self._params.trajectory_generator_name
        ]

        self._timer = self.create_timer(
            1.0 / self._params.update_frequency, self._publish_mpc_input_cb
        )
        self.get_logger().info("Node started.")

    def _buffer_size_cb(self, msg: Int64) -> None:
        self._buffer_size = msg.data

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
                    frame_of_interest: pin.Force(
                        np.array(configuration.desired_force), np.zeros(3)
                    )
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
                    frame_of_interest: np.concatenate(
                        (
                            np.asarray(weights.desired_force),
                            np.zeros(3),
                        )
                    )
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

    def _circle_generator(self, seq: int) -> WeightedTrajectoryPoint:
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

        t = seq * self._params.ocp_dt

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

        return point

    def _check_can_tarnsform(self) -> bool:
        parent = self._params.robot_base_frame
        child = self._params.frame_of_interest
        try:
            self._buffer.lookup_transform(parent, child, rclpy.time.Time())
            return True
        except TransformException as ex:
            self.get_logger().info(f"Could not transform {parent} to {child}: {ex}")
            return False

    def _get_current_pose(self) -> pin.SE3:
        parent = self._params.robot_base_frame
        child = self._params.frame_of_interest
        transform = self._buffer.lookup_transform(
            parent, child, rclpy.time.Time()
        ).transform
        return pin.XYZQUATToSE3(
            np.array(
                [
                    transform.translation.x,
                    transform.translation.y,
                    transform.translation.z,
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                    transform.rotation.w,
                ]
            )
        )

    def _publish_mpc_input_cb(self):
        """Callback publishing messages with trajectory to follow by MPC."""
        self._update_params()

        if self._motion_phase == MotionPhases.wait_for_data:
            if not self._check_can_tarnsform():
                self.get_logger().info(
                    f"Waiting for transformation between '{self._params.robot_base_frame}' "
                    f"and '{self._params.frame_of_interest}' to be available...",
                    throttle_duration_sec=5.0,
                )
                return
            self._motion_phase = MotionPhases.initialize
            self._start_point = self._generator_cb(0)
            self._sequence_cnt += 1
            return

        elif self._motion_phase == MotionPhases.initialize:
            self.get_logger().info(
                "Initial configuration received. Generating motion...", once=True
            )
            max_lin_vel = self._params.initialization.max_linear_velocity
            max_ang_vel = self._params.initialization.max_angular_velocity
            pose_tolerance = self._params.initialization.pose_tolerance
            rot_tolerance = self._params.initialization.rot_tolerance

            current = self._get_current_pose()
            end = pin.XYZQUATToSE3(
                self._start_point.point.end_effector_poses[
                    self._params.frame_of_interest
                ]
            )
            inputs = []
            for _ in range(self._params.n_buffer_points - self._buffer_size):
                vel = pin.log6(current.inverse() * end)
                lin_vel = np.linalg.norm(vel.linear)
                if lin_vel < pose_tolerance and np.all(vel.angular < rot_tolerance):
                    self.get_logger().info(
                        "Initial position reached. Following trajectory."
                    )
                    self._motion_phase = MotionPhases.perform_motion
                    break

                if lin_vel > max_lin_vel:
                    vel.linear = vel.linear / lin_vel * max_lin_vel

                ang_vel = np.linalg.norm(vel.angular)
                if ang_vel > max_ang_vel:
                    vel.angular = vel.linear / ang_vel * max_ang_vel

                current = current * pin.exp6(vel)
                point = copy.deepcopy(self._start_point)
                point.point.end_effector_poses[self._params.frame_of_interest] = (
                    pin.SE3ToXYZQUAT(current)
                )
                # point.point.end_effector_velocities[self._params.frame_of_interest] = vel
                inputs.append(weighted_traj_point_to_mpc_msg(point))
            mpc_input_array = MpcInputArray(inputs=inputs)

        elif self._motion_phase == MotionPhases.perform_motion:
            end_seq = self._sequence_cnt + (
                self._params.n_buffer_points - self._buffer_size
            )
            mpc_input_array = MpcInputArray(
                inputs=[
                    weighted_traj_point_to_mpc_msg(self._generator_cb(seq))
                    for seq in range(self._sequence_cnt, end_seq)
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
