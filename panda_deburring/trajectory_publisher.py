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
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Int64
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

        self._in_contact_subscriber = self.create_subscription(
            Bool,
            "/ft_calibration_filter/contact",
            self._in_contact_cb,
            qos_profile=QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )

        # Transform buffers
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, self)

        self._base_trajectory_point: WeightedTrajectoryPoint | None = None
        self._weights_name = "initialize"
        self._update_params(first_call=True)
        self._sequence_cnt = 0
        self._buffer_size = None
        self._in_contact = None
        self._last_in_contact_state = False

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

    def _in_contact_cb(self, msg: Bool) -> None:
        self._in_contact = msg.data

    def _update_params(self, first_call: bool = False) -> None:
        """Updates values of dynamic parameters and updated dependent objects.

        Args:
            first_call (bool, optional): If the function is called for the first time from __init__. Defaults to False.
        """
        if self._param_listener.is_old(self._params) or first_call:
            self._param_listener.refresh_dynamic_parameters()
            self._params = self._param_listener.get_params()
            self._update_weighted_trajectory_point(self._weights_name)

    def _update_weighted_trajectory_point(self, stame_name: str) -> None:
        self.get_logger().error(f"Setting weights to '{stame_name}'.")
        stage = self._params.get_entry(stame_name)

        frame_of_interest = self._params.frame_of_interest
        traj_point = TrajectoryPoint(
            time_ns=time.time_ns(),
            robot_configuration=np.asarray(stage.robot_configuration),
            robot_velocity=np.asarray(stage.robot_velocity),
            robot_acceleration=np.zeros(len(stage.robot_velocity)),
            robot_effort=np.zeros(len(stage.robot_velocity)),
            forces={
                frame_of_interest: pin.Force(np.array(stage.desired_force), np.zeros(3))
            },
            end_effector_poses={
                frame_of_interest: np.concatenate(
                    (
                        np.asarray(stage.frame_translation),
                        np.asarray(stage.frame_rotation),
                    ),
                )
            },
            end_effector_velocities={frame_of_interest: pin.Motion.Zero()},
        )

        traj_weights = TrajectoryPointWeights(
            w_robot_configuration=stage.w_robot_configuration,
            w_robot_velocity=stage.w_robot_velocity,
            w_robot_acceleration=[0.0] * len(stage.w_robot_configuration),
            w_robot_effort=stage.w_robot_effort,
            w_forces={
                frame_of_interest: np.concatenate(
                    (
                        np.asarray(stage.w_desired_force),
                        np.zeros(3),
                    )
                )
            },
            w_end_effector_poses={
                frame_of_interest: np.concatenate(
                    (
                        np.asarray(stage.w_frame_translation),
                        np.asarray(stage.w_frame_rotation),
                    )
                )
            },
            w_end_effector_velocities={
                frame_of_interest: np.concatenate(
                    (
                        np.asarray(stage.w_frame_linear_velocity),
                        np.asarray(stage.w_frame_angular_velocity),
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
            np.array(self._params.get_entry(self._weights_name).frame_translation)
            + circle
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
            self.get_logger().warn(f"Could not transform {parent} to {child}: {ex}")
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
            if self._buffer_size is None:
                self.get_logger().info(
                    f"Buffer size information to be published...",
                    throttle_duration_sec=5.0,
                )
                return
            if self._in_contact is None:
                self.get_logger().info(
                    f"Buffer size information to be published...",
                    throttle_duration_sec=5.0,
                )
                return
            if not self._check_can_tarnsform():
                self.get_logger().info(
                    f"Waiting for transformation between '{self._params.robot_base_frame}' "
                    f"and '{self._params.frame_of_interest}' to be available...",
                    throttle_duration_sec=5.0,
                )
                return
            self._motion_phase = MotionPhases.initialize
            self._start_point = self._generator_cb(0)
            self._current = self._get_current_pose()
            self._sequence_cnt += 1
            return

        elif self._motion_phase == MotionPhases.initialize:
            max_lin_vel = self._params.initialization.max_linear_velocity
            max_ang_vel = self._params.initialization.max_angular_velocity
            pose_tolerance = self._params.initialization.pose_tolerance
            rot_tolerance = self._params.initialization.rot_tolerance

            end = pin.XYZQUATToSE3(
                self._start_point.point.end_effector_poses[
                    self._params.frame_of_interest
                ]
            )
            inputs = []
            for _ in range(self._params.n_buffer_points - self._buffer_size):
                vel = pin.log6(self._current.inverse() * end)
                lin_vel = np.linalg.norm(vel.linear)
                if lin_vel < pose_tolerance and np.all(vel.angular < rot_tolerance):
                    self.get_logger().info(
                        "Initial position reached. Following trajectory."
                    )
                    self._motion_phase = MotionPhases.perform_motion
                    self._weights_name = "seek_contact"
                    self._update_weighted_trajectory_point(self._weights_name)
                    break

                if lin_vel > max_lin_vel:
                    vel.linear = vel.linear / lin_vel * max_lin_vel

                if np.any(vel.angular > max_ang_vel):
                    vel.angular = vel.linear / np.max(vel.angular) * max_ang_vel
                self._current = self._current * pin.exp6(vel * self._params.ocp_dt)
                point = copy.deepcopy(self._start_point)
                point.point.end_effector_poses[self._params.frame_of_interest] = (
                    pin.SE3ToXYZQUAT(self._current)
                )
                # print(pin.SE3ToXYZQUAT(self._current), flush=True)
                point.point.end_effector_velocities[self._params.frame_of_interest] = (
                    vel
                )
                inputs.append(weighted_traj_point_to_mpc_msg(point))
            mpc_input_array = MpcInputArray(inputs=inputs)

        elif self._motion_phase == MotionPhases.perform_motion:
            self.get_logger().info(
                "Initial configuration received. Generating motion...", once=True
            )

            if self._in_contact != self._last_in_contact_state:
                self._last_in_contact_state = self._in_contact
                self._weights_name = (
                    "in_contact" if self._in_contact else "seek_contact"
                )
                self._update_weighted_trajectory_point(self._weights_name)

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
