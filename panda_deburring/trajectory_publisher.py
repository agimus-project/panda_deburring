import copy
import math
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
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, ColorRGBA, Header, Int64
from std_srvs.srv import Trigger
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker, MarkerArray

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
            "/agimus_pytroller/mpc_input",
            qos_profile=QoSProfile(
                depth=1000,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )

        self._deburring_markers_pub = self.create_publisher(
            MarkerArray, "deburring_markers", 10
        )

        self._buffer_size_subscriber = self.create_subscription(
            Int64,
            "/agimus_pytroller/buffer_size",
            self._buffer_size_cb,
            10,
        )

        self._in_contact_subscriber = self.create_subscription(
            Bool,
            "/ft_calibration_filter/contact",
            self._in_contact_cb,
            10,
        )

        self._calibrate_srv = self.create_client(
            Trigger, "/ft_calibration_filter/calibrate"
        )
        while not self._calibrate_srv.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")

        self._calibration_future = None

        # Transform buffers
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, self)

        self._measurement_offset = pin.SE3(
            pin.rpy.rpyToMatrix(np.deg2rad(np.array([180.0, 0.0, 120.0]))),
            np.array([0.000, 0.000, 0.074]),
        )

        self._base_trajectory_point: WeightedTrajectoryPoint | None = None
        self._weights_name = "initialize"
        self._update_params(True)
        self._sequence_cnt = 0
        self._buffer_size = None
        self._in_contact = None
        self._last_in_contact_state = False

        self._trajectory_offset = 0
        self._first_trajectory_point = None
        self._robot_start_pose = pin.SE3()
        self._rot_vel = np.zeros(3)
        self._motion_phase = MotionPhases.wait_for_data

        self._marker_base = Marker(
            header=Header(frame_id="world"),
            ns="deburring",
            type=Marker.SPHERE,
            action=Marker.ADD,
            scale=Vector3(x=0.01, y=0.01, z=0.01),
            color=ColorRGBA(**dict(zip("rgba", [1.0, 0.0, 0.0, 1.0]))),
            lifetime=Duration(seconds=self._params.rate * 1.25).to_msg(),
        )

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

    def _update_weighted_trajectory_point(self, state_name: str) -> None:
        """Updated trajectory point parameters to use new set of weights.

        Args:
            state_name (str): Name of the state from which weights have to be loaded.
        """
        self.get_logger().info(f"Setting weights to '{state_name}'.")
        stage = self._params.get_entry(state_name)

        tool_frame_id = self._params.tool_frame_id
        measurement_frame_id = self._params.measurement_frame_id

        tool_pose = pin.SE3(
            pin.Quaternion(np.array(stage.frame_rotation)),
            np.array(stage.frame_translation),
        )

        traj_point = TrajectoryPoint(
            time_ns=time.time_ns(),
            robot_configuration=np.asarray(stage.robot_configuration),
            robot_velocity=np.asarray(stage.robot_velocity),
            robot_acceleration=np.zeros(len(stage.robot_velocity)),
            robot_effort=np.zeros(len(stage.robot_velocity)),
            forces={
                measurement_frame_id: pin.Force(
                    np.array(stage.desired_force), np.zeros(3)
                )
            },
            end_effector_poses={
                tool_frame_id: tool_pose,
                measurement_frame_id: tool_pose * self._measurement_offset,
            },
            end_effector_velocities={tool_frame_id: pin.Motion.Zero()},
        )

        self.get_logger().warn(f"{stage.w_desired_force}")

        traj_weights = TrajectoryPointWeights(
            w_robot_configuration=stage.w_robot_configuration,
            w_robot_velocity=stage.w_robot_velocity,
            w_robot_acceleration=[0.0] * len(stage.w_robot_configuration),
            w_robot_effort=stage.w_robot_effort,
            w_forces={
                measurement_frame_id: np.concatenate(
                    (
                        np.asarray(stage.w_desired_force),
                        np.zeros(3),
                    )
                )
            },
            w_end_effector_poses={
                tool_frame_id: np.concatenate(
                    (
                        np.asarray(stage.w_frame_translation),
                        np.asarray(stage.w_frame_rotation),
                    )
                ),
                measurement_frame_id: np.concatenate(
                    (
                        np.asarray(stage.w_frame_translation),
                        np.asarray(stage.w_frame_rotation),
                    )
                ),
            },
            w_end_effector_velocities={
                tool_frame_id: np.concatenate(
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

        # Pose increment
        omega = frequency * 2.0 * np.pi
        x = np.cos(t * omega) * r
        y = np.sin(t * omega) * r
        circle = np.array([x, y, 0.0])

        # Velocity
        dx = -r * omega * np.sin(t * omega)
        dy = r * omega * np.cos(t * omega)
        dcircle = np.array([dx, dy, 0.0])

        point.point.end_effector_poses[self._params.tool_frame_id].translation = (
            np.array(self._params.get_entry(self._weights_name).frame_translation)
            + circle
        )

        point.point.end_effector_velocities[self._params.tool_frame_id].linear = dcircle

        return point

    def _check_can_tarnsform(self) -> bool:
        """Checks if transformation between robot base and end effector can be obtained.

        Returns:
            bool: Indicates if transformation can be obtained.
        """
        parent = self._params.robot_base_frame
        child = self._params.tool_frame_id
        # For some reason standard check transform doesn't work in this node so
        # a walkaround had to be used...
        try:
            self._buffer.lookup_transform(parent, child, rclpy.time.Time())
            return True
        except TransformException as ex:
            self.get_logger().warn(f"Could not transform {parent} to {child}: {ex}")
            return False

    def _get_current_pose(self) -> pin.SE3:
        """Requests pose od the end effector from TF2 buffer and converts it to Pinocchio SE3 object.

        Returns:
            pin.SE3: End effector pose as SE3 object
        """
        parent = self._params.robot_base_frame
        child = self._params.tool_frame_id
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
            if self._in_contact is None:
                self.get_logger().info(
                    f"Waiting for contact info to be published...",
                    throttle_duration_sec=5.0,
                )
                return
            if self._buffer_size is None:
                self.get_logger().info(
                    f"Waiting for buffer size information to be published...",
                    throttle_duration_sec=5.0,
                )
                return
            if not self._check_can_tarnsform():
                self.get_logger().info(
                    f"Waiting for transformation between '{self._params.robot_base_frame}' "
                    f"and '{self._params.tool_frame_id}' to be available...",
                    throttle_duration_sec=5.0,
                )
                return
            # Obtain first point of the desired trajectory
            self._first_trajectory_point = self._generator_cb(0)
            # Get current pose of the robot, as a starting point of the interpolation
            self._robot_start_pose = self._get_current_pose()

            # Target pose of the interpolation is the first point of the trajectory
            self._robot_end_pose = (
                self._first_trajectory_point.point.end_effector_poses[
                    self._params.tool_frame_id
                ]
            )

            # Separately interpolate in SO3 and R3 separately to obtain linear motion
            diff = self._robot_start_pose.inverse() * self._robot_end_pose
            self._rot_vel = pin.log3(diff.rotation)

            max_ang = np.max(pin.rpy.matrixToRpy(diff.rotation))
            dist = np.linalg.norm(diff.translation)

            ang_time = max_ang / self._params.initialization.max_angular_velocity
            lin_time = dist / self._params.initialization.max_linear_velocity

            # Time for the interpolation is slightly larger than needed
            # to account for accelerations and decelerations, plus
            # to presume safety a margin
            interp_time = max(ang_time, lin_time) * 1.25
            self._interp_steps = math.ceil(interp_time / self._params.ocp_dt)

            self._sequence_cnt += 1

            self._motion_phase = MotionPhases.initialize
            return

        elif self._motion_phase == MotionPhases.initialize:
            # TODO implement proper handling of sequence counter in this section
            # Current implementation might not work the best with plotting of
            # the cost gradient in the initialization phase. Tho it might not be
            # that important as this section of the task does not have to be perfect.

            # If calibration service was called and the answer already arrived
            if self._calibration_future is not None and self._calibration_future.done():
                if not self._calibration_future.result().success:
                    self._calibration_future = self._calibrate_srv.call_async(
                        Trigger.Request()
                    )
                    # Retry calling the service by
                    self._calibration_future = None
                    self.get_logger().error("Failed to reset sensor bias!")

                self._motion_phase = MotionPhases.perform_motion
                self._last_in_contact_state = False
                self._weights_name = "seek_contact"
                self._update_weighted_trajectory_point(self._weights_name)
                return

            # If buffer is already empty
            if self._sequence_cnt >= self._interp_steps:
                current = self._get_current_pose()

                diff = current.inverse() * self._robot_end_pose
                lin_dist = np.max(diff.translation)
                ang_dist = np.max(pin.rpy.matrixToRpy(diff.rotation))
                if (
                    self._calibration_future is None
                    and lin_dist < self._params.initialization.pose_tolerance
                    and ang_dist < self._params.initialization.rot_tolerance
                ):
                    self.get_logger().info(
                        "Initial position reached. Calibrating sensor."
                    )
                    self._calibration_future = self._calibrate_srv.call_async(
                        Trigger.Request()
                    )

            inputs = []
            for _ in range(self._params.n_buffer_points - self._buffer_size):
                if self._sequence_cnt < self._interp_steps:
                    self._sequence_cnt += 1
                # self.get_logger().error(f"interpolation steps {self._buffer_size}")

                t = self._sequence_cnt / self._interp_steps

                target = pin.SE3()
                # Interpolate rotation
                target.rotation = self._robot_start_pose.rotation @ pin.exp3(
                    self._rot_vel * t
                )
                # Interpolate translation
                start_t = self._robot_start_pose.translation
                end_t = self._robot_end_pose.translation
                target.translation = (1.0 - t) * start_t + (t) * end_t
                point = copy.deepcopy(self._first_trajectory_point)
                point.point.end_effector_poses[self._params.tool_frame_id] = (
                    pin.SE3ToXYZQUAT(target)
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
