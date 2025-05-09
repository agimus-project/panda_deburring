import time

import numpy as np
import pinocchio as pin
from agimus_controller.factory.robot_model import RobotModelParameters, RobotModels
from agimus_controller.mpc import MPC
from agimus_controller.mpc_data import OCPResults
from agimus_controller.ocp.ocp_croco_generic import OCPCrocoGeneric
from agimus_controller.ocp_param_base import DTFactorsNSeq, OCPParamsBaseCroco
from agimus_controller.trajectories.sine_wave_cartesian_space import (
    SinusWaveCartesianSpace,
)
from agimus_controller.trajectories.sine_wave_params import SinWaveParams
from agimus_controller.trajectory import (
    TrajectoryBuffer,
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)
from agimus_controller.warm_start_reference import WarmStartReference
from agimus_controller.warm_start_shift_previous_solution import (
    WarmStartShiftPreviousSolution,
)
from agimus_msgs.msg import MpcInput
from agimus_pytroller_py.agimus_pytroller_base import (
    ControllerImpl as AgimusControllerImplBase,
)
from ament_index_python.packages import get_package_share_directory


class ControllerImpl(AgimusControllerImplBase):
    def __init__(self, robot_description: str) -> None:
        robot_params = RobotModelParameters(
            robot_urdf=robot_description,
            moving_joint_names=[f"fer_joint{i}" for i in range(1, 8)],
            free_flyer=False,
            collision_as_capsule=False,
            self_collision=False,
            armature=np.full(7, 0.1),
        )
        self.robot_models = RobotModels(robot_params)
        self.robot_data = self.robot_models.robot_model.createData()

        self._topic_map = {}
        self._first_call = True
        self._n_steps = 2

        dt_factor_n_seq = DTFactorsNSeq(factors=[1], n_steps=[self._n_steps])

        package_share_directory = get_package_share_directory("panda_deburring")

        yaml_file = OCPCrocoGeneric.get_default_yaml_file(
            package_share_directory + "/config/ocp_definition_file.yaml"
        )

        self._dt = 1 / 1000 * 2
        self._sin_dt = 2 / 1000
        self._timer = 0.0

        self.ocp_params = OCPParamsBaseCroco(
            dt=self._dt,
            dt_factor_n_seq=dt_factor_n_seq,
            horizon_size=self._n_steps,
            solver_iters=2,
            callbacks=False,
            qp_iters=150,
            use_debug_data=False,
        )

        ocp = OCPCrocoGeneric(self.robot_models, self.ocp_params, yaml_file)

        self.traj_buffer = TrajectoryBuffer(dt_factor_n_seq)

        ws_shift = WarmStartShiftPreviousSolution()
        ws_shift.setup(self.robot_models, self.ocp_params)

        # Use WarmStartReference for initialization
        ws_ref = WarmStartReference()
        ws_ref.setup(self.robot_models._robot_model)

        self.mpc = MPC()
        self.mpc.setup(ocp=ocp, warm_start=ws_shift, buffer=self.traj_buffer)

        self.trajectory = SinusWaveCartesianSpace(
            sine_wave_params=SinWaveParams(
                amplitude=[0.1, 0.1, 0.1],
                period=[4.0, 4.0, 0.0],
                scale_duration=[2.0, 2.0, 2.0],
            ),
            ee_frame_name="fer_hand_tcp",
            w_q=np.full(7, 1.0),
            w_qdot=np.full(7, 1.0),
            w_qddot=np.full(7, 0.000001),
            w_robot_effort=np.full(7, 1.0),
            w_pose=np.full(6, 1.0),
            mask=[True, True, True, False, False, False],
        )

        self._start_time = None

    def mpc_input_cb(self, msg: MpcInput):
        now = time.time_ns()

        xyz_quat_pose = np.array(
            [
                getattr(t, f)
                for t in (msg.pose.position, msg.pose.orientation)
                for f in "xyzw"
                if hasattr(t, f)
            ]
        )
        traj_point = TrajectoryPoint(
            time_ns=now,
            robot_configuration=np.array(msg.q, dtype=np.float64),
            robot_velocity=np.array(msg.qdot, dtype=np.float64),
            robot_acceleration=np.array(msg.qddot, dtype=np.float64),
            robot_effort=np.array(msg.robot_effort, dtype=np.float64),
            end_effector_poses={msg.ee_frame_name: pin.XYZQUATToSE3(xyz_quat_pose)},
        )

        traj_weights = TrajectoryPointWeights(
            w_robot_configuration=np.array(msg.w_q, dtype=np.float64),
            w_robot_velocity=np.array(msg.w_qdot, dtype=np.float64),
            w_robot_acceleration=np.array(msg.w_qddot, dtype=np.float64),
            w_robot_effort=np.array(msg.w_robot_effort, dtype=np.float64),
            w_end_effector_poses={msg.ee_frame_name: msg.w_pose},
        )
        for _ in range(4):
            self.mpc.append_trajectory_point(
                WeightedTrajectoryPoint(point=traj_point, weights=traj_weights)
            )

    def on_update(self, state: np.array) -> np.array:
        now = time.time()

        if self._first_call:
            self._first_call = False
            tau_g = pin.computeGeneralizedGravity(
                self.robot_models.robot_model, self.robot_data, state[:7]
            )
            dummy_warmstart = OCPResults(
                states=[state] * (self.mpc._ocp.n_controls + 1),
                feed_forward_terms=[tau_g] * self.mpc._ocp.n_controls,
            )

            self.mpc._warm_start.update_previous_solution(dummy_warmstart)

            self.trajectory.initialize(self.robot_models.robot_model, state[:7])
            self._start_time = now
            for _ in range(self._n_steps + 1):
                self.mpc.append_trajectory_point(
                    self.trajectory.get_traj_point_at_t(0.0)
                )
            return np.zeros(7)

        self.mpc.append_trajectory_point(
            self.trajectory.get_traj_point_at_t(self._timer)
        )
        self._timer += self._sin_dt

        x0_traj_point = TrajectoryPoint(
            time_ns=now,
            robot_configuration=state[:7],
            robot_velocity=state[7:],
            robot_acceleration=np.zeros(7),
        )

        ocp_res = self.mpc.run(
            initial_state=x0_traj_point,
            current_time_ns=now,
        )

        if ocp_res is None:
            res = np.zeros(7)
        else:
            res = ocp_res.feed_forward_terms[0] - pin.computeGeneralizedGravity(
                self.robot_models.robot_model, self.robot_data, ocp_res.states[0][:7]
            )

        # print(time.time() - now, flush=True)

        return res
