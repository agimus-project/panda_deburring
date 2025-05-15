import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml
from agimus_controller.factory.robot_model import RobotModelParameters, RobotModels
from agimus_controller.mpc import MPC
from agimus_controller.mpc_data import OCPResults
from agimus_controller.ocp_param_base import DTFactorsNSeq
from agimus_controller.trajectory import (
    TrajectoryBuffer,
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)
from agimus_controller.warm_start_reference import WarmStartReference
from agimus_msgs.msg import MpcInput
from agimus_pytroller_py.agimus_pytroller_base import (
    ControllerImpl as AgimusControllerImplBase,
)
from ament_index_python.packages import get_package_share_directory

from panda_deburring.ocp_croco_force_feedback import (
    OCPCrocoForceFeedback,
    OCPParamsCrocoForceFeedback,
)
from panda_deburring.warm_start_shift_previous_solution_force_feedback import (
    WarmStartShiftPreviousSolutionForceFeedback,
)


class ControllerImpl(AgimusControllerImplBase):
    def __init__(self, robot_description: str) -> None:
        self._topic_map = {}
        package_share_directory = Path(get_package_share_directory("panda_deburring"))
        config_file = package_share_directory / "config" / "pytroller_params.yaml"
        cfg = yaml.safe_load(config_file.read_text())["agimus_pytroller"][
            "ros__parameters"
        ]["pytroller_python_params"]
        # Convert all lists of numbers to numpy arrays
        for sub_cfg_key, sub_cfg in cfg.items():
            if sub_cfg_key == "dt_factor_n_seq":
                continue
            for key in sub_cfg.keys():
                if isinstance(sub_cfg[key], list) and not isinstance(
                    sub_cfg[key][0], str
                ):
                    sub_cfg[key] = np.asarray(sub_cfg[key], dtype=np.float64)

        robot_params = RobotModelParameters(
            robot_urdf=robot_description, **cfg["robot_model_params"]
        )

        self._robot_models = RobotModels(robot_params)
        self._robot_data = self._robot_models.robot_model.createData()

        dt_factor_n_seq = DTFactorsNSeq(**cfg["dt_factor_n_seq"])
        self._ocp_params = OCPParamsCrocoForceFeedback(
            dt_factor_n_seq=dt_factor_n_seq, **cfg["ocp_params_force_feedback"]
        )

        ocp = OCPCrocoForceFeedback(self._robot_models, self._ocp_params)

        traj_buffer = TrajectoryBuffer(dt_factor_n_seq)

        ws_shift = WarmStartShiftPreviousSolutionForceFeedback()
        ws_shift.setup(self._robot_models, self._ocp_params)

        # Use WarmStartReference for initialization
        ws_ref = WarmStartReference()
        ws_ref.setup(self._robot_models._robot_model)

        self.mpc = MPC()
        self.mpc.setup(ocp=ocp, warm_start=ws_shift, buffer=traj_buffer)

        self._external_forces = [
            pin.Force() for _ in range(self._robot_models.robot_model.nq + 1)
        ]
        self._ddq = np.zeros(self._robot_models.robot_model.nq)
        self._u_zeros = np.zeros(self._robot_models.robot_model.nv)
        self._frame_of_interest_id = self._robot_models.robot_model.getFrameId(
            self._ocp_params.frame_of_interest
        )

        self.__create_traj_weight_point(cfg["trajectory_params"])
        for _ in range(self._ocp_params.horizon_size + 5):
            self.mpc.append_trajectory_point(self._the_point)

        self._first_call = True

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
        self.mpc.append_trajectory_point(
            WeightedTrajectoryPoint(point=traj_point, weights=traj_weights)
        )

    def __create_traj_weight_point(self, params) -> None:
        traj_point = TrajectoryPoint(
            time_ns=time.time_ns(),
            robot_configuration=params["robot_configuration"],
            robot_velocity=params["robot_velocity"],
            robot_acceleration=np.zeros(7),
            robot_effort=np.zeros(7),
            forces={
                params["frame_of_interest"]: pin.Force(
                    params["desired_force"], np.zeros(3)
                )
            },
            end_effector_poses={
                params["frame_of_interest"]: pin.XYZQUATToSE3(
                    np.concatenate(
                        (
                            params["frame_target_pose_position"],
                            params["frame_target_pose_quat"],
                        )
                    )
                )
            },
        )

        traj_weights = TrajectoryPointWeights(
            w_robot_configuration=params["w_robot_configuration"],
            w_robot_velocity=params["w_robot_velocity"],
            w_robot_acceleration=np.zeros(7),
            w_robot_effort=params["w_robot_effort"],
            w_forces={params["frame_of_interest"]: params["desired_force_weights"]},
            w_end_effector_poses={
                params["frame_of_interest"]: np.concatenate(
                    (
                        params["frame_translation_weights"],
                        params["frame_rotation_weights"],
                    )
                )
            },
        )

        self._the_point = WeightedTrajectoryPoint(
            point=traj_point, weights=traj_weights
        )

    def on_update(self, state: np.array) -> np.array:
        # state[-6:] = -state[-6:]
        # print(state[-4], flush=True)
        now = time.time()
        nq = self._robot_models.robot_model.nq
        nv = self._robot_models.robot_model.nv

        # Extract state values
        q = state[:nq]
        dq = state[nq : nq + nv]
        force = state[-6:]

        # Compute gravity torque
        pin.framesForwardKinematics(self._robot_models.robot_model, self._robot_data, q)

        # On first call, initialize warmstart and return zero control
        if self._first_call:
            pin.computeJointJacobians(
                self._robot_models.robot_model, self._robot_data, q
            )
            oMc = self._robot_data.oMf[self._frame_of_interest_id]
            oMc.translation += self._ocp_params.oPc_offset
            force_world = oMc.actionInverse.T.dot(force)
            for i in range(nq):
                self._external_forces[i].vector = (
                    self._robot_data.oMi[i].inverse().actionInverse.T.dot(force_world)
                )
            tau_g = pin.rnea(
                self._robot_models.robot_model,
                self._robot_data,
                q,
                dq,
                self._ddq,
                self._external_forces,
            )

            T = self._ocp_params.horizon_size
            dummy_warmstart = OCPResults(
                states=[state[:-3]] * (T + 1), feed_forward_terms=[tau_g] * T
            )
            self.mpc._warm_start.update_previous_solution(dummy_warmstart)
            self._first_call = False
            # return preallocated array of zeros
            return self._u_zeros
            # return state

        x0_traj_point = TrajectoryPoint(
            time_ns=now,
            robot_configuration=q,
            robot_velocity=dq,
            robot_acceleration=self._ddq,
            forces=pin.Force(force),
        )

        self.mpc.append_trajectory_point(self._the_point)
        ocp_res = self.mpc.run(initial_state=x0_traj_point, current_time_ns=now)

        if ocp_res is None:
            return self._u_zeros
            # return state
        # return self._u_zeros
        # print(time.time() - now)
        # return ocp_res.states[1]

        tau_g = pin.rnea(self._robot_models.robot_model, self._robot_data, q, np.zeros(7), np.zeros(7))

        return ocp_res.feed_forward_terms[0] - tau_g
