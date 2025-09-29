import copy
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pinocchio as pin
import yaml
from agimus_controller.factory.robot_model import RobotModelParameters, RobotModels
from agimus_controller.mpc_data import OCPResults
from agimus_controller.ocp.ocp_croco_generic import (
    OCPCrocoGeneric,
    OCPParamsBaseCroco,
    add_modules,
)
from agimus_controller.ocp_param_base import DTFactorsNSeq
from agimus_controller.trajectory import (
    TrajectoryBuffer,
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)
from agimus_controller.warm_start_reference import WarmStartReference
from agimus_controller_ros.ros_utils import (
    mpc_debug_data_to_msg,
    mpc_msg_to_weighted_traj_point,
)
from agimus_msgs.msg import MpcDebug, MpcInputArray
from agimus_pytroller_py.agimus_pytroller_base import ControllerImplBase
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Int64

from panda_deburring.force_feedback_ocp_croco_generic import (
    OCPCrocoContactGeneric,
    get_globals,
)
from panda_deburring.mpc import DeburringMPC
from panda_deburring.warm_start_shift_previous_solution_force_feedback import (
    WarmStartShiftPreviousSolutionContact,
)

try:
    import crocoddyl_plotter
except:
    pass


class ControllerImpl(ControllerImplBase):
    def __init__(self, robot_description: str, environment_description: str) -> None:
        self._topic_map = {}
        package_share_directory = Path(get_package_share_directory("panda_deburring"))
        config_file = package_share_directory / "config" / "pytroller_params.yaml"
        cfg = yaml.safe_load(config_file.read_text())["agimus_pytroller"][
            "ros__parameters"
        ]["pytroller_python_params"]
        # Convert all lists of numbers to numpy arrays
        for sub_cfg_key, sub_cfg in cfg.items():
            if sub_cfg_key == "dt_factor_n_seq" or sub_cfg_key == "ocp_definition_file":
                continue
            if isinstance(sub_cfg, list):
                sub_cfg = np.asarray(sub_cfg, dtype=np.float64)
                continue
            for key in sub_cfg.keys():
                if isinstance(sub_cfg[key], list) and not isinstance(
                    sub_cfg[key][0], str
                ):
                    sub_cfg[key] = np.asarray(sub_cfg[key], dtype=np.float64)

        robot_params = RobotModelParameters(
            robot_urdf=robot_description,
            env_urdf=environment_description,
            **cfg["robot_model_params"],
        )

        self._robot_models = RobotModels(robot_params)
        self._robot_data = self._robot_models.robot_model.createData()

        dt_factor_n_seq = DTFactorsNSeq(**cfg["dt_factor_n_seq"])
        self._ocp_params = OCPParamsBaseCroco(
            dt_factor_n_seq=dt_factor_n_seq, **cfg["ocp_params"]
        )
        add_modules(get_globals())
        yaml_file = cfg["ocp_definition_file"]
        ocp = OCPCrocoContactGeneric(self._robot_models, self._ocp_params, yaml_file)

        traj_buffer = TrajectoryBuffer(dt_factor_n_seq)

        ws_shift = WarmStartShiftPreviousSolutionContact()
        ws_shift.setup(self._robot_models, self._ocp_params, yaml_file)

        # Use WarmStartReference for initialization
        ws_ref = WarmStartReference()
        ws_ref.setup(self._robot_models._robot_model)

        self.mpc = DeburringMPC()
        self.mpc.setup(ocp=ocp, warm_start=ws_shift, buffer=traj_buffer)

        self._external_forces = [
            pin.Force(np.zeros(6)) for _ in range(self._robot_models.robot_model.nq + 1)
        ]
        self._nv_zeros = np.zeros(self._robot_models.robot_model.nv)
        self._u_zeros = np.zeros(self._robot_models.robot_model.nv)

        self._in_pd_mode = True
        self._p_gains = cfg["p_gains"]
        self._d_gains = cfg["d_gains"]
        self._q_init = None

        self._publish_debug_data = cfg["ocp_params"]["use_debug_data"]

        self._first_call = True
        self._ocp_res_is_none = True

        plotter_cfg = cfg["cost_plotter"]
        if plotter_cfg["use_cost_plotter"]:
            self._use_cost_plotter = True
            self._iteration_counter = 0
            self._cost_plotter_server = crocoddyl_plotter.CrocoddylPlotterServer(
                plotter_cfg["cost_plotter_url"]
            )

    def mpc_input_cb(self, msg: MpcInputArray):
        for mpc_input in msg.inputs:
            self.mpc.append_trajectory_point(
                mpc_msg_to_weighted_traj_point(mpc_input, time.time_ns())
            )

    def get_ocp_results(self) -> MpcDebug:
        if self._ocp_res_is_none or not self._publish_debug_data:
            return MpcDebug()

        return mpc_debug_data_to_msg(self.mpc.mpc_debug_data)

    def get_buffer_size(self) -> Int64:
        return Int64(data=len(self.mpc._buffer))

    def on_update(self, state: npt.ArrayLike) -> npt.ArrayLike:
        # return self._u_zeros
        now = time.time()
        nq = self._robot_models.robot_model.nq
        nv = self._robot_models.robot_model.nv

        assert state.size == nq + nv + 6 + 1
        # Extract state values
        q = state[:nq]
        dq = state[nq : nq + nv]
        force = state[-7:-1]
        # Currently unused, kept for compatibility
        _ = state[-1] > 0.5

        # Compute gravity torque
        rmodel = self._robot_models.robot_model
        pin.framesForwardKinematics(rmodel, self._robot_data, q)

        # On first call, initialize warmstart and return zero control
        if self._first_call:
            self._q_init = q.copy()
            pin.computeJointJacobians(rmodel, self._robot_data, q)
            frame_of_interest_id = rmodel.getFrameId(self.mpc._ocp.frame_id)
            oMc = self._robot_data.oMf[frame_of_interest_id]
            oMc.translation += self.mpc._ocp.oPc
            force_world = oMc.actionInverse.T.dot(force)
            for i in range(nq + 1):
                self._external_forces[i].vector = (
                    self._robot_data.oMi[i].inverse().actionInverse.T.dot(force_world)
                )
            tau_g = pin.rnea(
                rmodel,
                self._robot_data,
                q,
                dq,
                self._nv_zeros,
                self._external_forces,
            )

            T = self._ocp_params.horizon_size
            directions = self.mpc._ocp.enabled_directions
            if sum(directions) == 3:
                states = state[:-4]
            else:
                states = np.concatenate((q, dq, force[:3][directions]))
            dummy_warmstart = OCPResults(
                states=[states] * (T + 1), feed_forward_terms=[tau_g] * T
            )
            self.mpc._warm_start.update_previous_solution(dummy_warmstart)
            self._first_call = False
            # return preallocated array of zeros
            return self._u_zeros

        x0_traj_point = TrajectoryPoint(
            time_ns=now,
            robot_configuration=q,
            robot_velocity=dq,
            robot_acceleration=self._nv_zeros,
            forces=pin.Force(force),
        )

        ocp_res = self.mpc.run(initial_state=x0_traj_point, current_time_ns=now)

        if ocp_res is None:
            self._ocp_res_is_none = True
            if self._in_pd_mode:
                # Compute safety PD control
                return (self._q_init - q) * self._p_gains - dq * self._d_gains
            return self._u_zeros
        self._ocp_res_is_none = False
        self._in_pd_mode = False

        tau_g = pin.rnea(
            rmodel,
            self._robot_data,
            q,
            self._nv_zeros,
            self._nv_zeros,
        )

        return ocp_res.feed_forward_terms[0] - tau_g

    def on_post_update(self) -> None:
        if self._use_cost_plotter and self._iteration_counter % 100:
            self._cost_plotter_server.send(
                self.mpc._ocp._solver.problem, self._iteration_counter
            )
        self._iteration_counter += 1
        self.mpc.update_references()
