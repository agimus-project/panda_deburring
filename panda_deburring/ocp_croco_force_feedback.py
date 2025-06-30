from dataclasses import dataclass

import crocoddyl
import force_feedback_mpc
import mim_solvers
import numpy as np
import pinocchio as pin
from agimus_controller.factory.robot_model import RobotModels
from agimus_controller.mpc_data import OCPDebugData, OCPResults
from agimus_controller.ocp_base_croco import OCPBaseCroco
from agimus_controller.ocp_param_base import OCPParamsBaseCroco
from agimus_controller.trajectory import WeightedTrajectoryPoint


@dataclass
class OCPReferences:
    reference: Any
    weights: np.array
    frame_id: Any = None
    use_residual: bool = True


@dataclass
class OCPCosts:
    state_reg: OCPReferences | None = None
    frame_translation: OCPReferences | None = None
    frame_rotation: OCPReferences | None = None
    frame_velocity: OCPReferences | None = None
    force_tracking: OCPReferences | None = None
    control_reg: OCPReferences | None = None


@dataclass
class OCPParamsCrocoForceFeedback(OCPParamsBaseCroco):
    with_gravity_torque_reg: bool = False

    x_reg_cost_weight: float = 0.0
    frame_translation_cost_weight: float = 0.0
    frame_rotation_cost_weight: float = 0.0
    frame_velocity_cost_weight: float = 0.0

    frame_of_interest: str = ""

    Kp: np.array = None
    Kv: np.array = None
    oPc_offset: np.array = None

    with_force_cost: bool = False

    contact_type: str = None

    solver_nthreads: int = 1

    def __post_init__(self):
        super().__post_init__()

        correct_masks = ("1Dx", "1Dy", "1Dz", "3D")
        assert self.contact_type in correct_masks, (
            f"Incorrect value '{self.contact_type}' for field 'contact_type'. "
            + f"Should be one of {correct_masks}."
        )

        if self.contact_type == "3D":
            contact_param_len = 3
        else:
            contact_param_len = 1

        for field, expected_length in (
            ("Kp", contact_param_len),
            ("Kv", contact_param_len),
            ("oPc_offset", 3),
        ):
            field_len = len(getattr(self, field))
            assert field_len == expected_length, (
                f"Field '{field}' has {field_len} elements, expected {expected_length} "
                + f"{'elements' if contact_param_len > 1 else 'element'}."
            )


class OCPCrocoForceFeedback(OCPBaseCroco):
    def __init__(
        self,
        robot_models: RobotModels,
        ocp_params: OCPParamsCrocoForceFeedback,
    ) -> None:
        """Defines common behavior for all OCP using croccodyl. This is an abstract class with some helpers to design OCPs in a more friendly way.

        Args:
            robot_models (RobotModels): All models of the robot.
            ocp_params (OCPParamsBaseCroco): Input data structure of the OCP.
        """
        # Setting the robot model
        self._robot_models = robot_models
        self._collision_model = self._robot_models.collision_model
        self._armature = self._robot_models.armature

        # Stat and actuation model
        self._state = crocoddyl.StateMultibody(self._robot_models.robot_model)
        self._actuation = crocoddyl.ActuationModelFull(self._state)

        # Setting the OCP parameters
        self._ocp_params = ocp_params
        self._solver = None
        self._ocp_results: OCPResults = None
        self._debug_data: OCPDebugData = OCPDebugData()

        # Create the running models
        self._running_model_list = self.create_running_model_list()
        # Create the terminal model
        self._terminal_model = self.create_terminal_model()
        # Create the shooting problem
        self._problem = crocoddyl.ShootingProblem(
            np.zeros(
                self._robot_models.robot_model.nq
                + self._robot_models.robot_model.nv
                + 3
            ),
            self._running_model_list,
            self._terminal_model,
        )
        self._problem.nthreads = self._ocp_params.solver_nthreads
        # Create solver + callbacks
        self._solver = mim_solvers.SolverCSQP(self._problem)

        # Merit function
        self._solver.use_filter_line_search = self._ocp_params.use_filter_line_search

        # Parameters of the solver
        if self._ocp_params.max_solve_time is not None:
            self._solver.max_solve_time = self._ocp_params.max_solve_time
        self._solver.termination_tolerance = self._ocp_params.termination_tolerance
        self._solver.max_qp_iters = self._ocp_params.qp_iters
        self._solver.eps_abs = self._ocp_params.eps_abs
        self._solver.eps_rel = self._ocp_params.eps_rel

        if self._ocp_params.callbacks:
            self._solver.setCallbacks(
                [mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()]
            )

        if self._ocp_params.contact_type == "3D":
            self._force_mask = None
        else:
            self._force_mask = "xyz".index(self._ocp_params.contact_type[-1])

        self._references_cache: list[OCPCosts] = []

    def create_running_model_list(self) -> list[crocoddyl.ActionModelAbstract]:
        running_model_list = []
        for dt in self._ocp_params.timesteps:
            # Running cost model
            running_cost_model = crocoddyl.CostModelSum(self._state)

            ### Creation of cost terms
            # State Regularization cost
            x_residual = crocoddyl.ResidualModelState(
                self._state,
                np.ones(
                    self._robot_models.robot_model.nq
                    + self._robot_models.robot_model.nv
                ),
            )
            x_reg_cost = crocoddyl.CostModelResidual(
                self._state,
                crocoddyl.ActivationModelWeightedQuad(
                    np.ones(
                        self._robot_models.robot_model.nq
                        + self._robot_models.robot_model.nv
                    )
                ),
                x_residual,
            )
            running_cost_model.addCost(
                "stateReg", x_reg_cost, self._ocp_params.x_reg_cost_weight
            )

            # Frame related costs
            frame_of_interest_id = self._robot_models.robot_model.getFrameId(
                self._ocp_params.frame_of_interest
            )
            frame_target_pose = pin.SE3(pin.Quaternion(), np.zeros(3))

            # Frame translation cost
            frame_translation_residual = crocoddyl.ResidualModelFrameTranslation(
                self._state,
                frame_of_interest_id,
                frame_target_pose.translation,
                self._actuation.nu,
            )
            frame_translation_activation = crocoddyl.ActivationModelWeightedQuad(
                np.zeros(3)
            )
            frame_translation_cost = crocoddyl.CostModelResidual(
                self._state,
                frame_translation_activation,
                frame_translation_residual,
            )
            running_cost_model.addCost(
                "frameTranslationCost",
                frame_translation_cost,
                self._ocp_params.frame_translation_cost_weight,
            )

            # Frame rotation cost
            frame_rotation_residual = crocoddyl.ResidualModelFrameRotation(
                self._state,
                frame_of_interest_id,
                frame_target_pose.rotation,
                self._actuation.nu,
            )
            frame_rotation_activation = crocoddyl.ActivationModelWeightedQuad(
                np.zeros(3)
            )
            frame_rotation_cost = crocoddyl.CostModelResidual(
                self._state, frame_rotation_activation, frame_rotation_residual
            )
            running_cost_model.addCost(
                "frameRotationCost",
                frame_rotation_cost,
                self._ocp_params.frame_rotation_cost_weight,
            )

            # Frame velocity cost
            frame_velocity_residual = crocoddyl.ResidualModelFrameVelocity(
                self._state,
                frame_of_interest_id,
                pin.Motion(np.zeros(6)),
                pin.WORLD,
                self._actuation.nu,
            )
            frame_velocity_activation = crocoddyl.ActivationModelWeightedQuad(
                np.zeros(6)
            )
            frameVelocityCost = crocoddyl.CostModelResidual(
                self._state, frame_velocity_activation, frame_velocity_residual
            )
            running_cost_model.addCost(
                "frameVelocityCost",
                frameVelocityCost,
                self._ocp_params.frame_velocity_cost_weight,
            )

            # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
            dam_params = {
                "state": self._state,
                "actuation": self._actuation,
                "costs": running_cost_model,
                "frameId": frame_of_interest_id,
                "Kp": self._ocp_params.Kp,
                "Kv": self._ocp_params.Kv,
                "oPc": self._ocp_params.oPc_offset,
            }
            if self._ocp_params.contact_type == "3D":
                running_DAM = force_feedback_mpc.DAMSoftContact3DAugmentedFwdDynamics(
                    **dam_params,
                )
                desired_force = np.zeros(3)
                desired_force_weights = np.zeros(3)
            elif self._ocp_params.contact_type in ("1Dx", "1Dy", "1Dz"):
                contact_type = getattr(
                    force_feedback_mpc.Vector3MaskType,
                    self._ocp_params.contact_type[-1],
                )
                running_DAM = force_feedback_mpc.DAMSoftContact1DAugmentedFwdDynamics(
                    **dam_params,
                    type=contact_type,
                )
                desired_force = np.zeros(1)
                desired_force_weights = np.zeros(1)
            running_model = force_feedback_mpc.IAMSoftContactAugmented(running_DAM, dt)
            running_model.differential.with_gravity_torque_reg = (
                self._ocp_params.with_gravity_torque_reg
            )
            running_model.differential.tau_grav_weight = 0.0

            # Frame force cost
            running_model.differential.with_force_cost = True
            running_model.differential.f_des = desired_force
            running_model.differential.f_weight = desired_force_weights
            running_model.differential.armature = self._robot_models.armature

            running_model_list.append(running_model)

        assert len(running_model_list) == self.n_controls
        return running_model_list

    def create_terminal_model(self) -> crocoddyl.ActionModelAbstract:
        # Terminal cost models
        terminal_cost_model = crocoddyl.CostModelSum(self._state)

        ### Creation of cost terms
        # State Regularization cost
        x_residual = crocoddyl.ResidualModelState(
            self._state,
            np.zeros(
                self._robot_models.robot_model.nq + self._robot_models.robot_model.nv
            ),
        )
        x_reg_cost = crocoddyl.CostModelResidual(
            self._state,
            crocoddyl.ActivationModelWeightedQuad(
                np.ones(
                    self._robot_models.robot_model.nq
                    + self._robot_models.robot_model.nv
                )
            ),
            x_residual,
        )
        terminal_cost_model.addCost(
            "stateReg", x_reg_cost, self._ocp_params.x_reg_cost_weight
        )

        # Frame related costs
        frame_of_interest_id = self._robot_models.robot_model.getFrameId(
            self._ocp_params.frame_of_interest
        )
        frame_target_pose = pin.SE3(pin.Quaternion(), np.zeros(3))

        # Frame translation cost
        frame_translation_residual = crocoddyl.ResidualModelFrameTranslation(
            self._state,
            frame_of_interest_id,
            frame_target_pose.translation,
            self._actuation.nu,
        )
        frame_translation_activation = crocoddyl.ActivationModelWeightedQuad(np.ones(3))
        frame_translation_cost = crocoddyl.CostModelResidual(
            self._state,
            frame_translation_activation,
            frame_translation_residual,
        )
        terminal_cost_model.addCost(
            "frameTranslationCost",
            frame_translation_cost,
            self._ocp_params.frame_translation_cost_weight,
        )

        # Frame rotation cost
        frame_rotation_residual = crocoddyl.ResidualModelFrameRotation(
            self._state,
            frame_of_interest_id,
            frame_target_pose.rotation,
            self._actuation.nu,
        )
        frame_rotation_activation = crocoddyl.ActivationModelWeightedQuad(np.ones(3))
        frame_rotation_cost = crocoddyl.CostModelResidual(
            self._state, frame_rotation_activation, frame_rotation_residual
        )
        terminal_cost_model.addCost(
            "frameRotationCost",
            frame_rotation_cost,
            self._ocp_params.frame_rotation_cost_weight,
        )

        # Frame velocity cost
        frame_velocity_residual = crocoddyl.ResidualModelFrameVelocity(
            self._state,
            frame_of_interest_id,
            pin.Motion(np.zeros(6)),
            pin.WORLD,
            self._actuation.nu,
        )
        frame_velocity_activation = crocoddyl.ActivationModelWeightedQuad(np.zeros(6))
        frameVelocityCost = crocoddyl.CostModelResidual(
            self._state, frame_velocity_activation, frame_velocity_residual
        )
        terminal_cost_model.addCost(
            "frameVelocityCost",
            frameVelocityCost,
            self._ocp_params.frame_velocity_cost_weight,
        )

        # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
        dam_params = {
            "state": self._state,
            "actuation": self._actuation,
            "costs": terminal_cost_model,
            "frameId": frame_of_interest_id,
            "Kp": self._ocp_params.Kp,
            "Kv": self._ocp_params.Kv,
            "oPc": self._ocp_params.oPc_offset,
        }
        if self._ocp_params.contact_type == "3D":
            terminal_DAM = force_feedback_mpc.DAMSoftContact3DAugmentedFwdDynamics(
                **dam_params
            )
            desired_force = np.zeros(3)
            desired_force_weights = np.zeros(3)
        elif self._ocp_params.contact_type in ("1Dx", "1Dy", "1Dz"):
            contact_type = getattr(
                force_feedback_mpc.Vector3MaskType,
                self._ocp_params.contact_type[-1],
            )
            terminal_DAM = force_feedback_mpc.DAMSoftContact1DAugmentedFwdDynamics(
                **dam_params,
                type=contact_type,
            )
            desired_force = np.zeros(1)
            desired_force_weights = np.zeros(1)
        terminal_model = force_feedback_mpc.IAMSoftContactAugmented(terminal_DAM, 0.0)

        terminal_model.differential.with_gravity_torque_reg = False
        terminal_model.differential.with_force_cost = True
        terminal_model.differential.f_des = desired_force
        terminal_model.differential.f_weight = desired_force_weights
        terminal_model.differential.armature = self._robot_models.armature
        return terminal_model

    def create_reference_cache(self):
        models = [self._solver.problem.runningModels]
        models.append(self._solver.problem.terminalModel)

        for model in models:
            running_model_diff = model.differential

            # Create new cache object
            ocp_ref_cache = OCPCosts()

            # By using `__dict__["element"]` obtain references to functions in order to cache them
            state_reg = running_model_diff.costs.costs["stateReg"]
            ocp_ref_cache.state_reg = OCPReferences(
                reference=state_reg.cost.residual.__dict__["reference"],
                weights=state_reg.cost.activation.__dict__["weights"],
            )

            frame_translation = running_model_diff.costs.costs["frameTranslationCost"]
            ocp_ref_cache.frame_translation = OCPReferences(
                reference=frame_translation.cost.residual.__dict__["reference"],
                weights=frame_translation.cost.activation.__dict__["weights"],
                frame_id=frame_translation.cost.residual.__dict__["id"],
            )

            frame_rotation = running_model_diff.costs.costs["frameRotationCost"]
            ocp_ref_cache.frame_rotation = OCPReferences(
                reference=frame_rotation.cost.residual.__dict__["reference"],
                weights=frame_rotation.cost.activation.__dict__["weights"],
                frame_id=frame_rotation.cost.residual.__dict__["id"],
            )

            frame_velocity = running_model_diff.costs.costs["frameVelocityCost"]
            ocp_ref_cache.frame_velocity = OCPReferences(
                reference=frame_velocity.cost.residual.__dict__["reference"],
                weights=frame_velocity.cost.activation.__dict__["weights"],
                frame_id=frame_velocity.cost.residual.__dict__["id"],
            )

            ocp_ref_cache.control_reg = OCPReferences(
                reference=None,
                weights=running_model_diff.__dict__["tau_grav_weight"],
            )

            ocp_ref_cache.force_tracking = OCPReferences(
                use_residual=running_model_diff.__dict__["active_contact"],
                reference=running_model_diff.__dict__["f_des"],
                weights=running_model_diff.__dict__["f_weight"],
            )

            self._references_cache.append(ocp_ref_cache)

    def set_reference_weighted_trajectory(
        self, reference_weighted_trajectory: list[WeightedTrajectoryPoint]
    ):
        """Set the reference trajectory for the OCP."""
        assert len(reference_weighted_trajectory) == self.n_controls + 1

        # Modify running costs reference and weights
        for i in range(len(reference_weighted_trajectory)):
            ref_weighted_pt = reference_weighted_trajectory[i]
            ocp_ref = self._references_cache[i]

            ocp_ref.state_reg.reference = np.concatenate(
                (
                    ref_weighted_pt.point.robot_configuration,
                    ref_weighted_pt.point.robot_velocity,
                )
            )
            ocp_ref.state_reg.weights = np.concatenate(
                (
                    ref_weighted_pt.point.robot_configuration,
                    ref_weighted_pt.point.robot_velocity,
                )
            )

            ee_name = ref_weighted_pt.weights.w_end_effector_poses.keys()[0]
            ee_id = self._robot_models.robot_model.getFrameId(ee_name)

            ocp_ref.frame_translation.frame_id = ee_id
            ocp_ref.frame_translation.reference = (
                ref_weighted_pt.point.end_effector_poses[ee_name].translation
            )
            ocp_ref.frame_translation.weights = (
                ref_weighted_pt.weights.w_end_effector_poses[ee_name][:3]
            )

            ocp_ref.frame_rotation.frame_id = ee_id
            ocp_ref.frame_rotation.reference = ref_weighted_pt.point.end_effector_poses[
                ee_name
            ].rotation
            ocp_ref.frame_rotation.weights = (
                ref_weighted_pt.weights.w_end_effector_poses[ee_name][:3]
            )

            ocp_ref.frame_velocity.frame_id = ee_id
            ocp_ref.frame_velocity.reference = (
                ref_weighted_pt.point.end_effector_velocities[ee_name]
            )
            ocp_ref.frame_velocity.weights = (
                ref_weighted_pt.weights.w_end_effector_velocities[ee_name]
            )

            ocp_ref.control_reg.weights = ref_weighted_pt.weights.w_robot_effort[0]

            desired_force_weights = ref_weighted_pt.weights.w_forces[ee_name]
            ocp_ref.frame_velocity.use_residual = (
                np.sum(np.abs(desired_force_weights)) > 1e-9
            )
            ocp_ref.frame_velocity.reference = ref_weighted_pt.point.forces[
                ee_name
            ].linear
            ocp_ref.frame_velocity.weights = desired_force_weights
