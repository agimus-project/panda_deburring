#include "panda_deburring/ft_calibration_filer.hpp"

#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/math/rpy.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/skew.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/wait_for_message.hpp>
#include <std_msgs/msg/string.hpp>

#include "controller_interface/helpers.hpp"

namespace ft_calibration_filter {

controller_interface::CallbackReturn FTCalibrationFilter::on_init() {
  try {
    param_listener_ = std::make_shared<ParamListener>(get_node());
    params_ = param_listener_->get_params();
  } catch (const std::exception &e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n",
            e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
FTCalibrationFilter::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration command_interfaces_config;
  command_interfaces_config.type =
      controller_interface::interface_configuration_type::INDIVIDUAL;
  // command_interfaces_config.names = params_.filtered_forces_interfaces_names;

  return command_interfaces_config;
}

controller_interface::InterfaceConfiguration
FTCalibrationFilter::state_interface_configuration() const {
  return controller_interface::InterfaceConfiguration{
      controller_interface::interface_configuration_type::ALL};
}

controller_interface::CallbackReturn FTCalibrationFilter::on_configure(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  RCLCPP_INFO(this->get_node()->get_logger(), "configure successful");

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn FTCalibrationFilter::on_activate(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  if (params_.reference_interfaces_names.size() > 0 && !is_in_chained_mode()) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Wrong activation, the controller needs to be in chain mode.");
  }

  params_ = param_listener_->get_params();

  if (!controller_interface::get_ordered_interfaces(
          command_interfaces_, params_.filtered_forces_interfaces_names,
          std::string(""), ordered_command_interfaces_)) {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Expected %zu command interfaces, found %zu",
                 params_.filtered_forces_interfaces_names.size(),
                 ordered_command_interfaces_.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  if (!controller_interface::get_ordered_interfaces(
          state_interfaces_, params_.state_force_interfaces_names,
          std::string(""), ordered_state_force_interfaces_)) {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Expected %zu force sensor state interfaces, found %zu",
                 params_.state_force_interfaces_names.size(),
                 ordered_state_force_interfaces_.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  if (!controller_interface::get_ordered_interfaces(
          state_interfaces_, params_.state_robot_position_interfaces_names,
          std::string(""), ordered_state_robot_position_interfaces_)) {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Expected %zu robot position state interfaces, found %zu",
                 params_.state_robot_position_interfaces_names.size(),
                 ordered_state_robot_position_interfaces_.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  std_msgs::msg::String robot_description_msg;
  auto robot_description_sub =
      get_node()->create_subscription<std_msgs::msg::String>(
          "/robot_description", rclcpp::QoS(1).transient_local(),
          [](const std::shared_ptr<const std_msgs::msg::String>) {});

  const std::size_t retires = 10;
  for (std::size_t i = 0; i < retires; i++) {
    if (rclcpp::wait_for_message(robot_description_msg, robot_description_sub,
                                 get_node()->get_node_options().context(),
                                 std::chrono::seconds(1))) {
      RCLCPP_INFO(get_node()->get_logger(), "Robot description received.");
      break;
    } else if (i == retires) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Filed to receive data on '/robot_description' topic.");
      return controller_interface::CallbackReturn::ERROR;
    }
    RCLCPP_INFO(get_node()->get_logger(),
                "Robot description still not received. Retrying...");
  }

  pinocchio::Model robot_model_full;
  pinocchio::urdf::buildModelFromXML(robot_description_msg.data,
                                     robot_model_full);
  std::vector<pinocchio::JointIndex> moving_joint_ids;
  for (const std::string joint_name : params_.moving_joint_names) {
    if (robot_model_full.existJointName(joint_name)) {
      const auto joint_id = robot_model_full.getJointId(joint_name);
      moving_joint_ids.push_back(joint_id);
    } else {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Filed to find joint with a name '%s' in the URDF!",
                   joint_name.c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
  }
  std::vector<pinocchio::JointIndex> locked_joint_ids;
  for (std::size_t i = 0; i < robot_model_full.njoints; i++) {
    if (std::find(moving_joint_ids.begin(), moving_joint_ids.end(), i) ==
        moving_joint_ids.end()) {
      locked_joint_ids.push_back(i);
    }
  }

  Eigen::VectorXd q_default_complete =
      Eigen::VectorXd::Zero(robot_model_full.nq);
  robot_model_ = pinocchio::buildReducedModel(
      robot_model_full, locked_joint_ids, q_default_complete);
  robot_data_ = pinocchio::Data(robot_model_);

  // Preallocate matrix storing bias values for averaging
  bias_measurements_ = Matrix6Xd(6, params_.bias_measurement_samples);
  // Preallocate state vector
  q_ = Eigen::VectorXd::Zero(robot_model_.nq);

  g_ = Eigen::Map<Eigen::Vector3d, Eigen::Unaligned>(
      params_.gravity_vector.data(), params_.gravity_vector.size());

  auto &calib = params_.calibration.measurement_frame;
  Eigen::Vector3d rpy = Eigen::Map<Eigen::Vector3d, Eigen::Unaligned>(
      calib.rpy.data(), calib.rpy.size());
  Eigen::Vector3d xyz = Eigen::Map<Eigen::Vector3d, Eigen::Unaligned>(
      calib.xyz.data(), calib.xyz.size());
  calibration_ = pinocchio::SE3(pinocchio::rpy::rpyToMatrix(rpy), xyz);
  calibration_trans_.template topLeftCorner<3, 3>() =
      Eigen::Matrix3d::Identity();
  calibration_trans_.template bottomLeftCorner<3, 3>() = pinocchio::skew(xyz);

  const auto &frame_name = params_.measurement_frame_id;
  if (!robot_model_.existFrame(frame_name)) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Filed to find find with a name '%s' in the URDF!",
                 frame_name.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  frame_of_interest_id_ = robot_model_.getFrameId(frame_name);

  bias_buffer_cnt_ = 0;
  bias_computed_ = false;
  avg_bias_ = pinocchio::Force::Zero();
  RCLCPP_INFO(this->get_node()->get_logger(), "activate successful");

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn FTCalibrationFilter::on_deactivate(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  ordered_state_force_interfaces_.clear();
  ordered_state_robot_position_interfaces_.clear();
  ordered_command_interfaces_.clear();
  release_interfaces();
  return controller_interface::CallbackReturn::SUCCESS;
}

bool FTCalibrationFilter::on_set_chained_mode(bool /*chained_mode*/) {
  return params_.reference_interfaces_names.size() > 0;
}

controller_interface::return_type
FTCalibrationFilter::update_and_write_commands(
    const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/) {
  // Gather measurements to later remove bias
  if (bias_buffer_cnt_ < bias_measurements_.size()) {
    for (std::size_t i = 0; i < bias_measurements_.cols(); i++) {
      bias_measurements_(bias_buffer_cnt_, i) =
          ordered_state_force_interfaces_[i].get().get_value();
    }
    bias_buffer_cnt_++;
    return controller_interface::return_type::OK;
  }
  // If all data required was acquired, compute average bias
  if (!bias_computed_) {
    avg_bias_.toVector() = bias_measurements_.rowwise().mean();
    bias_computed_ = true;
    RCLCPP_INFO(this->get_node()->get_logger(), "Bias computation finished.");
  }

  for (std::size_t i = 0; i < 6; i++) {
    force_.toVector()[i] = ordered_state_force_interfaces_[i].get().get_value();
  }
  for (std::size_t i = 0; i < q_.size(); i++) {
    q_[i] = ordered_state_robot_position_interfaces_[i].get().get_value();
  }

  pinocchio::forwardKinematics(robot_model_, robot_data_, q_);
  pinocchio::SE3 T_frame = pinocchio::updateFramePlacement(
      robot_model_, robot_data_, frame_of_interest_id_);

  const auto f = force_.toVector() - avg_bias_.toVector();
  const double m = params_.calibration.com.mass;
  const auto f_gravity =
      calibration_trans_ * (m * calibration_.rotation().transpose() *
                            T_frame.rotation().transpose() * g_);
  const auto f_out = f - f_gravity;

  for (std::size_t i = 0; i < f_out.size(); i++) {
    // reference_interfaces_[i] = f_out[i];
    ordered_command_interfaces_[i].get().set_value(f_out[i]);
  }

  return controller_interface::return_type::OK;
}

std::vector<hardware_interface::CommandInterface>
FTCalibrationFilter::on_export_reference_interfaces() {
  std::vector<hardware_interface::CommandInterface> reference_interfaces;

  const auto names = params_.reference_interfaces_names;
  reference_interfaces_.resize(names.size());
  for (std::size_t i = 0; i < names.size(); i++) {
    reference_interfaces.push_back(hardware_interface::CommandInterface(
        get_node()->get_name(), names[i], &reference_interfaces_[i]));
  }

  return reference_interfaces;
}

controller_interface::return_type
FTCalibrationFilter::update_reference_from_subscribers() {
  return controller_interface::return_type::OK;
}

}  // namespace ft_calibration_filter

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(ft_calibration_filter::FTCalibrationFilter,
                       controller_interface::ChainableControllerInterface)
