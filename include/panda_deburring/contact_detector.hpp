#ifndef PANDA_DEBURRING__CONTACT_DETECTOR_
#define PANDA_DEBURRING__CONTACT_DETECTOR_

#include <pinocchio/spatial/force.hpp>

namespace ft_calibration_filter {
class ContactDetector {
 public:
  ContactDetector() {}

  void update(const pinocchio::Force &force) {
    last_measured_state_ = force.linear().norm() > thresh_;
  }

  void set_hysteresis_samples(const unsigned int hysteresis_samples) {
    hysteresis_samples_ = hysteresis_samples;
  }

  unsigned int get_hysteresis_samples() const { return hysteresis_samples_; }

  bool in_contact() {
    if (last_measured_state_ != filterted_state_ &&
        samples_since_switch_ > hysteresis_samples_) {
      filterted_state_ = last_measured_state_;
      samples_since_switch_ = 0;
    } else {
      samples_since_switch_++;
    }

    return filterted_state_;
  }

 private:
  bool last_measured_state_ = false;
  bool filterted_state_ = false;
  const double thresh_ = 0.0;
  unsigned int samples_since_switch_ = 0;
  unsigned int hysteresis_samples_ = 0;
};
}  // namespace ft_calibration_filter

#endif  // PANDA_DEBURRING__CONTACT_DETECTOR_
