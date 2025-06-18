#ifndef PANDA_DEBURRING__CONTACT_DETECTOR_
#define PANDA_DEBURRING__CONTACT_DETECTOR_

#include <pinocchio/spatial/force.hpp>

namespace ft_calibration_filter {
class ContactDetector {
 public:
  ContactDetector() {}

  void update(const pinocchio::Force &force) {
    const double thresh =
        filterted_state_ ? lower_threshold_ : upper_threshold_;
    last_measured_state_ = force.linear().norm() > thresh;
  }

  bool in_contact() {
    if (last_measured_state_ != filterted_state_ &&
        samples_since_switch_ > hysteresis_samples_) {
      filterted_state_ = last_measured_state_;
      samples_since_switch_ = 0;
    } else {
      samples_since_switch_++;
      // Prevent integer overflow
      if (samples_since_switch_ > 5 * hysteresis_samples_) {
        samples_since_switch_ = hysteresis_samples_;
      }
    }

    return filterted_state_;
  }

  void set_hysteresis_samples(const unsigned int hysteresis_samples) {
    hysteresis_samples_ = hysteresis_samples;
  }

  unsigned int get_hysteresis_samples() const { return hysteresis_samples_; }

  void set_lower_threshold(const double lower_threshold) {
    lower_threshold_ = lower_threshold;
  }

  double get_lower_threshold() const { return lower_threshold_; }

  void set_upper_threshold(const double upper_threshold) {
    upper_threshold_ = upper_threshold;
  }

  double get_upper_threshold() const { return upper_threshold_; }

 private:
  bool last_measured_state_ = false;
  bool filterted_state_ = false;
  unsigned int samples_since_switch_ = 0;
  unsigned int hysteresis_samples_ = 0;
  double lower_threshold_ = 0.0;
  double upper_threshold_ = 0.0;
};
}  // namespace ft_calibration_filter

#endif  // PANDA_DEBURRING__CONTACT_DETECTOR_
