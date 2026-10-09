#pragma once

#include <array>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

struct LightBar {
  cv::RotatedRect rect;
  float length = 0.0F;
  float width = 0.0F;
  float long_axis_angle_deg = 0.0F;
  double area = 0.0;
};

struct ArmorObservation {
  std::array<cv::Point2f, 4> corners{};
  cv::Point2f center{};
  cv::Size2f size{};
  float score = 0.0F;
};

struct DetectionFrame {
  std::vector<LightBar> lights;
  std::vector<ArmorObservation> armors;
};

struct ProjectedArmor {
  int id = 0;
  bool valid = false;
  std::array<cv::Point2f, 4> corners{};
  cv::Point2f center{};
};

enum class BoxSource {
  kNone,
  kObserved,
  kPredicted,
};

struct ArmorSlotOutput {
  int id = 0;
  bool valid = false;
  BoxSource source = BoxSource::kNone;
  std::array<cv::Point2f, 4> corners{};
  cv::Point2f center{};
};

enum class MotionMode {
  kInitializing,
  kStopped,
  kUniform,
  kAccelerating,
  kDecelerating,
  kReversing,
  kLost,
};

struct SolverOutput {
  bool model_valid = false;
  bool prediction_valid = false;
  bool candidate_available = false;
  bool measurement_used = false;
  bool candidate_measurement_used = false;
  bool image_motion_prediction_used = false;
  bool image_prediction_coasting = false;
  bool image_candidate_used = false;
  bool image_handover_detected = false;
  bool image_future_handover = false;
  bool image_handover_spatial_vote = false;
  bool image_handover_progress_vote = false;
  bool image_handover_age_vote = false;
  int image_handover_vote_count = 0;
  int image_track_age_frames = 0;
  double image_track_progress = 0.0;
  double image_normalized_step_px = -1.0;
  int detected_slot_index = -1;
  int measurement_slot_index = -1;
  int candidate_association_slot_index = -1;
  int future_slot_index = -1;
  int missed_frames = 0;
  double phase_rad = 0.0;
  double future_phase_rad = 0.0;
  double prediction_lead_s = 0.0;
  int prediction_lead_frames = 0;
  double angular_speed_rad_s = 0.0;
  double angular_acceleration_rad_s2 = 0.0;
  double reprojection_error_px = 0.0;
  std::string direction = "UNKNOWN";
  MotionMode mode = MotionMode::kInitializing;
  ArmorObservation candidate;
  std::array<ArmorSlotOutput, 3> slots{};
  std::array<ArmorSlotOutput, 3> future_model_slots{};
  ArmorSlotOutput future_target{};
};
