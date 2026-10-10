#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "armor_detector.hpp"
#include "config.hpp"
#include "render_policy.hpp"
#include "rigid_model.hpp"
#include "types.hpp"

namespace {
class FramePreprocessor {
 public:
  FramePreprocessor(const Config& config, cv::Size input_size) : enabled_(config.camera_enabled) {
    if (!enabled_) return;
    const double sx = static_cast<double>(input_size.width) / config.camera_reference_width;
    const double sy = static_cast<double>(input_size.height) / config.camera_reference_height;
    camera_matrix_ = (cv::Mat_<double>(3, 3) << config.fx * sx, 0, config.cx * sx,
                      0, config.fy * sy, config.cy * sy, 0, 0, 1);
    distortion_ = (cv::Mat_<double>(1, 5) << config.k1, config.k2, config.p1, config.p2, config.k3);
  }

  cv::Mat process(const cv::Mat& input) const {
    if (!enabled_) return input;
    cv::Mat output;
    cv::undistort(input, output, camera_matrix_, distortion_);
    return output;
  }

 private:
  bool enabled_ = false;
  cv::Mat camera_matrix_, distortion_;
};

std::string mode_name(MotionMode mode) {
  switch (mode) {
    case MotionMode::kStopped: return "STOPPED";
    case MotionMode::kUniform: return "UNIFORM";
    case MotionMode::kAccelerating: return "ACCELERATING";
    case MotionMode::kDecelerating: return "DECELERATING";
    case MotionMode::kReversing: return "REVERSING";
    case MotionMode::kLost: return "LOST";
    default: return "INITIALIZING";
  }
}

std::string track_state_name(TrackState state) {
  switch (state) {
    case TrackState::kDetecting: return "DETECTING";
    case TrackState::kTracking: return "TRACKING";
    case TrackState::kTempLost: return "TEMP_LOST";
    default: return "LOST";
  }
}

cv::Mat motion_signature(const cv::Mat& frame, const Config& config) {
  const int x0 = cvRound(config.roi_x_min * frame.cols);
  const int y0 = cvRound(config.roi_y_min * frame.rows);
  const int x1 = cvRound(config.roi_x_max * frame.cols);
  const int y1 = cvRound(config.roi_y_max * frame.rows);
  cv::Mat gray, signature;
  cv::cvtColor(frame(cv::Rect(x0, y0, std::max(1, x1 - x0), std::max(1, y1 - y0))),
               gray, cv::COLOR_BGR2GRAY);
  cv::resize(gray, signature, {160, 96}, 0, 0, cv::INTER_AREA);
  return signature;
}

class OnlineMotionDetector {
 public:
  explicit OnlineMotionDetector(const Config& config) : config_(config) {}
  bool update(const cv::Mat& frame) {
    cv::Mat current = motion_signature(frame, config_);
    if (previous_.empty()) { previous_ = current; return false; }
    const double energy = cv::norm(current, previous_, cv::NORM_L1) /
                          static_cast<double>(current.total());
    previous_ = current;
    if (energy > config_.motion_difference_threshold) {
      positive_ = std::min(positive_ + 1, config_.motion_hold_frames);
      negative_ = 0;
      if (positive_ >= config_.motion_hold_frames) moving_ = true;
    } else {
      negative_ = std::min(negative_ + 1, config_.motion_hold_frames);
      positive_ = 0;
      if (negative_ >= config_.motion_hold_frames) moving_ = false;
    }
    return moving_;
  }
 private:
  Config config_;
  cv::Mat previous_;
  int positive_ = 0, negative_ = 0;
  bool moving_ = false;
};

void draw_quad(cv::Mat& image, const std::array<cv::Point2f, 4>& corners,
               const cv::Scalar& color, int thickness) {
  for (int i = 0; i < 4; ++i)
    cv::line(image, corners[i], corners[(i + 1) % 4], color, thickness, cv::LINE_AA);
}

void draw_dashed_quad(cv::Mat& image, const std::array<cv::Point2f, 4>& corners,
                      const cv::Scalar& color, int thickness) {
  constexpr float kDashLength = 6.0F;
  constexpr float kGapLength = 4.0F;
  for (int i = 0; i < 4; ++i) {
    const cv::Point2f start = corners[i];
    const cv::Point2f delta = corners[(i + 1) % 4] - start;
    const float length = static_cast<float>(cv::norm(delta));
    if (length < 1.0F) continue;
    const cv::Point2f direction = delta * (1.0F / length);
    for (float offset = 0.0F; offset < length; offset += kDashLength + kGapLength) {
      const float dash_end = std::min(length, offset + kDashLength);
      cv::line(image, start + direction * offset, start + direction * dash_end,
               color, thickness, cv::LINE_AA);
    }
  }
}

std::string source_name(BoxSource source) {
  switch (source) {
    case BoxSource::kObserved: return "current_detection";
    case BoxSource::kPredicted: return "current_model";
    default: return "none";
  }
}

void draw_overlay(cv::Mat& frame, const SolverOutput& state, int frame_number,
                  const Config& config,
                  const PredictionRenderDecision& prediction_render,
                  bool draw_labels, bool debug_lights) {
  for (const auto& slot : state.slots) {
    if (!slot.valid || slot.source == BoxSource::kNone) continue;
    const bool detected_now = slot.source == BoxSource::kObserved;
    const cv::Scalar color(0, 255, 0);
    if (detected_now) {
      draw_quad(frame, slot.corners, color, 2);
      cv::circle(frame, slot.center, 2, color, cv::FILLED);
    } else {
      draw_dashed_quad(frame, slot.corners, color, 1);
    }
    if (draw_labels) {
      const std::string label = "A" + std::to_string(slot.id) +
          (detected_now ? " NOW" : "");
      cv::putText(frame, label, slot.center + cv::Point2f(4, -5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv::LINE_AA);
    }
  }

  if (prediction_render.draw) {
    const cv::Scalar future_color(0, 0, 255);
    if (prediction_render.high_overlap) {
      cv::Mat overlay = frame.clone();
      draw_dashed_quad(overlay, state.future_target.corners, future_color, 1);
      cv::circle(overlay, state.future_target.center, 3, future_color, 1,
                 cv::LINE_AA);
      cv::addWeighted(overlay, config.prediction_overlap_alpha, frame,
                      1.0 - config.prediction_overlap_alpha, 0.0, frame);
    } else {
      draw_dashed_quad(frame, state.future_target.corners, future_color, 2);
    }
    if (draw_labels) {
      const std::string label = "A" + std::to_string(state.future_target.id) + " FUT +" +
          std::to_string(state.prediction_lead_frames) + "F";
      cv::putText(frame, label, state.future_target.center + cv::Point2f(4, -5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.43, future_color, 1, cv::LINE_AA);
    }
  }

  const int panel_width = std::min(frame.cols - 20, 540);
  const int panel_top = frame.rows - 125;
  cv::rectangle(frame, {10, panel_top}, {10 + panel_width, frame.rows - 10},
                cv::Scalar(10, 10, 10), cv::FILLED);
  std::string detection_state = "NO DETECTION";
  if (state.candidate_available) {
    detection_state = "DET A" + std::to_string(state.detected_slot_index + 1) +
        (state.candidate_measurement_used ? " / USED" : " / GATED");
  }
  cv::putText(frame, "frame=" + std::to_string(frame_number) + "  " + detection_state,
              {20, panel_top + 24}, cv::FONT_HERSHEY_SIMPLEX, 0.48,
              cv::Scalar(225, 225, 225), 1,
              cv::LINE_AA);
  std::ostringstream line2, line3, line4;
  line2 << "direction=" << state.direction << "  omega=" << std::fixed << std::setprecision(1)
        << std::abs(state.angular_speed_rad_s) * 180.0 / CV_PI << " deg/s";
  line3 << "track=" << track_state_name(state.track_state)
        << "  confidence=" << std::fixed << std::setprecision(2)
        << state.track_confidence << "  mode=" << mode_name(state.mode) << "  future=+"
        << state.prediction_lead_frames << "F / +" << std::fixed << std::setprecision(3)
        << state.prediction_lead_s << "s  predictor="
        << (state.image_motion_prediction_used
                ? (state.image_prediction_coasting
                       ? "IMAGE-COAST"
                       : (state.future_target_filter_used ? "IMAGE+STATE" : "IMAGE"))
                : "PHASE")
        << "  missed=" << state.missed_frames;
  line4 << "green=current detection/model  red=future target"
        << (state.prediction_valid
                ? (prediction_render.high_overlap ? " (dim overlap)" : "")
                : " (expired)");
  cv::putText(frame, line2.str(), {20, panel_top + 52}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
              cv::Scalar(0, 220, 255), 2, cv::LINE_AA);
  cv::putText(frame, line3.str(), {20, panel_top + 80}, cv::FONT_HERSHEY_SIMPLEX, 0.50,
              cv::Scalar(0, 220, 255), 1, cv::LINE_AA);
  cv::putText(frame, line4.str(), {20, panel_top + 106}, cv::FONT_HERSHEY_SIMPLEX, 0.47,
              cv::Scalar(0, 220, 255), 1, cv::LINE_AA);
  (void)debug_lights;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <input.mp4> [output.mp4] [config.yaml] [--model model.yaml]"
                 " [--no-gui] [--debug-lights]\n";
    return 1;
  }
  const std::string input_path = argv[1];
  bool show_gui = true, debug_lights = false;
  std::string model_path = "config/outpost_model.yaml";
  std::vector<std::string> positional;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--no-gui") {
      show_gui = false;
    } else if (arg == "--debug-lights") {
      debug_lights = true;
    } else if (arg == "--model") {
      if (i + 1 >= argc) {
        std::cerr << "--model requires a YAML path\n";
        return 1;
      }
      model_path = argv[++i];
    } else if (!arg.empty() && arg[0] == '-') {
      std::cerr << "unknown option: " << arg << '\n';
      return 1;
    } else {
      positional.push_back(arg);
    }
  }
  if (positional.size() > 2) {
    std::cerr << "too many positional arguments\n";
    return 1;
  }
  const std::string output_path = positional.empty()
      ? "output/result.mp4" : positional[0];
  const std::string config_path = positional.size() < 2
      ? "config/video.yaml" : positional[1];

  try {
    Config config = Config::load(config_path);
    if (config.solver_mode == "pnp")
      throw std::runtime_error("pnp mode requires matching calibration and is intentionally disabled; use affine or auto");
    cv::VideoCapture capture(input_path);
    if (!capture.isOpened())
      throw std::runtime_error("cannot open input video: " + input_path);
    const cv::Size image_size(
        static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT)));
    double fps = capture.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps) || fps < 1.0) fps = 30.0;
    config.resolve_prediction_horizon(fps);
    const CalibrationProfile profile = load_calibration_profile(model_path);
    if (profile.image_size != image_size)
      throw std::runtime_error("calibration profile image size does not match input");
    if (profile.undistorted != config.camera_enabled)
      throw std::runtime_error(
          "calibration profile undistortion mode does not match config");
    const auto same_parameter = [](double first, double second) {
      const double scale = std::max({1.0, std::abs(first), std::abs(second)});
      return std::abs(first - second) <= 1e-10 * scale;
    };
    if (config.camera_enabled &&
        (profile.camera_reference_size.width != config.camera_reference_width ||
         profile.camera_reference_size.height != config.camera_reference_height ||
         !same_parameter(profile.camera_matrix(0, 0), config.fx) ||
         !same_parameter(profile.camera_matrix(1, 1), config.fy) ||
         !same_parameter(profile.camera_matrix(0, 2), config.cx) ||
         !same_parameter(profile.camera_matrix(1, 2), config.cy) ||
         !same_parameter(profile.distortion[0], config.k1) ||
         !same_parameter(profile.distortion[1], config.k2) ||
         !same_parameter(profile.distortion[2], config.p1) ||
         !same_parameter(profile.distortion[3], config.p2) ||
         !same_parameter(profile.distortion[4], config.k3))) {
      throw std::runtime_error(
          "calibration profile camera parameters do not match config");
    }
    if (profile.phase_quad_model.bin_count() != config.shape_phase_bins ||
        profile.phase_quad_model.covered_bins() < config.shape_min_covered_bins) {
      throw std::runtime_error(
          "calibration profile phase bins do not match config");
    }
    const AffineGeometry geometry = profile.geometry;
    const PhaseQuadModel phase_quad_model = profile.phase_quad_model;
    const MotionPrior motion_prior;
    std::cout << "Config: camera_undistort=" << (config.camera_enabled ? "on" : "off")
              << ", physical=(armor " << config.armor_width_m << 'x'
              << config.armor_height_m << " m, radius " << config.rotation_radius_m
              << " m, pitch " << config.outpost_pitch_deg << " deg)"
              << ", speed_snap=" << (config.angular_speed_snap_enabled ? "on" : "off")
              << " (" << config.angular_speed_snap_rad_s << " +/- "
              << config.angular_speed_snap_tolerance_rad_s << " rad/s)"
              << ", visible_s=" << config.track_prediction_visible_s
              << ", lost_s=" << config.track_lost_timeout_s << '\n';
    std::cout << "Calibration profile: " << model_path << " (static geometry and phase shape only)\n";
    std::cout << "Geometry: samples=" << geometry.calibration_samples
              << " center=(" << geometry.center.x << ',' << geometry.center.y << ')'
              << " axes=(" << cv::norm(geometry.axis_cos) << ',' << cv::norm(geometry.axis_sin) << ')'
              << " rms=" << geometry.calibration_rms_px << " px\n"
              << "Phase quad model: valid"
              << ", covered_bins=" << phase_quad_model.covered_bins() << '/'
              << phase_quad_model.bin_count() << '\n'
              << "Future prediction: +" << config.prediction_lead_frames
              << " frames / +" << config.prediction_lead_s << " s\n";
    FramePreprocessor preprocessor(config, image_size);
    ArmorDetector detector(config);
    RigidArmorSolver solver(config, geometry, phase_quad_model, motion_prior);
    OnlineMotionDetector motion_detector(config);
    PredictionRenderState prediction_render_state;

    const auto output_parent = std::filesystem::path(output_path).parent_path();
    if (!output_parent.empty()) std::filesystem::create_directories(output_parent);
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, image_size);
    if (!writer.isOpened()) throw std::runtime_error("cannot create output video: " + output_path);
    std::filesystem::path csv_path(output_path);
    csv_path.replace_extension(".csv");
    std::ofstream csv(csv_path);
    if (!csv) throw std::runtime_error("cannot create CSV: " + csv_path.string());
    csv << "frame,time_s,candidate_available,measurement_used,candidate_measurement_used,"
           "image_motion_prediction_used,"
           "image_prediction_coasting,image_candidate_used,image_handover_detected,"
           "image_future_handover,image_handover_spatial_vote,image_handover_progress_vote,"
           "image_handover_age_vote,image_handover_vote_count,image_track_age_frames,"
           "image_track_progress,image_normalized_step_px,detected_slot_index,"
           "measurement_slot_index,candidate_association_slot_index,"
           "candidate_x,candidate_y,raw_ellipse_phase_rad,phase_rad,future_phase_rad,"
           "prediction_lead_s,omega_rad_s,alpha_rad_s2,direction,mode,reprojection_error_px,"
           "missed_frames,future_target_valid,future_slot_index,future_target_id,"
           "future_target_x,future_target_y,"
           "raw_future_target_valid,raw_future_target_id,raw_future_target_x,"
           "raw_future_target_y,future_target_filter_used,future_target_filter_reset,"
           "future_target_filter_innovation_px,future_target_filter_velocity_x,"
           "future_target_filter_velocity_y,future_render_drawn,"
           "future_render_high_overlap,future_render_distance_px,future_render_iou,"
           "future_model1_x,future_model1_y,future_model1_area,"
           "future_model2_x,future_model2_y,future_model2_area,future_model3_x,future_model3_y,future_model3_area,"
           "slot1_source,slot1_x,slot1_y,slot2_source,slot2_x,slot2_y,"
           "slot3_source,slot3_x,slot3_y,track_state,track_confirmed,"
           "phase_gate_passed,timestamp_gap_reset,track_confirm_hits,"
           "time_since_measurement_s,phase_variance_rad2,speed_variance_rad2_s2,"
           "phase_innovation_rad,phase_innovation_variance,phase_nis,"
           "association_cost,track_confidence,physical_speed_valid\n";

    cv::Mat raw;
    int frame_number = 0, measured_frames = 0;
    while (capture.read(raw)) {
      cv::Mat frame = preprocessor.process(raw);
      const DetectionFrame detections = detector.detect(frame);
      const bool scene_moving = motion_detector.update(frame);
      double timestamp_s = capture.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
      if (!(timestamp_s > 0.0)) timestamp_s = frame_number / fps;
      const SolverOutput state = solver.update(detections.armors, timestamp_s, scene_moving);
      if (state.measurement_used) ++measured_frames;
      const PredictionRenderDecision prediction_render =
          evaluate_prediction_render(state, config, &prediction_render_state);
      draw_overlay(frame, state, frame_number, config, prediction_render, true,
                   debug_lights);
      if (debug_lights) {
        for (const auto& light : detections.lights) {
          cv::Point2f p[4]; light.rect.points(p);
          for (int i = 0; i < 4; ++i)
            cv::line(frame, p[i], p[(i + 1) % 4], cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
        }
      }
      const double raw_phase = state.candidate_available ? geometry.phase_of(state.candidate.center) : 0.0;
      const auto projected_area = [](const ArmorSlotOutput& slot) {
        const std::vector<cv::Point2f> corners(slot.corners.begin(), slot.corners.end());
        return std::abs(cv::contourArea(corners));
      };
      csv << frame_number << ',' << std::fixed << std::setprecision(6) << timestamp_s << ','
          << state.candidate_available << ',' << state.measurement_used << ','
          << state.candidate_measurement_used << ','
          << state.image_motion_prediction_used << ',' << state.image_prediction_coasting << ','
          << state.image_candidate_used << ',' << state.image_handover_detected << ','
          << state.image_future_handover << ','
          << state.image_handover_spatial_vote << ','
          << state.image_handover_progress_vote << ','
          << state.image_handover_age_vote << ','
          << state.image_handover_vote_count << ','
          << state.image_track_age_frames << ','
          << state.image_track_progress << ','
          << state.image_normalized_step_px << ','
          << state.detected_slot_index << ',' << state.measurement_slot_index << ','
          << state.candidate_association_slot_index << ','
          << (state.candidate_available ? state.candidate.center.x : 0.0F) << ','
          << (state.candidate_available ? state.candidate.center.y : 0.0F) << ',' << raw_phase << ','
          << state.phase_rad << ',' << state.future_phase_rad << ','
          << state.prediction_lead_s << ',' << state.angular_speed_rad_s << ','
          << state.angular_acceleration_rad_s2 << ',' << state.direction << ','
          << mode_name(state.mode) << ',' << state.reprojection_error_px << ','
          << state.missed_frames << ',' << state.future_target.valid << ','
          << state.future_slot_index << ','
          << state.future_target.id << ',' << state.future_target.center.x << ','
          << state.future_target.center.y << ','
          << state.raw_future_target.valid << ',' << state.raw_future_target.id << ','
          << state.raw_future_target.center.x << ','
          << state.raw_future_target.center.y << ','
          << state.future_target_filter_used << ','
          << state.future_target_filter_reset << ','
          << state.future_target_filter_innovation_px << ','
          << state.future_target_filter_velocity_px_s.x << ','
          << state.future_target_filter_velocity_px_s.y << ','
          << prediction_render.draw << ',' << prediction_render.high_overlap << ','
          << prediction_render.center_distance_px << ','
          << prediction_render.quad_iou << ','
          << state.future_model_slots[0].center.x << ','
          << state.future_model_slots[0].center.y << ',' << projected_area(state.future_model_slots[0]) << ','
          << state.future_model_slots[1].center.x << ','
          << state.future_model_slots[1].center.y << ',' << projected_area(state.future_model_slots[1]) << ','
          << state.future_model_slots[2].center.x << ','
          << state.future_model_slots[2].center.y << ',' << projected_area(state.future_model_slots[2]) << ','
          << source_name(state.slots[0].source) << ','
          << state.slots[0].center.x << ',' << state.slots[0].center.y << ','
          << source_name(state.slots[1].source) << ',' << state.slots[1].center.x << ','
          << state.slots[1].center.y << ',' << source_name(state.slots[2].source) << ','
          << state.slots[2].center.x << ',' << state.slots[2].center.y << ','
          << track_state_name(state.track_state) << ',' << state.track_confirmed << ','
          << state.phase_gate_passed << ',' << state.timestamp_gap_reset << ','
          << state.track_confirm_hits << ',' << state.time_since_measurement_s << ','
          << state.phase_variance_rad2 << ',' << state.speed_variance_rad2_s2 << ','
          << state.phase_innovation_rad << ',' << state.phase_innovation_variance << ','
          << state.phase_nis << ',' << state.association_cost << ','
          << state.track_confidence << ',' << state.physical_speed_valid
          << '\n';
      writer.write(frame);
      if (show_gui) {
        cv::imshow("Outpost three-armor solver", frame);
        const int key = cv::waitKey(1) & 0xff;
        if (key == 27 || key == 'q') break;
        if (key == ' ') while ((cv::waitKey(0) & 0xff) != ' ');
      }
      ++frame_number;
    }
    if (show_gui) cv::destroyAllWindows();
    std::cout << "Processed " << frame_number << " frames; measurements used=" << measured_frames
              << "\nVideo: " << output_path << "\nCSV: " << csv_path << '\n';
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 2;
  }
  return 0;
}
