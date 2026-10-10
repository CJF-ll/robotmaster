#include "config.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <opencv2/core.hpp>

Config Config::load(const std::string& path) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) throw std::runtime_error("cannot open config: " + path);
  Config c;
#define READ(name) if (!fs[#name].empty()) fs[#name] >> c.name
  READ(solver_mode);
  int camera_enabled_int = 0;
  if (!fs["camera_enabled"].empty()) fs["camera_enabled"] >> camera_enabled_int;
  if (camera_enabled_int != 0 && camera_enabled_int != 1)
    throw std::runtime_error("camera_enabled must be 0 or 1");
  c.camera_enabled = camera_enabled_int != 0;
  READ(camera_reference_width); READ(camera_reference_height);
  READ(fx); READ(fy); READ(cx); READ(cy);
  READ(k1); READ(k2); READ(p1); READ(p2); READ(k3);
  READ(armor_width_m); READ(armor_height_m); READ(rotation_radius_m);
  READ(outpost_pitch_deg);
  READ(roi_x_min); READ(roi_y_min); READ(roi_x_max); READ(roi_y_max);
  READ(red_min); READ(red_green_diff_min); READ(red_blue_diff_min);
  READ(min_light_area); READ(max_light_area); READ(min_light_length);
  READ(max_light_length); READ(min_light_ratio); READ(max_light_tilt_deg);
  READ(min_pair_distance_ratio); READ(max_pair_distance_ratio);
  READ(max_pair_y_diff_ratio); READ(max_pair_length_ratio); READ(max_pair_angle_diff_deg);
  READ(quad_end_padding_ratio); READ(quad_end_padding_min_px);
  READ(quad_side_padding_ratio); READ(quad_side_padding_min_px);
  READ(quad_min_area_ratio); READ(quad_min_opposite_edge_ratio);
  READ(quad_max_opposite_edge_ratio);
  READ(geometry_calibration_stride); READ(ellipse_ransac_iterations);
  READ(ellipse_inlier_threshold); READ(min_geometry_samples);
  READ(shape_phase_bins); READ(shape_min_samples_per_bin); READ(shape_min_covered_bins);
  READ(shape_smoothing_radius_bins); READ(model_offset_gain);
  READ(model_offset_max_step_px); READ(model_offset_max_magnitude_px);
  READ(max_observation_distance_px); READ(acceleration_gain);
  READ(max_abs_speed_deg_s);
  READ(track_confirm_hits); READ(track_confirm_max_gap_s);
  READ(track_prediction_visible_s); READ(track_lost_timeout_s);
  READ(tracker_max_dt_s); READ(phase_kf_measurement_std_deg);
  READ(phase_kf_initial_phase_std_deg); READ(phase_kf_initial_speed_std_deg_s);
  READ(phase_kf_accel_noise_std_deg_s2); READ(phase_nis_gate);
  READ(association_pixel_sigma_px); READ(association_score_weight);
  READ(association_slot_switch_penalty); READ(covariance_floor);
  READ(physical_speed_handover_stride); READ(physical_speed_window);
  READ(physical_direction_window);
  READ(physical_speed_min_interval_s); READ(physical_speed_max_interval_s);
  READ(physical_speed_acceleration_gain);
  int angular_speed_snap_enabled_int = 0;
  if (!fs["angular_speed_snap_enabled"].empty())
    fs["angular_speed_snap_enabled"] >> angular_speed_snap_enabled_int;
  if (angular_speed_snap_enabled_int != 0 && angular_speed_snap_enabled_int != 1)
    throw std::runtime_error("angular_speed_snap_enabled must be 0 or 1");
  c.angular_speed_snap_enabled = angular_speed_snap_enabled_int != 0;
  READ(angular_speed_snap_rad_s); READ(angular_speed_snap_tolerance_rad_s);
  READ(prediction_lead_frames); READ(prediction_lead_s);
  READ(prediction_max_acceleration_deg_s2);
  int future_target_filter_enabled_int =
      c.future_target_filter_enabled ? 1 : 0;
  if (!fs["future_target_filter_enabled"].empty())
    fs["future_target_filter_enabled"] >> future_target_filter_enabled_int;
  if (future_target_filter_enabled_int != 0 &&
      future_target_filter_enabled_int != 1)
    throw std::runtime_error("future_target_filter_enabled must be 0 or 1");
  c.future_target_filter_enabled = future_target_filter_enabled_int != 0;
  READ(future_target_filter_position_gain);
  READ(future_target_filter_velocity_gain);
  READ(future_target_filter_shape_gain);
  READ(future_target_filter_reset_distance_px);
  READ(future_target_filter_max_dt_s);
  READ(future_target_filter_max_speed_px_s);
  READ(prediction_overlap_enter_px);
  READ(prediction_overlap_exit_px);
  READ(prediction_overlap_enter_iou);
  READ(prediction_overlap_exit_iou);
  READ(prediction_overlap_alpha);
  READ(image_prediction_window_frames); READ(image_prediction_min_samples);
  READ(image_prediction_velocity_gain); READ(image_prediction_history_timeout_s);
  READ(image_prediction_handover_hold_s);
  READ(image_prediction_max_step_px);
  READ(image_prediction_jump_min_px);
  READ(image_prediction_jump_max_px);
  READ(image_prediction_jump_direction_cos_max);
  READ(image_prediction_transition_y_tolerance_px);
  READ(image_prediction_wrap_margin_px);
  int image_prediction_handover_prior_enabled_int = 1;
  if (!fs["image_prediction_handover_prior_enabled"].empty())
    fs["image_prediction_handover_prior_enabled"] >>
        image_prediction_handover_prior_enabled_int;
  if (image_prediction_handover_prior_enabled_int != 0 &&
      image_prediction_handover_prior_enabled_int != 1)
    throw std::runtime_error("image_prediction_handover_prior_enabled must be 0 or 1");
  c.image_prediction_handover_prior_enabled = image_prediction_handover_prior_enabled_int != 0;
  READ(image_prediction_handover_vote_threshold);
  READ(stationary_speed_deg_s); READ(uniform_acceleration_deg_s2); READ(mode_hold_frames);
  READ(motion_difference_threshold); READ(motion_hold_frames);
#undef READ
  if (c.solver_mode != "affine" && c.solver_mode != "auto" && c.solver_mode != "pnp")
    throw std::runtime_error("solver_mode must be affine, auto, or pnp");
  const auto finite = [](double value) { return std::isfinite(value); };
  if (!(0 <= c.roi_x_min && c.roi_x_min < c.roi_x_max && c.roi_x_max <= 1 &&
        0 <= c.roi_y_min && c.roi_y_min < c.roi_y_max && c.roi_y_max <= 1))
    throw std::runtime_error("invalid normalized ROI");
  if (!finite(c.fx) || !finite(c.fy) || !finite(c.cx) || !finite(c.cy) ||
      !finite(c.k1) || !finite(c.k2) || !finite(c.p1) || !finite(c.p2) ||
      !finite(c.k3))
    throw std::runtime_error("camera parameters must be finite");
  if (c.camera_enabled &&
      (c.fx <= 0 || c.fy <= 0 || c.camera_reference_width <= 0 ||
       c.camera_reference_height <= 0))
    throw std::runtime_error("camera_enabled requires valid intrinsics and reference size");
  if (!finite(c.armor_width_m) || !finite(c.armor_height_m) ||
      !finite(c.rotation_radius_m) || !finite(c.outpost_pitch_deg) ||
      c.armor_width_m < 0 || c.armor_height_m < 0 || c.rotation_radius_m < 0 ||
      c.outpost_pitch_deg <= -90.0 || c.outpost_pitch_deg >= 90.0)
    throw std::runtime_error("invalid physical model parameters");
  if (c.solver_mode == "pnp" &&
      (c.armor_width_m <= 0 || c.armor_height_m <= 0 || c.rotation_radius_m <= 0))
    throw std::runtime_error("pnp requires positive physical dimensions");
  if (c.geometry_calibration_stride < 1 || c.ellipse_ransac_iterations < 1 ||
      !finite(c.ellipse_inlier_threshold) || c.ellipse_inlier_threshold <= 0 ||
      c.min_geometry_samples < 5)
    throw std::runtime_error("invalid geometry-calibration parameters");
  if (!finite(c.max_abs_speed_deg_s) || c.max_abs_speed_deg_s <= 0 ||
      !finite(c.angular_speed_snap_rad_s) || c.angular_speed_snap_rad_s < 0 ||
      !finite(c.angular_speed_snap_tolerance_rad_s) ||
      c.angular_speed_snap_tolerance_rad_s < 0 ||
      (c.angular_speed_snap_enabled && c.angular_speed_snap_rad_s <= 0) ||
      (c.angular_speed_snap_enabled &&
       c.angular_speed_snap_rad_s > c.max_abs_speed_deg_s * CV_PI / 180.0))
    throw std::runtime_error("invalid angular-speed constraints");
  if (c.track_confirm_hits < 1 ||
      !finite(c.track_confirm_max_gap_s) || c.track_confirm_max_gap_s <= 0 ||
      !finite(c.track_prediction_visible_s) || c.track_prediction_visible_s < 0 ||
      !finite(c.track_lost_timeout_s) ||
      c.track_lost_timeout_s < c.track_prediction_visible_s ||
      !finite(c.tracker_max_dt_s) || c.tracker_max_dt_s <= 0 ||
      c.track_confirm_max_gap_s > c.tracker_max_dt_s ||
      !finite(c.phase_kf_measurement_std_deg) || c.phase_kf_measurement_std_deg <= 0 ||
      !finite(c.phase_kf_initial_phase_std_deg) || c.phase_kf_initial_phase_std_deg <= 0 ||
      !finite(c.phase_kf_initial_speed_std_deg_s) || c.phase_kf_initial_speed_std_deg_s <= 0 ||
      !finite(c.phase_kf_accel_noise_std_deg_s2) || c.phase_kf_accel_noise_std_deg_s2 <= 0 ||
      !finite(c.phase_nis_gate) || c.phase_nis_gate <= 0 ||
      !finite(c.association_pixel_sigma_px) || c.association_pixel_sigma_px <= 0 ||
      !finite(c.association_score_weight) || c.association_score_weight < 0 ||
      !finite(c.association_slot_switch_penalty) || c.association_slot_switch_penalty < 0 ||
      !finite(c.covariance_floor) || c.covariance_floor <= 0)
    throw std::runtime_error("invalid tracking-filter parameters");
  if (c.physical_speed_handover_stride < 1 ||
      c.physical_speed_window < 1 || c.physical_speed_window > 31 ||
      c.physical_direction_window < 5 || c.physical_direction_window > 120 ||
      !finite(c.physical_speed_min_interval_s) ||
      c.physical_speed_min_interval_s <= 0 ||
      !finite(c.physical_speed_max_interval_s) ||
      c.physical_speed_max_interval_s <= c.physical_speed_min_interval_s ||
      !finite(c.physical_speed_acceleration_gain) ||
      c.physical_speed_acceleration_gain <= 0 ||
      c.physical_speed_acceleration_gain > 1)
    throw std::runtime_error("invalid physical-speed estimator parameters");
  if (c.quad_end_padding_ratio < 0 || c.quad_end_padding_min_px < 0 ||
      c.quad_side_padding_ratio < 0 || c.quad_side_padding_min_px < 0 ||
      c.quad_min_area_ratio <= 0 || c.quad_min_opposite_edge_ratio <= 0 ||
      c.quad_max_opposite_edge_ratio < c.quad_min_opposite_edge_ratio)
    throw std::runtime_error("invalid light-band quadrilateral parameters");
  if (c.shape_phase_bins < 12 || c.shape_min_samples_per_bin < 1 ||
      c.shape_min_covered_bins < 1 || c.shape_min_covered_bins > c.shape_phase_bins ||
      c.shape_smoothing_radius_bins < 0 ||
      c.shape_smoothing_radius_bins >= c.shape_phase_bins / 2)
    throw std::runtime_error("invalid phase-shape model parameters");
  if (c.model_offset_gain < 0 || c.model_offset_gain > 1 ||
      c.model_offset_max_step_px <= 0 || c.model_offset_max_magnitude_px <= 0)
    throw std::runtime_error("invalid model-offset filter parameters");
  if (c.prediction_lead_frames < 1 || !finite(c.prediction_lead_s) ||
      c.prediction_lead_s < 0 ||
      !finite(c.prediction_max_acceleration_deg_s2) ||
      c.prediction_max_acceleration_deg_s2 < 0 ||
      !finite(c.future_target_filter_position_gain) ||
      c.future_target_filter_position_gain <= 0 ||
      c.future_target_filter_position_gain > 1 ||
      !finite(c.future_target_filter_velocity_gain) ||
      c.future_target_filter_velocity_gain < 0 ||
      c.future_target_filter_velocity_gain > 1 ||
      !finite(c.future_target_filter_shape_gain) ||
      c.future_target_filter_shape_gain <= 0 ||
      c.future_target_filter_shape_gain > 1 ||
      !finite(c.future_target_filter_reset_distance_px) ||
      c.future_target_filter_reset_distance_px <= 0 ||
      !finite(c.future_target_filter_max_dt_s) ||
      c.future_target_filter_max_dt_s <= 0 ||
      !finite(c.future_target_filter_max_speed_px_s) ||
      c.future_target_filter_max_speed_px_s <= 0 ||
      !finite(c.prediction_overlap_enter_px) ||
      !finite(c.prediction_overlap_exit_px) ||
      c.prediction_overlap_enter_px < 0 ||
      c.prediction_overlap_exit_px < c.prediction_overlap_enter_px ||
      !finite(c.prediction_overlap_enter_iou) ||
      !finite(c.prediction_overlap_exit_iou) ||
      c.prediction_overlap_enter_iou < 0 ||
      c.prediction_overlap_enter_iou > 1 ||
      c.prediction_overlap_exit_iou < 0 ||
      c.prediction_overlap_exit_iou > c.prediction_overlap_enter_iou ||
      !finite(c.prediction_overlap_alpha) ||
      c.prediction_overlap_alpha <= 0 ||
      c.prediction_overlap_alpha > 1)
    throw std::runtime_error("invalid future-prediction parameters");
  if (c.image_prediction_window_frames < 3 || c.image_prediction_window_frames > 15 ||
      c.image_prediction_min_samples < 2 ||
      c.image_prediction_min_samples > c.image_prediction_window_frames ||
      !finite(c.image_prediction_velocity_gain) ||
      c.image_prediction_velocity_gain < 0 || c.image_prediction_velocity_gain > 1 ||
      !finite(c.image_prediction_history_timeout_s) ||
      c.image_prediction_history_timeout_s <= 0 ||
      !finite(c.image_prediction_handover_hold_s) ||
      c.image_prediction_handover_hold_s < c.prediction_lead_s ||
      c.image_prediction_handover_hold_s > c.track_lost_timeout_s ||
      !finite(c.image_prediction_max_step_px) ||
      c.image_prediction_max_step_px <= 0 ||
      !finite(c.image_prediction_jump_min_px) ||
      c.image_prediction_jump_min_px <= c.image_prediction_max_step_px ||
      !finite(c.image_prediction_jump_max_px) ||
      c.image_prediction_jump_max_px <= c.image_prediction_jump_min_px ||
      !finite(c.image_prediction_jump_direction_cos_max) ||
      c.image_prediction_jump_direction_cos_max < -1.0 ||
      c.image_prediction_jump_direction_cos_max > 0.0 ||
      !finite(c.image_prediction_transition_y_tolerance_px) ||
      c.image_prediction_transition_y_tolerance_px < 0 ||
      !finite(c.image_prediction_wrap_margin_px) ||
      c.image_prediction_wrap_margin_px < 0 ||
      c.image_prediction_handover_vote_threshold < 1 ||
      c.image_prediction_handover_vote_threshold > 3)
    throw std::runtime_error("invalid image-motion prediction parameters");
  return c;
}

void Config::resolve_prediction_horizon(double fps) {
  if (!std::isfinite(fps) || fps <= 0.0)
    throw std::runtime_error("prediction horizon requires a positive finite FPS");
  if (prediction_lead_s > 0.0) {
    const double requested_frames = prediction_lead_s * fps;
    if (!std::isfinite(requested_frames) ||
        requested_frames > std::numeric_limits<int>::max())
      throw std::runtime_error("prediction horizon is too large");
    prediction_lead_frames = std::max(
        1, static_cast<int>(std::lround(requested_frames)));
  }
  prediction_lead_s = prediction_lead_frames / fps;
}
