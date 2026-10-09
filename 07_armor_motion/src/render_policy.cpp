#include "render_policy.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

double convex_quad_iou(const std::array<cv::Point2f, 4>& first,
                       const std::array<cv::Point2f, 4>& second) {
  const std::vector<cv::Point2f> first_polygon(first.begin(), first.end());
  const std::vector<cv::Point2f> second_polygon(second.begin(), second.end());
  const double first_area = std::abs(cv::contourArea(first_polygon));
  const double second_area = std::abs(cv::contourArea(second_polygon));
  if (first_area <= 1e-6 || second_area <= 1e-6) return 0.0;
  std::vector<cv::Point2f> intersection;
  const double intersection_area = cv::intersectConvexConvex(
      first_polygon, second_polygon, intersection, true);
  const double union_area = first_area + second_area - intersection_area;
  return union_area > 1e-6
      ? std::clamp(intersection_area / union_area, 0.0, 1.0)
      : 0.0;
}

PredictionRenderDecision evaluate_prediction_render(
    const SolverOutput& solver_output, const Config& config,
    PredictionRenderState* state) {
  PredictionRenderDecision decision;
  decision.draw = solver_output.future_target.valid;
  if (!state) return decision;
  if (!decision.draw) {
    state->high_overlap = false;
    decision.high_overlap = false;
    return decision;
  }
  if (!solver_output.candidate_available) {
    decision.high_overlap = state->high_overlap;
    return decision;
  }

  decision.center_distance_px = cv::norm(
      solver_output.future_target.center - solver_output.candidate.center);
  decision.quad_iou = convex_quad_iou(
      solver_output.future_target.corners, solver_output.candidate.corners);
  if (state->high_overlap) {
    const bool separated =
        decision.center_distance_px >= config.prediction_overlap_exit_px &&
        decision.quad_iou <= config.prediction_overlap_exit_iou;
    if (separated) state->high_overlap = false;
  } else {
    const bool overlaps =
        decision.center_distance_px <= config.prediction_overlap_enter_px ||
        decision.quad_iou >= config.prediction_overlap_enter_iou;
    if (overlaps) state->high_overlap = true;
  }
  decision.high_overlap = state->high_overlap;
  return decision;
}
