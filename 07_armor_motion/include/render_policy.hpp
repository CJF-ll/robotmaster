#pragma once

#include "config.hpp"
#include "types.hpp"

struct PredictionRenderState {
  bool high_overlap = false;
};

struct PredictionRenderDecision {
  bool draw = false;
  bool high_overlap = false;
  double center_distance_px = -1.0;
  double quad_iou = 0.0;
};

double convex_quad_iou(const std::array<cv::Point2f, 4>& first,
                       const std::array<cv::Point2f, 4>& second);

PredictionRenderDecision evaluate_prediction_render(
    const SolverOutput& solver_output, const Config& config,
    PredictionRenderState* state);
