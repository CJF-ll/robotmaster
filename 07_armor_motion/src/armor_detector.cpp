#include "armor_detector.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace {
double angle_difference(double a, double b) {
  double d = std::fmod(std::abs(a - b), 180.0);
  return std::min(d, 180.0 - d);
}

constexpr float kGeometryEpsilon = 1e-4F;

struct BarAxes {
  cv::Point2f center;
  cv::Point2f down;
  float length = 0.0F;
  float width = 0.0F;
};

float cross_product(const cv::Point2f& a, const cv::Point2f& b) {
  return a.x * b.y - a.y * b.x;
}

cv::Point2f normalized(const cv::Point2f& value) {
  const float magnitude = static_cast<float>(cv::norm(value));
  return magnitude > kGeometryEpsilon ? value * (1.0F / magnitude) : cv::Point2f{};
}

BarAxes bar_axes(const LightBar& light) {
  cv::Point2f vertices[4];
  light.rect.points(vertices);
  const cv::Point2f edge0 = vertices[1] - vertices[0];
  const cv::Point2f edge1 = vertices[2] - vertices[1];
  cv::Point2f down = cv::norm(edge0) >= cv::norm(edge1) ? edge0 : edge1;
  down = normalized(down);
  if (down.y < 0.0F || (std::abs(down.y) <= kGeometryEpsilon && down.x < 0.0F))
    down *= -1.0F;
  return {light.rect.center, down, light.length, light.width};
}

std::array<cv::Point2f, 4> expanded_bar_vertices(const BarAxes& bar,
                                                  const cv::Point2f& normal,
                                                  float half_length, float half_width) {
  return {bar.center - bar.down * half_length - normal * half_width,
          bar.center - bar.down * half_length + normal * half_width,
          bar.center + bar.down * half_length + normal * half_width,
          bar.center + bar.down * half_length - normal * half_width};
}

std::vector<cv::Point2f> as_contour(const std::array<cv::Point2f, 4>& corners) {
  return {corners.begin(), corners.end()};
}

bool contains_bars(const std::array<cv::Point2f, 4>& corners,
                   const std::array<cv::Point2f, 4>& left_vertices,
                   const std::array<cv::Point2f, 4>& right_vertices) {
  const std::vector<cv::Point2f> contour = as_contour(corners);
  for (const auto& point : left_vertices)
    if (cv::pointPolygonTest(contour, point, true) < -1.0) return false;
  for (const auto& point : right_vertices)
    if (cv::pointPolygonTest(contour, point, true) < -1.0) return false;
  return true;
}

bool opposite_edge_ratio_valid(double first, double second, const Config& config) {
  if (first <= kGeometryEpsilon || second <= kGeometryEpsilon) return false;
  const double ratio = first / second;
  return ratio >= config.quad_min_opposite_edge_ratio &&
         ratio <= config.quad_max_opposite_edge_ratio;
}

bool valid_quad(const std::array<cv::Point2f, 4>& corners, const Config& config,
                cv::Size image_size, double reference_area) {
  const std::vector<cv::Point2f> contour = as_contour(corners);
  if (!cv::isContourConvex(contour)) return false;

  float cross_sign = 0.0F;
  for (int i = 0; i < 4; ++i) {
    const cv::Point2f first = corners[(i + 1) % 4] - corners[i];
    const cv::Point2f second = corners[(i + 2) % 4] - corners[(i + 1) % 4];
    const float cross = cross_product(first, second);
    if (std::abs(cross) <= kGeometryEpsilon) return false;
    if (cross_sign == 0.0F) cross_sign = cross;
    else if (cross * cross_sign <= 0.0F) return false;
  }

  const double area = std::abs(cv::contourArea(contour));
  if (area < config.quad_min_area_ratio * reference_area) return false;

  const double top = cv::norm(corners[1] - corners[0]);
  const double right = cv::norm(corners[2] - corners[1]);
  const double bottom = cv::norm(corners[2] - corners[3]);
  const double left = cv::norm(corners[3] - corners[0]);
  if (!opposite_edge_ratio_valid(top, bottom, config) ||
      !opposite_edge_ratio_valid(left, right, config))
    return false;

  constexpr float kImageTolerancePx = 2.0F;
  for (const auto& point : corners) {
    if (point.x < -kImageTolerancePx || point.y < -kImageTolerancePx ||
        point.x > image_size.width - 1.0F + kImageTolerancePx ||
        point.y > image_size.height - 1.0F + kImageTolerancePx)
      return false;
  }
  return true;
}

bool intersect_projection_support(const cv::Point2f& base, const cv::Point2f& direction,
                                  const cv::Point2f& projection_axis, float projection,
                                  cv::Point2f* intersection) {
  const float denominator = direction.dot(projection_axis);
  if (std::abs(denominator) <= kGeometryEpsilon) return false;
  const float distance = (projection - base.dot(projection_axis)) / denominator;
  *intersection = base + direction * distance;
  return true;
}

bool repair_with_support_lines(
    const BarAxes& left, const BarAxes& right, const cv::Point2f& left_normal,
    const cv::Point2f& right_normal, float left_half_width, float right_half_width,
    const std::array<cv::Point2f, 4>& left_vertices,
    const std::array<cv::Point2f, 4>& right_vertices,
    std::array<cv::Point2f, 4>* corners) {
  const cv::Point2f projection_axis = normalized(left.down + right.down);
  if (cv::norm(projection_axis) <= kGeometryEpsilon) return false;

  float top_projection = left_vertices[0].dot(projection_axis);
  float bottom_projection = top_projection;
  for (const auto& point : left_vertices) {
    top_projection = std::min(top_projection, point.dot(projection_axis));
    bottom_projection = std::max(bottom_projection, point.dot(projection_axis));
  }
  for (const auto& point : right_vertices) {
    top_projection = std::min(top_projection, point.dot(projection_axis));
    bottom_projection = std::max(bottom_projection, point.dot(projection_axis));
  }

  const cv::Point2f left_base = left.center + left_normal * left_half_width;
  const cv::Point2f right_base = right.center + right_normal * right_half_width;
  return intersect_projection_support(left_base, left.down, projection_axis, top_projection,
                                      &(*corners)[0]) &&
         intersect_projection_support(right_base, right.down, projection_axis, top_projection,
                                      &(*corners)[1]) &&
         intersect_projection_support(right_base, right.down, projection_axis, bottom_projection,
                                      &(*corners)[2]) &&
         intersect_projection_support(left_base, left.down, projection_axis, bottom_projection,
                                      &(*corners)[3]);
}
}  // namespace

std::optional<ArmorObservation> build_armor_observation(
    const LightBar& first, const LightBar& second, const Config& config, cv::Size image_size) {
  const LightBar* left_light = &first;
  const LightBar* right_light = &second;
  if (left_light->rect.center.x > right_light->rect.center.x)
    std::swap(left_light, right_light);

  const BarAxes left = bar_axes(*left_light);
  const BarAxes right = bar_axes(*right_light);
  const cv::Point2f center_delta = right.center - left.center;
  const float center_distance = static_cast<float>(cv::norm(center_delta));
  if (center_distance <= kGeometryEpsilon || cv::norm(left.down) <= kGeometryEpsilon ||
      cv::norm(right.down) <= kGeometryEpsilon)
    return std::nullopt;
  const cv::Point2f baseline = center_delta * (1.0F / center_distance);

  cv::Point2f left_normal(-left.down.y, left.down.x);
  cv::Point2f right_normal(-right.down.y, right.down.x);
  if (left_normal.dot(baseline * -1.0F) < 0.0F) left_normal *= -1.0F;
  if (right_normal.dot(baseline) < 0.0F) right_normal *= -1.0F;

  const float left_half_length = 0.5F * left.length + static_cast<float>(std::max(
      config.quad_end_padding_min_px, config.quad_end_padding_ratio * left.length));
  const float right_half_length = 0.5F * right.length + static_cast<float>(std::max(
      config.quad_end_padding_min_px, config.quad_end_padding_ratio * right.length));
  const float left_half_width = 0.5F * left.width + static_cast<float>(std::max(
      config.quad_side_padding_min_px, config.quad_side_padding_ratio * left.width));
  const float right_half_width = 0.5F * right.width + static_cast<float>(std::max(
      config.quad_side_padding_min_px, config.quad_side_padding_ratio * right.width));

  const auto left_vertices = expanded_bar_vertices(
      left, left_normal, left_half_length, left_half_width);
  const auto right_vertices = expanded_bar_vertices(
      right, right_normal, right_half_length, right_half_width);
  std::array<cv::Point2f, 4> corners{
      left.center - left.down * left_half_length + left_normal * left_half_width,
      right.center - right.down * right_half_length + right_normal * right_half_width,
      right.center + right.down * right_half_length + right_normal * right_half_width,
      left.center + left.down * left_half_length + left_normal * left_half_width};

  const double mean_length = 0.5 * (left.length + right.length);
  const double reference_area = center_distance * mean_length;
  if (!valid_quad(corners, config, image_size, reference_area) ||
      !contains_bars(corners, left_vertices, right_vertices)) {
    if (!repair_with_support_lines(left, right, left_normal, right_normal,
                                   left_half_width, right_half_width,
                                   left_vertices, right_vertices, &corners) ||
        !valid_quad(corners, config, image_size, reference_area) ||
        !contains_bars(corners, left_vertices, right_vertices))
      return std::nullopt;
  }

  ArmorObservation armor;
  armor.corners = corners;
  armor.center = 0.5F * (left.center + right.center);
  const float top = static_cast<float>(cv::norm(corners[1] - corners[0]));
  const float right_edge = static_cast<float>(cv::norm(corners[2] - corners[1]));
  const float bottom = static_cast<float>(cv::norm(corners[2] - corners[3]));
  const float left_edge = static_cast<float>(cv::norm(corners[3] - corners[0]));
  armor.size = {0.5F * (top + bottom), 0.5F * (left_edge + right_edge)};

  const float y_ratio = std::abs(center_delta.y) / static_cast<float>(mean_length);
  const float separation = center_distance / static_cast<float>(mean_length);
  const float length_ratio = std::max(left.length, right.length) /
                             std::max(1.0F, std::min(left.length, right.length));
  const float symmetry = 1.0F / length_ratio;
  const float alignment = 1.0F - std::min(1.0F, y_ratio);
  const float nominal_spacing =
      1.0F - std::min(1.0F, std::abs(separation - 2.0F) / 2.0F);
  armor.score = 0.45F * symmetry + 0.35F * alignment + 0.20F * nominal_spacing;
  return armor;
}

DetectionFrame ArmorDetector::detect(const cv::Mat& bgr, cv::Mat* mask_out) const {
  DetectionFrame output;
  if (bgr.empty() || bgr.type() != CV_8UC3) return output;

  const int x0 = cvRound(config_.roi_x_min * bgr.cols);
  const int y0 = cvRound(config_.roi_y_min * bgr.rows);
  const int x1 = cvRound(config_.roi_x_max * bgr.cols);
  const int y1 = cvRound(config_.roi_y_max * bgr.rows);
  const cv::Rect roi(x0, y0, std::max(1, x1 - x0), std::max(1, y1 - y0));
  const cv::Mat cropped = bgr(roi);

  std::vector<cv::Mat> channels;
  cv::split(cropped, channels);
  cv::Mat r16, g16, b16, red_green, red_blue, red_mask, rg_mask, rb_mask, mask;
  channels[2].convertTo(r16, CV_16S);
  channels[1].convertTo(g16, CV_16S);
  channels[0].convertTo(b16, CV_16S);
  cv::subtract(r16, g16, red_green);
  cv::subtract(r16, b16, red_blue);
  cv::compare(channels[2], config_.red_min, red_mask, cv::CMP_GT);
  cv::compare(red_green, config_.red_green_diff_min, rg_mask, cv::CMP_GT);
  cv::compare(red_blue, config_.red_blue_diff_min, rb_mask, cv::CMP_GT);
  cv::bitwise_and(red_mask, rg_mask, mask);
  cv::bitwise_and(mask, rb_mask, mask);
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                   cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));
  if (mask_out) {
    mask_out->create(bgr.size(), CV_8UC1);
    mask_out->setTo(0);
    mask.copyTo((*mask_out)(roi));
  }

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  for (auto contour : contours) {
    const double area = cv::contourArea(contour);
    if (area < config_.min_light_area || area > config_.max_light_area || contour.size() < 4)
      continue;
    for (auto& point : contour) point += roi.tl();
    cv::RotatedRect rect = cv::minAreaRect(contour);
    cv::Point2f vertices[4];
    rect.points(vertices);
    cv::Point2f e0 = vertices[1] - vertices[0];
    cv::Point2f e1 = vertices[2] - vertices[1];
    cv::Point2f long_edge = cv::norm(e0) >= cv::norm(e1) ? e0 : e1;
    const float length = static_cast<float>(cv::norm(long_edge));
    const float width = static_cast<float>(std::min(cv::norm(e0), cv::norm(e1)));
    if (width < 0.8F || length < config_.min_light_length ||
        length > config_.max_light_length || length / width < config_.min_light_ratio)
      continue;
    float angle = static_cast<float>(std::atan2(long_edge.y, long_edge.x) * 180.0 / CV_PI);
    if (angle < 0) angle += 180.0F;
    if (std::abs(90.0F - angle) > config_.max_light_tilt_deg) continue;
    output.lights.push_back({rect, length, width, angle, area});
  }
  std::sort(output.lights.begin(), output.lights.end(), [](const auto& a, const auto& b) {
    return a.rect.center.x < b.rect.center.x;
  });

  for (std::size_t i = 0; i < output.lights.size(); ++i) {
    for (std::size_t j = i + 1; j < output.lights.size(); ++j) {
      const auto& left = output.lights[i];
      const auto& right = output.lights[j];
      const float mean_length = 0.5F * (left.length + right.length);
      const float dx = right.rect.center.x - left.rect.center.x;
      const float dy = right.rect.center.y - left.rect.center.y;
      const float separation = std::hypot(dx, dy) / mean_length;
      const float y_ratio = std::abs(dy) / mean_length;
      const float length_ratio = std::max(left.length, right.length) /
                                 std::max(1.0F, std::min(left.length, right.length));
      if (separation < config_.min_pair_distance_ratio ||
          separation > config_.max_pair_distance_ratio ||
          y_ratio > config_.max_pair_y_diff_ratio ||
          length_ratio > config_.max_pair_length_ratio ||
          angle_difference(left.long_axis_angle_deg, right.long_axis_angle_deg) >
              config_.max_pair_angle_diff_deg)
        continue;

      const auto armor = build_armor_observation(left, right, config_, bgr.size());
      if (armor) output.armors.push_back(*armor);
    }
  }
  std::sort(output.armors.begin(), output.armors.end(), [](const auto& a, const auto& b) {
    return a.score > b.score;
  });
  std::vector<ArmorObservation> non_overlapping;
  for (const auto& candidate : output.armors) {
    bool duplicate = false;
    for (const auto& kept : non_overlapping) {
      if (cv::norm(candidate.center - kept.center) < 0.5F *
          std::min(candidate.size.width, kept.size.width)) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) non_overlapping.push_back(candidate);
  }
  output.armors = std::move(non_overlapping);
  return output;
}
