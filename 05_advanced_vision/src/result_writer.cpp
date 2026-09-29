#include "result_writer.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace {

void draw_cross(cv::Mat& image, const cv::Point2f& point, const cv::Scalar& color) {
  const cv::Point center(cvRound(point.x), cvRound(point.y));
  cv::line(image, center + cv::Point(-8, 0), center + cv::Point(8, 0), color, 2);
  cv::line(image, center + cv::Point(0, -8), center + cv::Point(0, 8), color, 2);
}

std::string number_or_empty(bool valid, float value) {
  if (!valid) {
    return {};
  }
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(2) << value;
  return stream.str();
}

}  // namespace

void ResultWriter::run() {
  try {
    std::filesystem::create_directories(
        std::filesystem::path(config_.output_csv).parent_path());
    std::filesystem::create_directories(
        std::filesystem::path(config_.output_video).parent_path());

    std::ofstream csv(config_.output_csv);
    if (!csv) {
      throw std::runtime_error("cannot open output CSV: " + config_.output_csv);
    }
    csv << "frame_id,timestamp_us,detected,predicted,measure_x,measure_y,"
           "track_x,track_y,vx,vy\n";

    cv::VideoWriter video;
    OutputPacket packet;
    std::size_t written_frames = 0;
    while (input_.pop(packet)) {
      if (!video.isOpened()) {
        const double fps = 30.0;
        const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        video.open(config_.output_video, fourcc, fps, packet.frame.image.size(), true);
        if (!video.isOpened()) {
          throw std::runtime_error("cannot open output video: " + config_.output_video);
        }
      }

      cv::Mat visualization = packet.frame.image.clone();
      if (packet.observation.detected) {
        const cv::Point measured_center(cvRound(packet.observation.center.x),
                                        cvRound(packet.observation.center.y));
        cv::circle(visualization, measured_center,
                   cvRound(packet.observation.radius), cv::Scalar(0, 255, 255), 2);
      }
      if (packet.tracking.valid) {
        const cv::Scalar color = packet.tracking.predicted
                                     ? cv::Scalar(0, 0, 255)
                                     : cv::Scalar(0, 255, 0);
        draw_cross(visualization, packet.tracking.filtered_position, color);
        cv::putText(visualization,
                    packet.tracking.predicted ? "PREDICTED" : "TRACKED",
                    cv::Point(16, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
      } else {
        cv::putText(visualization, "NO TRACK", cv::Point(16, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(160, 160, 160), 2);
      }
      video.write(visualization);

      const bool measured = packet.observation.detected;
      const bool tracked = packet.tracking.valid;
      csv << packet.frame.sequence << ',' << packet.frame.timestamp_us << ','
          << (measured ? 1 : 0) << ','
          << (packet.tracking.predicted ? 1 : 0) << ','
          << number_or_empty(measured, packet.observation.center.x) << ','
          << number_or_empty(measured, packet.observation.center.y) << ','
          << number_or_empty(tracked, packet.tracking.filtered_position.x) << ','
          << number_or_empty(tracked, packet.tracking.filtered_position.y) << ','
          << number_or_empty(tracked, packet.tracking.velocity.x) << ','
          << number_or_empty(tracked, packet.tracking.velocity.y) << '\n';
      ++written_frames;
    }

    if (written_frames == 0) {
      throw std::runtime_error("input video contains no readable frames");
    }
    std::cout << "Wrote " << written_frames << " frames to:\n  "
              << config_.output_video << "\n  " << config_.output_csv << '\n';
  } catch (...) {
    errors_.capture_current_exception();
    input_.close();
  }
}
