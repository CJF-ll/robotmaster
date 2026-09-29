#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

int main(int argc, char* argv[]) {
  const std::string output = argc > 1 ? argv[1] : "data/test_video.mp4";
  std::filesystem::create_directories(std::filesystem::path(output).parent_path());

  constexpr int width = 640;
  constexpr int height = 360;
  constexpr double fps = 30.0;
  constexpr int frame_count = 240;
  cv::VideoWriter writer(output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                         cv::Size(width, height));
  if (!writer.isOpened()) {
    std::cerr << "Error: cannot create video: " << output << '\n';
    return 1;
  }

  cv::RNG random(20260929);
  for (int frame_id = 0; frame_id < frame_count; ++frame_id) {
    const int brightness = 35 + frame_id / 10;
    cv::Mat frame(height, width, CV_8UC3,
                  cv::Scalar(brightness, brightness, brightness));

    for (int i = 0; i < 10; ++i) {
      const cv::Point noise(random.uniform(0, width), random.uniform(0, height));
      cv::circle(frame, noise, random.uniform(2, 5), cv::Scalar(220, 80, 30), -1);
    }

    const bool occluded = frame_id >= 105 && frame_id <= 109;
    if (!occluded) {
      const float t = static_cast<float>(frame_id);
      const cv::Point center(cvRound(70.0F + 1.9F * t),
                             cvRound(180.0F + 75.0F * std::sin(t * 0.045F)));
      cv::circle(frame, center, 22, cv::Scalar(255, 80, 20), -1, cv::LINE_AA);
    }
    writer.write(frame);
  }

  std::cout << "Generated " << frame_count << " frames: " << output << '\n';
  return 0;
}
