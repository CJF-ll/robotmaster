#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>

#include "armor_detector.hpp"
#include "config.hpp"
#include "rigid_model.hpp"

namespace {

class FramePreprocessor {
 public:
  FramePreprocessor(const Config& config, cv::Size input_size)
      : enabled_(config.camera_enabled) {
    if (!enabled_) return;
    const double sx = static_cast<double>(input_size.width) /
                      config.camera_reference_width;
    const double sy = static_cast<double>(input_size.height) /
                      config.camera_reference_height;
    camera_matrix_ = (cv::Mat_<double>(3, 3) <<
        config.fx * sx, 0, config.cx * sx,
        0, config.fy * sy, config.cy * sy,
        0, 0, 1);
    distortion_ = (cv::Mat_<double>(1, 5) <<
        config.k1, config.k2, config.p1, config.p2, config.k3);
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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: " << argv[0]
              << " <calibration.mp4> <model.yaml> [config.yaml]\n";
    return 1;
  }
  const std::string video_path = argv[1];
  const std::string model_path = argv[2];
  const std::string config_path = argc == 4 ? argv[3] : "config/video.yaml";

  try {
    const Config config = Config::load(config_path);
    cv::VideoCapture capture(video_path);
    if (!capture.isOpened())
      throw std::runtime_error("cannot open calibration video: " + video_path);
    const cv::Size image_size(
        static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT)));
    if (image_size.width <= 0 || image_size.height <= 0)
      throw std::runtime_error("calibration video has an invalid image size");

    FramePreprocessor preprocessor(config, image_size);
    ArmorDetector detector(config);
    std::vector<ArmorObservation> detected_samples;
    cv::Mat raw;
    int frame_count = 0;
    while (capture.read(raw)) {
      const cv::Mat frame = preprocessor.process(raw);
      const DetectionFrame detections = detector.detect(frame);
      if (frame_count % config.geometry_calibration_stride == 0 &&
          !detections.armors.empty()) {
        detected_samples.push_back(detections.armors.front());
      }
      ++frame_count;
    }
    if (detected_samples.size() <
        static_cast<std::size_t>(config.min_geometry_samples)) {
      throw std::runtime_error(
          "calibration video has too few sampled armor observations");
    }

    CalibrationProfile profile;
    profile.image_size = image_size;
    profile.undistorted = config.camera_enabled;
    profile.camera_reference_size = {
        config.camera_reference_width, config.camera_reference_height};
    profile.camera_matrix = {
        config.fx, 0.0, config.cx,
        0.0, config.fy, config.cy,
        0.0, 0.0, 1.0};
    profile.distortion = {
        config.k1, config.k2, config.p1, config.p2, config.k3};
    profile.sample_count = static_cast<int>(detected_samples.size());
    profile.geometry = calibrate_affine_geometry(
        detected_samples, config, image_size);
    profile.phase_quad_model = calibrate_phase_quad_model(
        detected_samples, profile.geometry, config);
    if (!profile.phase_quad_model.valid())
      throw std::runtime_error(
          "phase-shape model has insufficient phase coverage");

    const std::filesystem::path parent =
        std::filesystem::path(model_path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    save_calibration_profile(model_path, profile);
    std::cout << "Calibration profile written: " << model_path
              << "\nframes=" << frame_count
              << ", sampled_observations=" << profile.sample_count
              << "\ngeometry_rms=" << profile.geometry.calibration_rms_px
              << " px, covered_bins="
              << profile.phase_quad_model.covered_bins() << '/'
              << profile.phase_quad_model.bin_count() << '\n';
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 2;
  }
  return 0;
}
