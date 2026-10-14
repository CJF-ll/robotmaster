#pragma once
#include "detector.hpp"
#include "tracker.hpp"
#include "visibility.hpp"
#include "target_slots.hpp"
#include "rigid_estimator.hpp"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <optional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <exception>

namespace outpost {
struct GeometryConfig {
  std::string mode="image_2d";
  bool undistort = false, allow_uniform_resize = false;
  int calibration_width = 0, calibration_height = 0;
  std::vector<double> camera_matrix, distortion, slot_height_offsets_m;
  // Rotation from target spin-axis coordinates (Z up) into camera coordinates.
  // Unknown is empty; never infer identity from a fixed-camera video.
  std::vector<double> camera_from_target_rotation;
  std::optional<double> armor_width_m,armor_height_m,rotation_radius_m,armor_pitch_deg;
  // Board dimensions above are NOT lightbar endpoint dimensions.
  std::optional<double> lightbar_length_m,lightbar_center_distance_m;
};
// Same full-size pixel plane is used by detection, tracking and rendering.
class ImagePreprocessor {
 public:
  ImagePreprocessor() = default;
  ImagePreprocessor(const GeometryConfig& geometry, cv::Size size);
  cv::Mat process(const cv::Mat& image) const;
  const cv::Mat& camera_matrix() const { return camera_; }
  bool rectified() const { return !map_x_.empty(); }
 private:
  cv::Size size_;
  cv::Mat camera_, map_x_, map_y_;
};
struct Config {
  DetectorConfig detector;
  TrackerConfig tracker;
  VisibilityConfig visibility;
  SlotConfig slots;
  GeometryConfig geometry;
  RigidEstimatorConfig rigid;
  // Effective consumers only: shared detection plus the selected estimator.
  // Inactive fields remain validated and are retained in source_text.
  std::string estimation_signature;
  std::string source_text; // Exact bytes parsed into this accepted snapshot.
  // Effective normalized K/D only when rectifying; raw pixels ignore calibration.
  std::string preprocessing_signature;
  std::string detector_runtime_signature;
  std::string input;
  std::string timestamp_file;
  int prefetch_frames=0;
  double prediction_seconds = 0.10;
  double coast_seconds = 0.30;
  bool display = true;
  static Config load(const std::string& path);
  static Config parse(const std::string& text);
};
// Uses effective intrinsics for the input size; rectified pixels use zero D.
RigidEstimatorConfig calibrated_config(const Config&,cv::Size input_size);
enum class ReloadResult { Unchanged, LiveUpdated, ResetRequired, Rejected };
class ConfigRuntime {
 public:
  explicit ConfigRuntime(const std::string& path);
  ReloadResult poll(std::string& message);
  ReloadResult apply_text(const std::string& text,std::string& message);
  const Config& current() const { return current_; }
  std::uint64_t update_attempt() const { return update_attempt_; }
  void bind_input_size(cv::Size size);
  const ImagePreprocessor& preprocessor() const { return preprocessor_; }
 private:
  std::string path_,last_text_;
  bool has_last_text_=false,last_rejected_=false;
  std::uint64_t update_attempt_=0;
  std::string last_rejection_;
  Config current_;
  cv::Size input_size_;
  ImagePreprocessor preprocessor_;
};
struct Frame {
  cv::Mat image;
  long index = 0;
  double time_seconds = 0;
};
// Explicit per-decoded-frame timestamp sidecar, or deterministic CFR time.
// A bounded decoder producer can overlap downstream inference; never drops frames.
class VideoSource {
 public:
  explicit VideoSource(const std::string& path,const std::string& timestamps="",int prefetch=0);
  explicit VideoSource(const Config& c):VideoSource(c.input,c.timestamp_file,c.prefetch_frames){}
  ~VideoSource();
  VideoSource(const VideoSource&)=delete;
  VideoSource& operator=(const VideoSource&)=delete;
  bool next(Frame& frame);
  const char* time_basis()const{return stamps_.empty()?"frame_index/nominal_fps":"timestamp_sidecar_seconds";}
  size_t queue_high_water()const;
  // -1: no exact future timestamp; -2: beyond sidecar EOF. No nearest-frame scoring.
  long forecast_frame(long frame,double horizon)const;
  double frame_interval(long frame)const;
  double fps() const { return fps_; }
  // Container/backend declaration, not independently verified ground truth.
  // -1 means unavailable or not an integral frame count.
  long reported_frame_count() const { return reported_frames_; }
 private:
  cv::VideoCapture capture_;
  double fps_;
  long index_ = 0;
  long reported_frames_ = -1;
  bool ended_ = false;
  bool decode(Frame&);
  void produce();
  std::vector<double> stamps_;
  int capacity_=0;
  std::thread worker_;
  mutable std::mutex mutex_;
  std::condition_variable ready_,space_;
  std::deque<Frame> queue_;
  bool stopped_=false,done_=false;
  size_t high_water_=0;
  std::exception_ptr error_;
};
enum class Availability { Unavailable, Available };
struct RuntimeTelemetry {
  Availability pose = Availability::Unavailable;
  Availability bullet_speed = Availability::Unavailable;
  double timestamp_seconds = 0;
};
}  // namespace outpost
