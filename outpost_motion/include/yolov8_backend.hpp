#pragma once
#include "detection_backend.hpp"
#include "yolov8_decoder.hpp"
#include <openvino/openvino.hpp>
#include <deque>
namespace outpost {
// Optional experimental backend. Does not synthesize visible lamps from predictions.
class YoloV8Backend final:public DetectionBackend {
 public:
  YoloV8Backend(const std::string& model_path,double score,double nms,int inference_threads=0);
  ~YoloV8Backend()override;
  YoloV8Backend(const YoloV8Backend&)=delete;
  YoloV8Backend& operator=(const YoloV8Backend&)=delete;
  struct AsyncResult {
    long frame=-1;double time=0,latency_ms=0;
    std::vector<NeuralCandidate> candidates;
  };
  // Experimental bounded request pool; caller consumes strictly in submit order.
  // Each slot owns its OpenVINO input tensor until wait() has completed.
  void configure_async(size_t capacity);
  void submit(long frame,double time,const cv::Mat& image);
  AsyncResult collect();
  size_t pending()const{return pending_.size();}
  DetectionFrame detect(const cv::Mat& image)override;
  // Candidate-only path avoids building an unused DetectionFrame in fusion.
  void infer_candidates(const cv::Mat& image);
  const DetectionTiming& timing()const{return timing_;}
  const std::vector<NeuralCandidate>& candidates()const{return candidates_;}
 private:
  ov::Core core_;ov::CompiledModel model_;ov::InferRequest request_;
  double score_,nms_;std::vector<NeuralCandidate> candidates_;
  cv::Mat input_;
  DetectionTiming timing_;
  struct AsyncSlot {
    ov::InferRequest request;cv::Mat input;cv::Size image_size,resized;
    long frame=-1;double time=0;DetectionClock::time_point begin;
  };
  std::vector<AsyncSlot> slots_;
  std::deque<size_t> pending_,free_;
  long last_frame_=-1;double last_time_=-1;bool failed_=false;
};
}
