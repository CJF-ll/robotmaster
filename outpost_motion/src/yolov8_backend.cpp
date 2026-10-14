#include "yolov8_backend.hpp"
#include "yolov8_input.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
namespace outpost {
YoloV8Backend::YoloV8Backend(const std::string& path,double score,double nms,int inference_threads):score_(score),nms_(nms){
  if(inference_threads<0||inference_threads>256)throw std::runtime_error("invalid inference thread count");
  if(!std::isfinite(score)||score<0||score>1||!std::isfinite(nms)||nms<0||nms>1)
    throw std::runtime_error("invalid neural score/NMS threshold");
  std::cerr<<"OpenVINO: reading model\n";
  auto graph=core_.read_model(path);
  if(graph->inputs().size()!=1||graph->outputs().size()!=1||
     graph->input().get_shape()!=ov::Shape{1,3,416,416}||graph->input().get_element_type()!=ov::element::f32)
    throw std::runtime_error("model must have f32 input [1,3,416,416]");
  const auto output=graph->output().get_shape();
  if(output.size()!=3||output[0]!=1||output[1]!=14||output[2]==0||graph->output().get_element_type()!=ov::element::f32)
    throw std::runtime_error("model must have f32 output [1,14,N]");
  std::cerr<<"OpenVINO: compiling CPU model\n";
  ov::AnyMap properties{ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)};
  if(inference_threads>0)properties.emplace(ov::inference_num_threads(inference_threads));
  model_=core_.compile_model(graph,"CPU",properties);
  std::cerr<<"YOLO effective inference threads="<<model_.get_property(ov::inference_num_threads)<<'\n';
  std::cerr<<"OpenVINO: CPU model ready\n";
  request_=model_.create_infer_request();
}
void YoloV8Backend::infer_candidates(const cv::Mat& image){
  if(!pending_.empty())throw std::runtime_error("cannot mix synchronous and pending asynchronous inference");
  timing_={};const auto begin=DetectionClock::now();
  candidates_.clear();
  auto tensor=request_.get_input_tensor();
  const auto resized=prepare_yolov8_input(image,input_,tensor.data<float>(),tensor.get_size());
  const auto prepared=DetectionClock::now();
  request_.infer();const auto inferred=DetectionClock::now();
  auto output=request_.get_output_tensor();const auto shape=output.get_shape();
  candidates_=decode_yolov8(output.data<float>(),shape[1],shape[2],image.size(),resized,score_,nms_);
  timing_.network_preprocess_ms=detection_ms(begin,prepared);
  timing_.network_infer_ms=detection_ms(prepared,inferred);
  timing_.network_decode_ms=detection_ms(inferred,DetectionClock::now());
  timing_.network_candidates=candidates_.size();
}
DetectionFrame YoloV8Backend::detect(const cv::Mat& image){
  infer_candidates(image);
  DetectionFrame f;f.backend="yolov8_ir_experimental";
  f.timing=timing_;
  f.observations.reserve(candidates_.size());
  for(const auto& c:candidates_)f.observations.push_back(c.detection);
  return f;
}
YoloV8Backend::~YoloV8Backend(){
  for(auto index:pending_)try{slots_[index].request.wait();}catch(...){}
}
void YoloV8Backend::configure_async(size_t capacity){
  if(capacity<1||capacity>4||!pending_.empty()||failed_)
    throw std::runtime_error("async capacity must be 1..4 with no pending/failed requests");
  slots_.clear();free_.clear();slots_.resize(capacity);last_frame_=-1;last_time_=-1;
  for(size_t i=0;i<capacity;++i){slots_[i].request=model_.create_infer_request();free_.push_back(i);}
}
void YoloV8Backend::submit(long frame,double time,const cv::Mat& image){
  if(failed_||free_.empty())throw std::runtime_error("async pool failed, unconfigured or full");
  if(frame<0||frame<=last_frame_||!std::isfinite(time)||time<0||time<=last_time_)
    throw std::runtime_error("async frame/time must be strictly increasing");
  auto index=free_.front();auto& slot=slots_[index];
  slot.begin=DetectionClock::now();slot.image_size=image.size();
  auto tensor=slot.request.get_input_tensor();
  slot.resized=prepare_yolov8_input(image,slot.input,tensor.data<float>(),tensor.get_size());
  slot.frame=frame;slot.time=time;
  // Keep ownership registered even if start_async throws; a failed pool is never reused.
  free_.pop_front();pending_.push_back(index);
  try{slot.request.start_async();}catch(...){failed_=true;throw;}
  last_frame_=frame;last_time_=time;
}
YoloV8Backend::AsyncResult YoloV8Backend::collect(){
  if(failed_||pending_.empty())throw std::runtime_error("async pool failed or empty");
  const auto index=pending_.front();auto& slot=slots_[index];
  try{
    slot.request.wait();auto tensor=slot.request.get_output_tensor();const auto shape=tensor.get_shape();
    AsyncResult result;result.frame=slot.frame;result.time=slot.time;
    result.candidates=decode_yolov8(tensor.data<float>(),shape[1],shape[2],slot.image_size,slot.resized,score_,nms_);
    result.latency_ms=detection_ms(slot.begin,DetectionClock::now());
    pending_.pop_front();free_.push_back(index);return result;
  }catch(...){failed_=true;throw;}
}
}
