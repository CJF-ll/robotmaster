#include "yolov8_backend.hpp"
#include "classifier_contract.hpp"
#include "supported_observations.hpp"
#include "roi_refinement.hpp"
#include <future>
namespace outpost {
class NeuralSupportedBackend final:public DetectionBackend {
 public:
  explicit NeuralSupportedBackend(DetectorConfig config):config_(config),traditional_(config),
    network_(config.neural_model,config.neural_score,config.neural_nms,config.neural_inference_threads){
    auto graph=core_.read_model(config.classifier_model);
    if(graph->inputs().size()!=1||graph->outputs().size()!=1||
       !graph->input().get_partial_shape().compatible(ov::PartialShape{1,1,32,32}))
      throw std::runtime_error("incompatible classifier input");
    graph->reshape(ov::Shape{1,1,32,32});
    if(graph->input().get_element_type()!=ov::element::f32||graph->output().get_element_type()!=ov::element::f32||
       graph->output().get_shape()!=ov::Shape{1,9})throw std::runtime_error("classifier must output f32 [1,9]");
    model_=core_.compile_model(graph,"CPU",ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    request_=model_.create_infer_request();
  }
  void reconfigure(const DetectorConfig& config)override{
    // ConfigRuntime rejects neural-settings changes before committing them.
    traditional_.reconfigure(config);config_=config;
  }
  DetectionFrame detect(const cv::Mat& image)override{
    // At most one traditional task in flight. Join before return/reconfigure;
    // Mat ownership survives exceptions and no OpenVINO request is shared.
    std::future<DetectionFrame> task;
    DetectionFrame measured;
    if(config_.parallel_branches)task=std::async(std::launch::async,[this,image]{return traditional_.detect(image);});
    else measured=traditional_.detect(image);
    network_.infer_candidates(image);
    if(task.valid())measured=task.get();
    std::vector<Detection> candidates;
    auto timing=network_.timing();timing.traditional_ms=measured.timing.traditional_ms;
    const auto classification_begin=DetectionClock::now();
    candidates.reserve(network_.candidates().size());
    for(const auto& result:network_.candidates()){
      auto d=result.detection;d.kind=ObservationKind::NeuralKeypoints;
      d.detection_confidence=result.confidence;d.raw_network_class=result.raw_class;
      const auto pattern=classifier_pattern(image,d.measurement_points,config_.classifier_crop_half_scale);
      if(!pattern.image.empty()&&!pattern.clipped){
        ++timing.classifier_calls;
        const auto input=classifier_input(pattern.image);auto tensor=request_.get_input_tensor();
        std::copy(input.ptr<float>(),input.ptr<float>()+1024,tensor.data<float>());
        request_.infer();auto output=request_.get_output_tensor();const auto p=classifier_probabilities(output.data<float>());
        const auto index=std::max_element(p.begin(),p.end())-p.begin();
        d.classified_index=int(index);d.classification_confidence=p[size_t(index)];
      }
      candidates.push_back(d);
    }
    const auto matching_begin=DetectionClock::now();
    timing.classification_ms=detection_ms(classification_begin,matching_begin);
    measured=refine_roi_lamp_support(image,std::move(measured),candidates,config_);
    auto result=supported_observations(std::move(measured),candidates,config_);
    timing.matching_ms=detection_ms(matching_begin,DetectionClock::now());
    result.timing=timing;return result;
  }
 private:
  DetectorConfig config_;TraditionalBackend traditional_;YoloV8Backend network_;
  ov::Core core_;ov::CompiledModel model_;ov::InferRequest request_;
};
std::unique_ptr<DetectionBackend> make_neural_supported(DetectorConfig config){
  return std::make_unique<NeuralSupportedBackend>(std::move(config));
}
}
