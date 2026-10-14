#include "detection_backend.hpp"
namespace outpost {
#ifdef OUTPOST_OPENVINO_ENABLED
std::unique_ptr<DetectionBackend> make_neural_supported(DetectorConfig);
#endif
std::unique_ptr<DetectionBackend> make_detector(const std::string& backend,DetectorConfig config){
  if(backend=="traditional")return std::make_unique<TraditionalBackend>(config);
#ifdef OUTPOST_OPENVINO_ENABLED
  if(backend=="neural_supported")return make_neural_supported(config);
#endif
  throw std::runtime_error("detector backend unavailable: "+backend+"; no silent fallback");
}
}
