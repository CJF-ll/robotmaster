#pragma once
#include "detection_backend.hpp"
#include "detection_correspondence.hpp"
namespace outpost {
inline const char* neural_classification_rejection(const Detection& candidate,
                                                  const DetectorConfig& config){
  if(candidate.kind!=ObservationKind::NeuralKeypoints||candidate.classified_index<0||candidate.classified_index>=9)
    return "invalid_classification";
  if(candidate.classified_index!=config.classifier_target_index||
     !std::isfinite(candidate.classification_confidence)||candidate.classification_confidence>1||
     candidate.classification_confidence<config.classifier_threshold)return "class_rejected";
  return nullptr;
}
// Preserve measured geometry and all lamp/pair reservations. No unmatched
// neural keypoint can become a lamp measurement or physical identity.
inline DetectionFrame supported_observations(DetectionFrame measured,
    const std::vector<Detection>& neural,const DetectorConfig& config){
  // Semantic rejection happens before one-to-one matching. A rejected network
  // duplicate is diagnostic evidence, not a competitor for a measured lamp.
  std::vector<Detection> qualified;
  std::vector<int> qualified_index(neural.size(),-1);
  for(size_t i=0;i<neural.size();++i)if(!neural_classification_rejection(neural[i],config)){
    qualified_index[i]=int(qualified.size());qualified.push_back(neural[i]);
  }
  const auto matches=detection_correspondence(qualified,measured.observations,config.support_endpoint_fraction);
  const auto original=measured.observations;measured.observations.clear();measured.reviews.clear();
  measured.backend="neural_supported";
  // A refined candidate can use a globally existing lamp. Enforce unique lamp
  // ownership across all accepted pairs, not just unique pair indices.
  std::vector<int> owners(measured.lamps.size(),0);
  for(const auto& match:matches)if(match.traditional>=0){
    const auto& pair=original.at(size_t(match.traditional));
    for(int lamp:{pair.left,pair.right})if(lamp>=0&&size_t(lamp)<owners.size())++owners[size_t(lamp)];
  }
  for(size_t i=0;i<neural.size();++i){const auto& candidate=neural[i];
    std::string reason;
    if(const auto* rejection=neural_classification_rejection(candidate,config))reason=rejection;
    else{
      const auto& match=matches.at(size_t(qualified_index[i]));
      if(match.ambiguous)reason="ambiguous_lamp_support";
      else if(match.traditional<0)reason="no_lamp_support";
      else{
        auto accepted=original.at(size_t(match.traditional));
        if(accepted.kind!=ObservationKind::LampPair)throw std::runtime_error("support geometry is not measured lamps");
        const bool shared=(accepted.left>=0&&size_t(accepted.left)<owners.size()&&owners[size_t(accepted.left)]>1)||
                          (accepted.right>=0&&size_t(accepted.right)<owners.size()&&owners[size_t(accepted.right)]>1);
        if(shared)reason="shared_lamp_support";
        else{
          accepted.kind=ObservationKind::NeuralSupportedLampPair;
          accepted.detection_confidence=candidate.detection_confidence;
          accepted.classification_confidence=candidate.classification_confidence;
          accepted.classified_index=candidate.classified_index;
          accepted.raw_network_class=candidate.raw_network_class;
          measured.observations.push_back(accepted);
          reason=accepted.roi_refined?"accepted_roi_measured_pair":"accepted_measured_pair";
        }
      }
    }
    measured.reviews.push_back({candidate,reason});
  }
  return measured;
}
}
