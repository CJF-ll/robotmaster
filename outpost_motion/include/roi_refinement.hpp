#pragma once
#include "supported_observations.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <limits>

namespace outpost {
namespace roi_detail {
inline bool touches(const Lightbar& first,const Lightbar& second){
  const std::vector<cv::Point2f> a(first.vertices.begin(),first.vertices.end());
  const std::vector<cv::Point2f> b(second.vertices.begin(),second.vertices.end());
  std::vector<cv::Point2f> intersection;
  return cv::intersectConvexConvex(a,b,intersection,true)>1e-4;
}
inline int global_lamp(const Lightbar& lamp,std::vector<Lightbar>& lamps){
  int found=-1;
  for(size_t i=0;i<lamps.size();++i)if(touches(lamp,lamps[i])){
    // Overlapping image evidence must never acquire another independent lamp
    // index. If it could contain two old lamps or changes axis, reject it.
    if(found>=0||std::abs(lamp.axis.dot(lamps[i].axis))<.9)return -1;
    found=int(i);
  }
  if(found>=0)return found;
  lamps.push_back(lamp);return int(lamps.size())-1;
}
inline void translate(Lightbar& lamp,cv::Point2f offset){
  lamp.center+=offset;for(auto& point:lamp.vertices)point+=offset;
}
inline void translate(Detection& pair,cv::Point2f offset){
  pair.center+=offset;for(auto& point:pair.corners)point+=offset;
  for(auto& point:pair.measurement_points)point+=offset;
}
}

// Network geometry only proposes where to search. Every returned point is
// reconstructed from original pixels, retaining global lamp and pair indices.
// The caller subsequently runs supported_observations to enforce mutual unique
// network/pair matching and globally exclusive lamp ownership.
inline DetectionFrame refine_roi_lamp_support(const cv::Mat& image,DetectionFrame measured,
    const std::vector<Detection>& neural,const DetectorConfig& config){
  if(!config.roi_refinement_enabled)return measured;
  if(image.type()!=CV_8UC3||config.roi_brightness<1||config.roi_brightness>config.brightness||
     !std::isfinite(config.roi_padding_lengths)||config.roi_padding_lengths<=0||
     config.roi_padding_lengths>2)throw std::runtime_error("invalid ROI refinement input/configuration");
  std::vector<Detection> qualified;
  for(const auto& d:neural)if(!neural_classification_rejection(d,config))qualified.push_back(d);
  // Only missing measured support needs another segmentation pass. An existing
  // ambiguous match is never resolved by rescanning the same image evidence.
  const auto initial=detection_correspondence(qualified,measured.observations,config.support_endpoint_fraction);
  for(size_t index=0;index<qualified.size();++index){
    if(initial[index].traditional>=0||initial[index].ambiguous)continue;
    const auto& candidate=qualified[index];
    float minx=std::numeric_limits<float>::infinity(),miny=minx,maxx=-minx,maxy=-minx;
    bool finite=true;
    for(auto p:candidate.measurement_points){
      finite=finite&&std::isfinite(p.x)&&std::isfinite(p.y);
      minx=std::min(minx,p.x);maxx=std::max(maxx,p.x);miny=std::min(miny,p.y);maxy=std::max(maxy,p.y);
    }
    if(!finite)continue;
    const double length=.5*(cv::norm(candidate.measurement_points[3]-candidate.measurement_points[0])+
                             cv::norm(candidate.measurement_points[2]-candidate.measurement_points[1]));
    if(length<config.min_length||length>config.max_length||minx>=maxx||miny>=maxy)continue;
    const double margin=length*config.roi_padding_lengths;
    // Clip only the search rectangle. Contours touching either the artificial
    // crop edge or a physical image edge are incomplete and rejected below.
    const int x0=int(std::floor(std::clamp(double(minx)-margin,0.,double(image.cols))));
    const int y0=int(std::floor(std::clamp(double(miny)-margin,0.,double(image.rows))));
    const int x1=int(std::ceil(std::clamp(double(maxx)+margin+1,0.,double(image.cols))));
    const int y1=int(std::ceil(std::clamp(double(maxy)+margin+1,0.,double(image.rows))));
    if(x1-x0<3||y1-y0<3)continue;
    auto local_config=config;local_config.brightness=config.roi_brightness;
    local_config.reject_border_contours=true;local_config.border_margin_px=0;
    Detector local(local_config);local.detect(image(cv::Rect(x0,y0,x1-x0,y1-y0)));
    auto pairs=local.pair_candidates();
    const cv::Point2f offset{float(x0),float(y0)};
    for(auto& pair:pairs)roi_detail::translate(pair,offset);
    const auto correspondence=detection_correspondence({candidate},pairs,config.support_endpoint_fraction);
    if(correspondence.front().traditional<0)continue;
    const auto& pair=pairs.at(size_t(correspondence.front().traditional));
    auto left=local.bars().at(size_t(pair.left)),right=local.bars().at(size_t(pair.right));
    roi_detail::translate(left,offset);roi_detail::translate(right,offset);
    auto staged_lamps=measured.lamps;
    const int left_index=roi_detail::global_lamp(left,staged_lamps);
    const int right_index=roi_detail::global_lamp(right,staged_lamps);
    if(left_index<0||right_index<0||left_index==right_index)continue;
    Detection restored;
    if(!enclose(staged_lamps[size_t(left_index)],staged_lamps[size_t(right_index)],config,restored))continue;
    if(detection_correspondence({candidate},{restored},config.support_endpoint_fraction).front().traditional<0)continue;
    restored.left=left_index;restored.right=right_index;restored.roi_refined=true;
    measured.lamps=std::move(staged_lamps);
    const auto same_pair=[&](const Detection& p){return p.left==left_index&&p.right==right_index;};
    if(std::none_of(measured.observations.begin(),measured.observations.end(),same_pair))
      measured.observations.push_back(restored);
    if(std::none_of(measured.pairs.begin(),measured.pairs.end(),same_pair)){
      measured.pairs.push_back(restored);measured.pair_status.push_back(Detector::PairStatus::Accepted);
    }
  }
  return measured;
}
}
