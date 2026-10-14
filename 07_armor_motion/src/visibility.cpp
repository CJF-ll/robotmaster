#include "visibility.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
namespace outpost {
RenderDecision visibility(const TrackOutput& t,const std::vector<Lightbar>& bars,
                          const VisibilityConfig& c) {
  // A numeric center may survive a degenerate translated silhouette. Keep
  // geometry rejection at the rendering boundary, not in the state output.
  for(auto point:t.current)if(!std::isfinite(point.x)||!std::isfinite(point.y))return {};
  const std::vector<cv::Point2f> polygon(t.current.begin(),t.current.end());
  if(!cv::isContourConvex(polygon)||cv::contourArea(polygon)<=4)return {};
  if(t.measurement_used)return {Evidence::AcceptedPair,true,t.future_valid};
  for(const auto& b:bars) {
    if(b.length<=0||!std::isfinite(b.length)||!std::isfinite(b.center.x)||
       !std::isfinite(b.center.y)||!std::isfinite(b.axis.x)||!std::isfinite(b.axis.y)||
       std::abs(cv::norm(b.axis)-1)>1e-3)continue;
    for(auto indices:{std::pair<int,int>{0,3},{1,2}}){
      const auto a=t.lamp_support_points[indices.first],end=t.lamp_support_points[indices.second];
      if(!std::isfinite(a.x)||!std::isfinite(a.y)||!std::isfinite(end.x)||!std::isfinite(end.y))continue;
      const auto side=end-a;const double length=cv::norm(side);
      if(length<=0||!std::isfinite(length))continue;
      const double ratio=length/b.length;
      if(ratio>c.length_ratio||ratio<1/c.length_ratio)continue;
      const cv::Point2f axis=side/float(length);
      const double cosine=std::clamp(std::abs(double(axis.dot(b.axis))),0.,1.);
      if(std::acos(cosine)*180/CV_PI>c.angle_degrees)continue;
      // Compare with the finite predicted lamp centerline, never an expanded
      // display edge. The tolerance is an explicit ratio of measured length.
      const double along=(b.center-a).dot(axis);
      // A nearby endpoint is not sufficient evidence: require overlap of
      // the two finite lamp extents along the predicted lamp axis.
      const double projected_length=b.length*cosine;
      if(projected_length<=1e-9)continue;
      const double overlap=std::max(0.,std::min(length,along+projected_length*.5)-
                                       std::max(0.,along-projected_length*.5));
      if(overlap/std::min(length,projected_length)+1e-9<c.min_axial_overlap)continue;
      const auto closest=a+axis*float(std::clamp(along,0.,length));
      if(cv::norm(b.center-closest)<=c.side_distance_ratio*b.length)
        return {Evidence::CompatibleSingleLamp,true,false};
    }
  }
  return {};
}
std::vector<RenderDecision> visibility_batch(const std::vector<TrackOutput>& tracks,
    const std::vector<Lightbar>& bars,const std::vector<Detection>& detections,const VisibilityConfig& c){
  std::vector<bool> reserved(bars.size(),false);
  // Reserve all paired lamps, including tentative or rejected pair candidates:
  // none may simultaneously corroborate an unrelated coasting track.
  for(const auto& d:detections)for(int index:{d.left,d.right})
    if(index>=0&&size_t(index)<bars.size())reserved[index]=true;
  std::vector<RenderDecision> result(tracks.size());
  std::vector<std::vector<size_t>> supports(bars.size());
  for(size_t i=0;i<tracks.size();++i){
    if(tracks[i].measurement_used){result[i]=visibility(tracks[i],{},c);continue;}
    for(size_t j=0;j<bars.size();++j)if(!reserved[j]&&visibility(tracks[i],{bars[j]},c).current)
      supports[j].push_back(i);
  }
  for(const auto& owners:supports)if(owners.size()==1)
    result[owners.front()]={Evidence::CompatibleSingleLamp,true,false};
  return result;
}
}
