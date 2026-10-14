#include "roi_refinement.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace outpost;
namespace {
void require(bool value,const char* reason){if(!value)throw std::runtime_error(reason);}
Detection network_at(float left,float right,float top,float bottom){
  Detection d;d.kind=ObservationKind::NeuralKeypoints;
  d.measurement_points={cv::Point2f(left,top),{right,top},{right,bottom},{left,bottom}};
  d.corners=d.measurement_points;d.center={(left+right)/2,(top+bottom)/2};
  d.classified_index=6;d.classification_confidence=.95;d.detection_confidence=.9;return d;
}
DetectionFrame run(const cv::Mat& image,const std::vector<Detection>& candidates,const DetectorConfig& config){
  auto measured=TraditionalBackend(config).detect(image);
  measured=refine_roi_lamp_support(image,std::move(measured),candidates,config);
  return supported_observations(std::move(measured),candidates,config);
}
cv::Mat weak_pair(){
  cv::Mat image(120,180,CV_8UC3,cv::Scalar());
  cv::rectangle(image,{42,45,5,25},{0,30,110},cv::FILLED);
  cv::rectangle(image,{82,45,5,25},{0,60,240},cv::FILLED);return image;
}
}
int main(){try{
  DetectorConfig config;config.roi_refinement_enabled=true;
  auto image=weak_pair();const auto candidate=network_at(44,84,45,69);
  auto original=TraditionalBackend(config).detect(image);
  require(original.observations.empty()&&original.lamps.size()==1,"weak fixture must miss one global lamp");
  auto disabled=config;disabled.roi_refinement_enabled=false;
  require(run(image,{candidate},disabled).observations.empty(),"disabled ROI experiment changed baseline");
  auto result=run(image,{candidate},config);
  require(result.observations.size()==1,"ROI did not restore a true weak pair");
  const auto recovered=result.observations.front();
  require(recovered.roi_refined&&recovered.kind==ObservationKind::NeuralSupportedLampPair,
          "ROI provenance missing");
  require(result.lamps.size()==2&&recovered.right==0&&recovered.left==1,
          "original strong-lamp global index was not preserved");
  require(result.pairs.size()==result.pair_status.size()&&result.pairs.size()==1,
          "restored pair global provenance incomplete");
  auto reference_config=config;reference_config.brightness=config.roi_brightness;
  const auto reference=TraditionalBackend(reference_config).detect(image);
  require(reference.observations.size()==1,"full-image reference segmentation failed");
  for(size_t k=0;k<4;++k)require(cv::norm(recovered.measurement_points[k]-reference.observations[0].measurement_points[k])<1e-5,
                               "ROI coordinates are not original image coordinates");
  require(result.reviews.size()==1&&result.reviews[0].decision=="accepted_roi_measured_pair","refinement not audited");

  // Semantic rejection occurs before correspondence; review ordering is still
  // exactly the caller's neural ordering.
  auto wrong=candidate;wrong.classified_index=8;
  result=run(image,{wrong,candidate},config);
  require(result.observations.size()==1&&result.reviews.size()==2&&
          result.reviews[0].decision=="class_rejected"&&result.reviews[1].decision=="accepted_roi_measured_pair",
          "rejected semantic duplicate stole unique lamp support");
  wrong.classified_index=6;wrong.classification_confidence=.1;
  require(run(image,{wrong,candidate},config).observations.size()==1,"low-confidence duplicate caused ambiguity");
  result=run(image,{candidate,candidate},config);
  require(result.observations.empty()&&result.lamps.size()==2&&result.pairs.size()==1,
          "duplicate neural ROIs duplicated lamps/pairs or accepted ambiguous support");

  image=weak_pair();cv::rectangle(image,{40,40,10,35},{0,0,0},cv::FILLED);
  require(run(image,{candidate},config).observations.empty(),"single lamp was inflated into a full pair");
  require(run(cv::Mat(120,180,CV_8UC3,cv::Scalar()),{candidate},config).observations.empty(),"blank ROI generated a measurement");
  image=weak_pair();
  // Deliberately undersized learned keypoints crop the actual lamp endpoints.
  auto tight=config;tight.roi_padding_lengths=.05;
  require(run(image,{network_at(44,84,50,64)},tight).observations.empty(),"artificial ROI border formed truncated complete lamps");
  image.setTo(cv::Scalar());
  for(int x:{42,82})cv::rectangle(image,{x,0,5,25},{0,30,110},cv::FILLED);
  auto permissive=config;permissive.reject_border_contours=false;
  require(run(image,{network_at(44,84,0,24)},permissive).observations.empty(),"ROI image-edge truncation bypassed completeness check");

  image.setTo(cv::Scalar());
  for(int x:{30,60,90})cv::rectangle(image,{x,45,5,25},{0,30,110},cv::FILLED);
  const auto first=network_at(32,62,45,69),second=network_at(62,92,45,69);
  result=run(image,{first,second},config);
  require(result.lamps.size()==3&&result.observations.empty(),"two recovered targets reserved the same global lamp");
  require(result.reviews.size()==2&&result.reviews[0].decision=="shared_lamp_support"&&
          result.reviews[1].decision=="shared_lamp_support","shared ROI lamp ambiguity not explained");

  image=weak_pair();auto bright=config;bright.brightness=90;bright.roi_brightness=90;
  auto direct=run(image,{candidate},bright);
  require(direct.observations.size()==1&&!direct.observations[0].roi_refined,
          "already supported observation was unnecessarily rebuilt");
  for(size_t k=0;k<4;++k)require(direct.observations[0].measurement_points[k]==reference.observations[0].measurement_points[k],
                               "ROI path changed an existing measured pair");
  std::cout<<"ROI lamp recovery and global provenance contracts passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
