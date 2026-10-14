#include "detector.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdexcept>
using namespace outpost;
void require(bool value,const char* reason){if(!value)throw std::runtime_error(reason);}
int main(){try{
  DetectorConfig config;cv::Mat image(120,180,CV_8UC3,cv::Scalar());
  for(int x:{40,80})cv::rectangle(image,{x,40,7,30},{0,60,240},cv::FILLED);
  Detector clean(config);auto pairs=clean.detect(image);
  require(pairs.size()==1&&pairs[0].quality_available,"clean quality missing");
  require(pairs[0].fit_residual_px<1e-5&&pairs[0].center_variance_scale[0]<1.00001&&pairs[0].center_variance_scale[1]<1.00001,
          "ordinary lamp width incorrectly counted as fit error");
  auto padded=config;padded.side_margin_min_px=10;padded.end_margin_min_px=10;
  const auto expanded=Detector(padded).detect(image);
  require(expanded.size()==1&&expanded[0].measurement_points==pairs[0].measurement_points&&
          expanded[0].center==pairs[0].center&&expanded[0].fit_residual_px==pairs[0].fit_residual_px&&
          expanded[0].center_variance_scale==pairs[0].center_variance_scale,
          "display padding changed measurement quality or coordinates");
  auto left=clean.bars()[0],right=clean.bars()[1];
  left.quality_available=right.quality_available=false;Detection legacy;
  require(enclose(left,right,config,legacy)&&!legacy.quality_available&&legacy.center_variance_scale==std::array<double,2>{1,1},
          "legacy lamp fixtures lost compatible quality defaults");
  // Repeated one-sided contour notches retain the same outer min-area box.
  // Only the measured boundary residual increases; the display does not move.
  for(int x:{40,80})for(int y=45;y<64;y+=4)cv::rectangle(image,{x,y,2,2},{0,0,0},cv::FILLED);
  const auto irregular=Detector(config).detect(image);
  require(irregular.size()==1&&irregular[0].quality_available&&irregular[0].fit_residual_px>.1,
          "jagged fitted boundaries did not expose residual");
  require(irregular[0].center_variance_scale[0]>irregular[0].center_variance_scale[1]&&
          irregular[0].center_variance_scale[1]>=1,"vertical side noise lost directional information");
  require(irregular[0].measurement_points==pairs[0].measurement_points&&irregular[0].center==pairs[0].center,
          "quality extraction modified traditional coordinates");
  cv::Mat tilted;
  cv::warpAffine(image,tilted,cv::getRotationMatrix2D({60,55},25,1),image.size(),cv::INTER_NEAREST);
  const auto oblique=Detector(config).detect(tilted);
  require(oblique.size()==1&&oblique[0].quality_available&&
          oblique[0].center_variance_scale[0]>1&&oblique[0].center_variance_scale[1]>1,
          "tilted contour residual did not project uncertainty to both image axes");
  image.setTo(cv::Scalar());for(int x:{40,80})cv::line(image,{x,40},{x,69},{0,60,240},1);
  const auto thin=Detector(config).detect(image);
  require(thin.size()==1&&thin[0].low_resolution&&thin[0].quality_available&&thin[0].fit_residual_px==0&&
          thin[0].center_variance_scale[0]==2&&thin[0].center_variance_scale[1]==1,
          "unresolved one-column width lacks explicit transverse uncertainty");
  std::cout<<"Contour quality and geometry independence contracts passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
