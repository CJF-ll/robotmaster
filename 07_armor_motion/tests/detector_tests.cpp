#include "detector.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdexcept>
#include <limits>
using namespace outpost;
void require(bool value,const char* message) {
  if(!value) throw std::runtime_error(message);
}
Lightbar bar(float x,float y,float length,float width,float tilt) {
  Lightbar b; b.center={x,y}; b.length=length;b.width=width;
  b.axis={std::sin(tilt),std::cos(tilt)};
  cv::Point2f n(b.axis.y,-b.axis.x);
  b.vertices={b.center-b.axis*length/2-n*width/2,b.center-b.axis*length/2+n*width/2,
              b.center+b.axis*length/2+n*width/2,b.center+b.axis*length/2-n*width/2};
  return b;
}
int main() {
  DetectorConfig config; Detection d;
  cv::Mat thin_image(100,120,CV_8UC3,cv::Scalar());
  auto old_config=config;old_config.thin_vertical_enabled=false;
  Detector old_thin(old_config),thin_detector(config);
  for(int x:{30,60})cv::line(thin_image,{x,30},{x,45},{0,80,255},1);
  require(old_thin.detect(thin_image).empty(),"thin fixture did not reproduce previous miss");
  auto recovered=thin_detector.detect(thin_image);
  require(recovered.size()==1&&recovered[0].low_resolution&&thin_detector.thin_accepted_lamps()==2,
          "continuous single-column pair not recovered/tagged");
  for(const auto& b:thin_detector.bars())require(b.width==1&&b.length==16&&b.low_resolution,
                                               "thin support geometry invented a resolved width");
  thin_image.setTo(cv::Scalar());
  for(int x:{30,60})cv::line(thin_image,{x,30},{x,35},{0,80,255},1);
  require(thin_detector.detect(thin_image).empty(),"short noise runs recovered");
  thin_image.setTo(cv::Scalar());
  for(int x:{30,60})for(int y=20;y<50;y+=2)thin_image.at<cv::Vec3b>(y,x)={0,80,255};
  require(thin_detector.detect(thin_image).empty(),"disconnected noise joined into lamps");
  thin_image.setTo(cv::Scalar());
  for(int x:{30,60})cv::line(thin_image,{x,30},{x+15,30},{0,80,255},1);
  require(thin_detector.detect(thin_image).empty(),"horizontal one-pixel streaks recovered");
  thin_image.setTo(cv::Scalar());
  for(int x:{30,60})cv::line(thin_image,{x,0},{x,15},{0,80,255},1);
  require(thin_detector.detect(thin_image).empty(),"thin recovery bypassed border rejection");
  thin_image.setTo(cv::Scalar());cv::line(thin_image,{30,30},{30,45},{0,80,255},1);
  require(thin_detector.detect(thin_image).empty(),"one thin lamp made a full observation");
  thin_image.setTo(cv::Scalar());thin_detector.detect(thin_image);
  require(thin_detector.thin_accepted_lamps()==0,"thin diagnostics leaked across frames");
  auto permissive=config;permissive.reject_border_contours=false;
  Detector old_border(permissive),safe_border(config);
  for(int edge=0;edge<4;++edge){
    cv::Mat clipped(100,120,CV_8UC3,cv::Scalar());
    int x1=20,x2=50,y=30;
    if(edge==0)y=0;
    if(edge==1)y=80;
    if(edge==2){x1=0;x2=30;}
    if(edge==3){x1=86;x2=116;}
    cv::rectangle(clipped,{x1,y,4,20},{0,80,255},cv::FILLED);
    cv::rectangle(clipped,{x2,y,4,20},{0,80,255},cv::FILLED);
    require(old_border.detect(clipped).size()==1,"border regression did not exercise old false complete pair");
    require(safe_border.detect(clipped).empty()&&safe_border.border_rejected_contours()>0,
            "truncated boundary lamp accepted as complete armor");
    require(safe_border.pair_candidates().empty(),"boundary lamp leaked into pair candidates");
  }
  cv::Mat just_inside(100,120,CV_8UC3,cv::Scalar());
  for(int x:{1,31})cv::rectangle(just_inside,{x,30,4,20},{0,80,255},cv::FILLED);
  require(safe_border.detect(just_inside).size()==1&&safe_border.border_rejected_contours()==0,
          "complete one-pixel-inset lamps rejected");
  auto padded=config;padded.border_margin_px=2;Detector padded_border(padded);
  require(padded_border.detect(just_inside).empty()&&padded_border.border_rejected_contours()==1,
          "configured image safety margin ignored");
  safe_border.detect(cv::Mat(100,120,CV_8UC3,cv::Scalar()));
  require(safe_border.border_rejected_contours()==0,"border diagnostics leaked across frames");
  auto a=bar(30,40,20,3,0),b=bar(60,40,20,3,0);
  require(enclose(a,b,config,d),"vertical lamps rejected");
  a=bar(30,40,18,3,-.15);b=bar(60,44,24,4,.18);
  require(enclose(a,b,config,d),"tilted unequal lamps rejected");
  require(cv::norm(d.measurement_points[0]-(a.center-a.axis*a.length*.5f))<1e-5,
          "measurement point is not lamp centerline endpoint");
  Detection expanded;auto expanded_config=config;expanded_config.end_margin_ratio=.5;
  expanded_config.side_margin_ratio=.5;
  expanded_config.end_margin_min_px=10;
  expanded_config.side_margin_min_px=5;
  require(enclose(a,b,expanded_config,expanded),"expanded enclosure invalid");
  for(int i=0;i<4;++i)require(cv::norm(expanded.measurement_points[i]-d.measurement_points[i])<1e-5,
                            "display padding changed measurement points");
  std::vector<cv::Point2f> poly(d.corners.begin(),d.corners.end());
  require(cv::isContourConvex(poly),"nonconvex display quad");
  for(auto* lamp:{&a,&b}) for(auto p:lamp->vertices)
    require(cv::pointPolygonTest(poly,p,false)>=0,"lamp vertex excluded");
  require(!enclose(b,a,config,d),"reversed pair accepted");
  b=a;require(!enclose(a,b,config,d),"degenerate pair accepted");
  b=bar(60,40,0,3,0);require(!enclose(a,b,config,d),"zero length accepted");
  a=bar(30,40,20,3,0);b=bar(60,48,20,3,0);
  require(enclose(a,b,config,d),"vertically staggered pair rejected");
  const auto upper=d.corners[1]-d.corners[0];
  require(std::abs(upper.y/upper.x-8./30)<1e-5,"top edge ignores measured vertical offset");
  require(cv::norm(d.center-cv::Point2f(45,44))<1e-5,"enclosure changed measurement center");
  // Old horizontal support enclosure: width 34.5, height 31.2.
  require(cv::contourArea(std::vector<cv::Point2f>(d.corners.begin(),d.corners.end()))<34.5*31.2,
          "staggered enclosure not tighter than horizontal support lines");
  b.center.x=std::numeric_limits<float>::quiet_NaN();
  require(!enclose(a,b,config,d),"nonfinite lamp accepted");
  b=bar(60,40,20,3,0);b.axis={0,2};
  require(!enclose(a,b,config,d),"non-unit lamp axis accepted");
  int accepted=0;
  for(float dy:{-12.f,-6.f,0.f,6.f,12.f})for(float left_angle:{-.3f,0.f,.3f})
    for(float right_angle:{-.3f,0.f,.3f})for(float length:{14.f,20.f,26.f}){
      a=bar(40,50,20,3,left_angle);b=bar(70,50+dy,length,4,right_angle);
      if(!enclose(a,b,config,d))continue;++accepted;
      std::vector<cv::Point2f> quad(d.corners.begin(),d.corners.end());
      require(cv::isContourConvex(quad),"support lines produced flipped quad");
      for(auto* lamp:{&a,&b})for(auto p:lamp->vertices)
        require(cv::pointPolygonTest(quad,p,false)>=0,"sloped support excluded lamp vertex");
      require(cv::norm(d.center-(a.center+b.center)*.5f)<1e-5,"shape changed center");
    }
  require(accepted>100,"geometry sweep rejected excessive valid samples");
  cv::Mat image(100,120,CV_8UC3,cv::Scalar(0));
  cv::rectangle(image,{20,30,4,20},{0,80,255},cv::FILLED);
  Detector detector(config);
  require(detector.detect(image).empty(),"single lamp made armor");
  cv::rectangle(image,{50,30,4,20},{0,80,255},cv::FILLED);
  require(detector.detect(image).size()==1,"synthetic pair not detected");
  cv::rectangle(image,{80,30,4,20},{0,80,255},cv::FILLED);
  require(detector.detect(image).empty(),"three equal lamps arbitrarily paired by index");
  require(detector.pair_candidates().size()==3&&detector.pair_statuses().size()==3,
          "rejected pair provenance lost");
  for(auto state:detector.pair_statuses())require(state==Detector::PairStatus::AmbiguousLamp,
                                               "ambiguous equal-score pair not marked");
  image.setTo(cv::Scalar(0));
  cv::rectangle(image,{20,30,4,20},{0,80,255},cv::FILLED);
  cv::rectangle(image,{50,30,4,20},{0,80,255},cv::FILLED);
  cv::rectangle(image,{80,33,4,14},{0,80,255},cv::FILLED);
  auto best=detector.detect(image);
  require(best.size()==1&&best[0].left==0&&best[0].right==1,
          "poor tied alternatives suppressed clearly better pair");
  cv::Mat separated(100,240,CV_8UC3,cv::Scalar(0));
  for(int x:{20,50,160,190})cv::rectangle(separated,{x,30,4,20},{0,80,255},cv::FILLED);
  require(detector.detect(separated).size()==2,"independent pairs rejected as ambiguity");
  require(detector.detect(cv::Mat(100,120,CV_8UC3,cv::Scalar(0))).empty()&&
          detector.pair_candidates().empty()&&detector.pair_statuses().empty(),
          "pair diagnostics leaked across empty frames");
  image.setTo(cv::Scalar(0));
  cv::rectangle(image,{20,30,25,4},{0,80,255},cv::FILLED);
  cv::rectangle(image,{60,30,25,4},{0,80,255},cv::FILLED);
  require(detector.detect(image).empty(),"horizontal clutter treated as upright lamps");
  std::cout<<"detector geometry and synthetic image checks passed\n";
}
