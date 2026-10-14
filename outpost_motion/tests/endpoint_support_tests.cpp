#include "detector.hpp"
#include "tracker.hpp"
#include "roi_refinement.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <set>
#include <iomanip>
using namespace outpost;
namespace {
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
cv::Mat fixture(int top=40,int bottom=69){
  cv::Mat image(120,180,CV_8UC3,cv::Scalar());
  for(int x:{40,80})cv::rectangle(image,{x,top,7,bottom-top+1},{0,60,240},cv::FILLED);
  return image;
}
Detection only(Detector& detector,const cv::Mat& image){
  static int call=0;++call;const auto results=detector.detect(image);
  if(results.size()!=1)throw std::runtime_error("expected exactly one measured pair, fixture call "+std::to_string(call));
  return results.front();
}
void units(){
  DetectorConfig base,enabled;enabled.endpoint_refinement_enabled=true;
  Detector plain(base),refined(enabled);
  auto image=fixture();const auto before=only(plain,image),after=only(refined,image);
  require(before.endpoint_evidence[0]==EndpointEvidence::NotEvaluated&&!before.endpoint_refined,
          "disabled endpoint path changed metadata");
  require(after.endpoint_refined&&after.endpoint_evidence[0]==EndpointEvidence::ContrastSupported&&
          after.endpoint_evidence[1]==EndpointEvidence::ContrastSupported,"intact contrast edges not measured");
  require(std::abs(after.measurement_points[0].y-39.5)<.05&&std::abs(after.measurement_points[3].y-69.5)<.05,
          "subpixel half-contrast edges are not at pixel support boundaries");
  auto padded=enabled;padded.end_margin_min_px=10;padded.side_margin_min_px=10;Detector wide(padded);
  require(only(wide,image).measurement_points==after.measurement_points,"display padding changed endpoint measurements");
  auto disabled=enabled;disabled.endpoint_refinement_enabled=false;disabled.endpoint_search_fraction=.8;
  Detector legacy(disabled);require(only(legacy,image).measurement_points==before.measurement_points,
                                   "disabled experimental settings changed baseline coordinates");
  // Known gray ramps: the binary R>=130 mask clips the low-intensity ends,
  // whereas the actual 50%-contrast crossings remain at y=40 and y=70.
  image.setTo(cv::Scalar());
  for(int y=30;y<80;++y){const double signal=200*std::clamp((y-35.)/10.,0.,1.)*std::clamp((75.-y)/10.,0.,1.);
    for(int x0:{40,80})for(int x=x0;x<x0+7;++x)image.at<cv::Vec3b>(y,x)={0,0,uchar(cvRound(signal))};}
  const auto clipped=only(plain,image),ramp=only(refined,image);
  require(clipped.measurement_points[0].y>=42&&clipped.measurement_points[3].y<=68,"ramp did not reproduce threshold clipping");
  require(std::abs(ramp.measurement_points[0].y-40)<.3&&std::abs(ramp.measurement_points[3].y-70)<.3,
          "gray support did not recover real ramp crossings");
  // Rotate the actual image, not invented corner coordinates.
  cv::Mat tilted;cv::warpAffine(fixture(),tilted,cv::getRotationMatrix2D({63,55},20,1),fixture().size(),cv::INTER_LINEAR);
  const auto oblique=only(refined,tilted);
  require(oblique.endpoint_refined&&std::abs(oblique.measurement_points[0].x-oblique.measurement_points[3].x)>3,
          "tilted measured axis lost during endpoint refinement");
  // Both image and artificial ROI boundaries remain unmeasured, even when an
  // old caller disables the contour boundary gate.
  require(refined.detect(fixture(0,29)).empty(),"physical boundary lamp became complete");
  auto permissive=enabled;permissive.reject_border_contours=false;Detector edge(permissive);
  require(edge.detect(fixture(0,29)).empty()&&!edge.bars().empty()&&
          edge.bars()[0].endpoint_evidence==EndpointEvidence::BoundaryUnsupported,"no dark edge support was fabricated");
  cv::Mat near=fixture(4,33);require(refined.detect(near).empty(),"search crossing image boundary silently extrapolated");
  auto roi_config=enabled;roi_config.roi_refinement_enabled=true;roi_config.roi_padding_lengths=.05;
  Detection neural=after;neural.kind=ObservationKind::NeuralKeypoints;neural.classified_index=6;neural.classification_confidence=.95;
  image=fixture();for(int y=0;y<image.rows;++y)for(int x=0;x<image.cols;++x)
    if(image.at<cv::Vec3b>(y,x)[2])image.at<cv::Vec3b>(y,x)={0,30,110};
  neural.measurement_points={cv::Point2f(43,48),{83,48},{83,62},{43,62}};
  auto roi_frame=refine_roi_lamp_support(image,TraditionalBackend(roi_config).detect(image),{neural},roi_config);
  require(supported_observations(std::move(roi_frame),{neural},roi_config).observations.empty(),
          "artificial crop border supplied invented complete endpoints");
  image=fixture();cv::rectangle(image,{38,38,12,34},{0,0,0},cv::FILLED);
  require(refined.detect(image).empty(),"single refined lamp generated a complete armor");

  const auto full=only(refined,fixture()),short_pair=only(refined,fixture(48,69));
  require(short_pair.fit_residual_px<1e-5&&short_pair.endpoint_evidence[0]==EndpointEvidence::ContrastSupported,
          "single-frame occlusion ambiguity fixture no longer rectangular/contrast-supported");
  TrackerConfig policy;policy.common_endpoint_guard_enabled=true;Tracker tracker(policy);
  std::vector<TrackOutput> output;
  for(int i=0;i<5;++i)output=tracker.step(i/30.,{full},.1);
  require(output.size()==1&&output[0].measurement_used,"guard fixture did not mature");const auto id=output[0].id;
  for(int i=5;i<11;++i){output=tracker.step(i/30.,{short_pair},.1);
    require(output.size()==1&&output[0].id==id&&!output[0].measurement_used&&!output[0].future_valid&&
            output[0].endpoint_truncation_suspected,"common clipping became a measurement or duplicate birth");}
  output=tracker.step(11/30.,{full},.1);
  require(output.size()==1&&output[0].id==id&&output[0].measurement_used&&!output[0].endpoint_truncation_suspected,
          "recovered actual endpoints failed to retain identity");
  // Severe shortening used to fail ordinary length/area association then birth.
  Tracker severe(policy);for(int i=0;i<5;++i)severe.step(i/30.,{full},.1);
  const auto severe_pair=only(refined,fixture(58,69));output=severe.step(5/30.,{severe_pair},.1);
  require(output.size()==1&&!output[0].measurement_used&&output[0].endpoint_truncation_suspected,
          "severe truncation bypassed history through a new birth");
  require(std::any_of(severe.association_diagnostics().begin(),severe.association_diagnostics().end(),
      [](const auto& d){return std::string(d.reason)=="suspected_common_endpoint_truncation";}),"missing truncation diagnostic");
  // A whole target uniformly getting smaller is not common ENDPOINT clipping.
  auto scaled=full;for(auto& p:scaled.measurement_points)p=scaled.center+(p-scaled.center)*.65f;
  for(auto& p:scaled.corners)p=scaled.center+(p-scaled.center)*.65f;
  Tracker scale(policy);for(int i=0;i<5;++i)scale.step(i/30.,{full},.1);
  output=scale.step(5/30.,{scaled},.1);
  require(output.size()==1&&output[0].measurement_used&&!output[0].endpoint_truncation_suspected,
          "uniform perspective/distance scaling mistaken for common endpoint clipping");
  Tracker fresh(policy);for(int i=0;i<5;++i)output=fresh.step(i/30.,{short_pair},.1);
  require(output.size()==1&&output[0].measurement_used&&!output[0].endpoint_truncation_suspected,
          "single-frame shortening claimed known occlusion without history");
  auto uncertain=full;uncertain.endpoint_evidence[0]=EndpointEvidence::Uncertain;
  Tracker fallback(policy);for(int i=0;i<5;++i)output=fallback.step(i/30.,{uncertain},.1);
  require(output.size()==1&&output[0].measurement_used,"gray refinement uncertainty discarded measured binary contour");
  uncertain.endpoint_evidence[0]=EndpointEvidence::BoundaryUnsupported;
  Tracker invalid(policy);require(invalid.step(0,{uncertain},.1).empty(),"unsupported boundary birthed a measured track");
  auto impossible=enabled;impossible.endpoint_min_contrast=255;Detector no_gray(impossible);
  const auto retained=only(no_gray,fixture());
  require(!retained.endpoint_refined&&retained.endpoint_evidence[0]==EndpointEvidence::Uncertain&&
          retained.measurement_points==before.measurement_points,"failed gray refinement altered or removed binary measurements");
  std::cout<<"Endpoint gray-support, ambiguity, ROI boundaries and common-truncation contracts passed\n";
}
void video(const char* path){
  cv::VideoCapture capture(path);require(capture.isOpened(),"cannot open endpoint regression input");
  const double fps=capture.get(cv::CAP_PROP_FPS);require(fps>0,"invalid video fps");
  DetectorConfig base,experiment;base.endpoint_refinement_enabled=false;experiment=base;experiment.endpoint_refinement_enabled=true;
  Detector original(base),refined(experiment);
  TrackerConfig tracking,guard;guard.common_endpoint_guard_enabled=true;
  Tracker plain_track(tracking),refined_track(tracking),guard_track(guard),original_guard_track(guard);
  size_t frames=0,base_pairs=0,new_pairs=0,matched=0,invalid_geometry=0,uncertain_lamps=0,contrast_lamps=0;
  size_t used_plain=0,used_refined=0,used_guard=0,used_original_guard=0,suspect_frames=0,original_suspect_frames=0;
  size_t future_plain=0,future_refined=0,future_guard=0;double delta_sum=0,max_delta=0,center_sum=0;
  std::vector<long> flags,original_flags;cv::Mat image;
  while(capture.read(image)){
    const auto a=original.detect(image),b=refined.detect(image);base_pairs+=a.size();new_pairs+=b.size();
    for(const auto& bar:refined.bars()){
      uncertain_lamps+=bar.endpoint_evidence==EndpointEvidence::Uncertain;
      contrast_lamps+=bar.endpoint_evidence==EndpointEvidence::ContrastSupported;
    }
    for(const auto& d:b){
      const std::vector<cv::Point2f> polygon(d.measurement_points.begin(),d.measurement_points.end());
      if(!cv::isContourConvex(polygon)||cv::contourArea(polygon)<=0)++invalid_geometry;
      for(auto p:d.measurement_points)if(!std::isfinite(p.x)||!std::isfinite(p.y)||p.x<0||p.y<0||p.x>=image.cols||p.y>=image.rows)++invalid_geometry;
      std::vector<const Detection*> matches;for(const auto& old:a)if(cv::norm(old.center-d.center)<5)matches.push_back(&old);
      if(matches.size()==1){++matched;center_sum+=cv::norm(matches[0]->center-d.center);
        for(int k=0;k<4;++k){const auto delta=cv::norm(matches[0]->measurement_points[k]-d.measurement_points[k]);delta_sum+=delta;max_delta=std::max(max_delta,delta);}}
    }
    const double time=frames/fps;
    for(const auto& t:plain_track.step(time,a,.1)){used_plain+=t.measurement_used;future_plain+=t.future_valid;}
    for(const auto& t:refined_track.step(time,b,.1)){used_refined+=t.measurement_used;future_refined+=t.future_valid;}
    bool suspect=false,original_suspect=false;
    for(const auto& t:guard_track.step(time,b,.1)){used_guard+=t.measurement_used;future_guard+=t.future_valid;suspect|=t.endpoint_truncation_suspected;}
    for(const auto& t:original_guard_track.step(time,a,.1)){used_original_guard+=t.measurement_used;original_suspect|=t.endpoint_truncation_suspected;}
    if(suspect){++suspect_frames;flags.push_back(long(frames));}if(original_suspect){++original_suspect_frames;original_flags.push_back(long(frames));}
    ++frames;
  }
  std::cout<<std::setprecision(12)<<"frames="<<frames<<" fps="<<fps<<" baseline_pairs="<<base_pairs<<" refined_pairs="<<new_pairs
    <<" matched_pairs="<<matched<<" invalid_refined_geometry="<<invalid_geometry<<" contrast_lamps="<<contrast_lamps<<" uncertain_lamps="<<uncertain_lamps
    <<" mean_endpoint_delta_px="<<(matched?delta_sum/(4*matched):0)<<" max_endpoint_delta_px="<<max_delta
    <<" mean_center_delta_px="<<(matched?center_sum/matched:0)<<"\naccepted_confirmed_observations baseline="<<used_plain
    <<" refined="<<used_refined<<" refined_guard="<<used_guard<<" baseline_guard="<<used_original_guard
    <<"\nmodel_future_valid baseline="<<future_plain<<" refined="<<future_refined<<" refined_guard="<<future_guard
    <<"\nsuspect_frames="<<suspect_frames<<" frames:";
  for(long f:flags)std::cout<<' '<<f;
  std::cout<<"\nbaseline_guard_suspect_frames="<<original_suspect_frames<<" frames:";for(long f:original_flags)std::cout<<' '<<f;
  std::cout<<"\nNo labels: coordinate deltas and coverage are NOT endpoint accuracy or true-positive metrics.\n";
  require(frames==925&&invalid_geometry==0,"real-video completeness/geometry regression failed");
}
}
int main(int argc,char** argv){try{if(argc==1)units();else if(argc==2)video(argv[1]);else throw std::runtime_error("usage: endpoint_support_tests [video]");}
catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
