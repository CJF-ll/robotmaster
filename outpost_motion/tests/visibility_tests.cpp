#include "visibility.hpp"
#include <stdexcept>
#include <iostream>
#include <opencv2/imgproc.hpp>
using namespace outpost;
void require(bool v,const char* why){if(!v)throw std::runtime_error(why);}
int main(){
  VisibilityConfig config;TrackOutput t;t.future_valid=true;
  t.current={cv::Point2f(20,20),{60,20},{60,40},{20,40}};
  t.lamp_support_points=t.current;
  auto r=visibility(t,{},config);
  require(!r.current&&!r.future,"unobserved free-floating box rendered");
  t.measurement_used=true;r=visibility(t,{},config);
  require(r.current&&r.future&&r.evidence==Evidence::AcceptedPair,"accepted pair suppressed");
  t.measurement_used=false;
  Lightbar b;b.center={22,30};b.axis={0,1};b.length=17;b.width=3;
  r=visibility(t,{b},config);
  require(r.current&&!r.future&&r.evidence==Evidence::CompatibleSingleLamp,"partial evidence misreported");
  require(!t.measurement_used,"single lamp promoted to full measurement");
  auto tip=b;tip.center={20,19};tip.length=20;
  require(!visibility(t,{tip},config).current,"lamp near endpoint but mostly outside supported coast");
  tip.center={20,41};require(!visibility(t,{tip},config).current,"lower endpoint bypassed overlap gate");
  tip.center={20,22};require(visibility(t,{tip},config).current,"overlap boundary rejected");
  tip.center={20,21.9f};require(!visibility(t,{tip},config).current,"below-boundary overlap accepted");
  tip.center={20,30};tip.length=10;
  require(visibility(t,{tip},config).current,"contained shorter lamp rejected");
  tip.length=40;require(visibility(t,{tip},config).current,"contained predicted extent rejected");
  tip.length=20;tip.axis={0,-1};require(visibility(t,{tip},config).current,"axis sign changed support");
  auto disabled=config;disabled.min_axial_overlap=0;tip.center={20,19};
  require(visibility(t,{tip},disabled).current,"zero overlap setting not backwards compatible");
  const auto unpadded=t.current;
  for(auto& p:t.current)p=cv::Point2f(40,30)+(p-cv::Point2f(40,30))*4;
  require(visibility(t,{b},config).current,"display padding changed single-lamp evidence");
  t.current=unpadded;
  auto missing=t;missing.lamp_support_points={};
  require(!visibility(missing,{b},config).current,"missing lamp geometry fell back to display");
  auto malformed=b;malformed.axis={0,2};
  require(!visibility(t,{malformed},config).current,"nonunit lamp axis accepted");
  b.center={40,30};require(!visibility(t,{b},config).current,"interior unrelated lamp supported a side");
  b.center={22,80};require(!visibility(t,{b},config).current,"infinite side extension used");
  b.center={22,30};b.axis={1,0};require(!visibility(t,{b},config).current,"horizontal lamp supported upright side");
  b.axis={0,1};b.length=2;require(!visibility(t,{b},config).current,"wrong lamp scale accepted");
  b.length=17;t.future_valid=false;require(!visibility(t,{b},config).future,"invalid future rendered");
  Detection pair;pair.left=0;pair.right=1;
  auto batch=visibility_batch({t},{b},{pair},config);
  require(!batch[0].current,"paired lamp reused by unrelated coasting track");
  batch=visibility_batch({t,t},{b},{},config);
  require(!batch[0].current&&!batch[1].current,"one lamp supports two ambiguous tracks");
  cv::Mat image(100,120,CV_8UC3,cv::Scalar(0));
  // Align the ambiguous lamps with the predicted extent so this test isolates
  // pair reservation, rather than being rejected earlier by axial overlap.
  for(int x:{20,50,80})cv::rectangle(image,{x,20,4,20},{0,80,255},cv::FILLED);
  Detector detector(DetectorConfig{});
  require(detector.detect(image).empty(),"ambiguous synthetic lamps accepted");
  require(visibility(t,detector.bars(),config).current,"test does not exercise lamp support bypass");
  batch=visibility_batch({t},detector.bars(),detector.pair_candidates(),config);
  require(!batch[0].current&&!batch[0].future,"ambiguous pairing bypassed rejection via single-lamp support");
  TrackerConfig tc;tc.confirmation_hits=1;Tracker tracker(tc);
  for(int i=0;i<20;++i){Detection d;d.center={float(100+i),50};
    d.measurement_points={cv::Point2f(90+i,40),{float(110+i),40},{float(110+i),60},{float(90+i),60}};
    for(int k=0;k<4;++k)d.corners[k]=d.center+(d.measurement_points[k]-d.center)*4;
    tracker.step(i/30.,{d},.1);
  }
  const auto coast=tracker.step(20/30.,{},.1).at(0);
  require(!coast.measurement_used&&cv::norm(coast.lamp_support_points[0]-(coast.current_center+cv::Point2f(-10,-10)))<1e-4,
          "coast centerline did not follow predicted center");
  Lightbar support;support.center=coast.current_center+cv::Point2f(-10,0);support.axis={0,1};support.length=20;
  const auto rendered=visibility(coast,{support},config);
  require(rendered.current&&!rendered.future,"coast centerline not used for single-lamp evidence");
  std::cout<<"visibility evidence, finite sides, scale, angle and prediction validity passed\n";
}
