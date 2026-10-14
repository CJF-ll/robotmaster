#include "detection_correspondence.hpp"
#include <limits>
#include <iostream>
using namespace outpost;
void require(bool ok){if(!ok)throw std::runtime_error("correspondence contract failed");}
int main(){try{
 Detection a{};a.measurement_points={cv::Point2f(10,10),{30,10},{30,30},{10,30}};
 auto b=a;for(auto& p:b.measurement_points)p.x+=2;
 auto r=detection_correspondence({b},{a},.2);require(r[0].traditional==0&&r[0].max_endpoint_error==2);
 // Display enclosure does not participate.
 for(auto& p:b.corners)p={999,999};require(detection_correspondence({b},{a},.2)[0].traditional==0);
 require(detection_correspondence({b},{a},.05)[0].eligible==0);
 require(detection_correspondence({a},{a,a},.2)[0].ambiguous);
 r=detection_correspondence({a,a},{a},.2);require(r[0].ambiguous&&r[1].ambiguous&&r[0].traditional==-1);
 b=a;b.measurement_points[0].x+=10;require(detection_correspondence({b},{a},.2)[0].traditional==-1);
 b=a;b.measurement_points[0].x=std::numeric_limits<float>::quiet_NaN();require(detection_correspondence({b},{a},.2)[0].eligible==0);
 require(detection_correspondence({a},{},.2)[0].traditional==-1);
 require(detection_correspondence({},{a},.2).empty());
 bool rejected=false;try{detection_correspondence({a},{a},0);}catch(...){rejected=true;}require(rejected);
 std::cout<<"correspondence contracts passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
