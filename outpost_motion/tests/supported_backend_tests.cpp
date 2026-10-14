#include "supported_observations.hpp"
#include <iostream>
using namespace outpost;
void require(bool value){if(!value)throw std::runtime_error("supported backend contract failed");}
int main(){try{
 DetectorConfig config;Detection observed{};
 observed.measurement_points={cv::Point2f(10,10),{30,10},{30,30},{10,30}};
 observed.corners=observed.measurement_points;observed.center={20,20};observed.score=2;observed.left=0;observed.right=1;
 DetectionFrame frame;frame.observations={observed};frame.pairs={observed};frame.lamps.resize(2);
 frame.pair_status={Detector::PairStatus::Accepted};
 auto network=observed;network.kind=ObservationKind::NeuralKeypoints;network.detection_confidence=.8;
 network.classified_index=6;network.classification_confidence=.9;
 for(auto& p:network.measurement_points)p.x+=1;
 auto result=supported_observations(frame,{network},config);
 require(result.observations.size()==1&&result.reviews.size()==1);
 const auto& accepted=result.observations[0];
 require(accepted.kind==ObservationKind::NeuralSupportedLampPair&&accepted.measurement_points==observed.measurement_points&&
   accepted.corners==observed.corners&&accepted.center==observed.center&&accepted.score==2&&accepted.classification_confidence==.9);
 frame.observations.clear();result=supported_observations(frame,{network},config);
 require(result.observations.empty()&&result.reviews[0].decision=="no_lamp_support"&&result.lamps.size()==2&&result.pairs.size()==1);
 frame.observations={observed};network.classified_index=8;result=supported_observations(frame,{network},config);
 require(result.observations.empty()&&result.reviews[0].decision=="class_rejected");
 network.classified_index=6;network.classification_confidence=.1;require(supported_observations(frame,{network},config).observations.empty());
 network.classification_confidence=.9;result=supported_observations(frame,{network,network},config);
 require(result.observations.empty()&&result.reviews.size()==2&&result.reviews[0].decision=="ambiguous_lamp_support");
 frame.observations.push_back(observed);require(supported_observations(frame,{network},config).observations.empty());
 frame.observations={observed};network.classification_confidence=NAN;require(supported_observations(frame,{network},config).observations.empty());
 std::cout<<"supported backend contracts passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
