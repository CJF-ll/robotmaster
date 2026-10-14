#include "prediction_fit.hpp"
#include <stdexcept>
#include <iostream>
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){
  using namespace outpost;
  for(double fps:{15.,30.,60.}){
    std::vector<TimedCenter> samples;
    for(int i=0;i<9;++i){double t=i/fps;samples.push_back({10000+t,{float(100+20*t+30*t*t),float(50-10*t)}});}
    const double target=8/fps+.1;
    auto result=quadratic_forecast(samples,.1);
    require(result&&cv::norm(*result-cv::Point2f(float(100+20*target+30*target*target),float(50-10*target)))<.002,
            "quadratic forecast fails exact acceleration or absolute timestamp invariance");
    require(!quadratic_forecast(samples,-1),"negative horizon accepted");
    samples[3].time=samples[2].time;
    require(!quadratic_forecast(samples,.1),"duplicate sample timestamp accepted");
  }
  require(!quadratic_forecast({},.1),"empty history accepted");
  require(!linear_forecast({},.1),"linear empty history accepted");
  for(double fps:{15.,30.,60.}){
    std::vector<TimedCenter> samples;
    for(int i=0;i<7;++i){const double t=(i+.1*(i%2))/fps;
      samples.push_back({10000+t,{float(100-200*t),float(20+30*t)}});}
    const double target=samples.back().time-10000+.1;
    auto p=linear_forecast(samples,.1);
    require(p&&cv::norm(*p-cv::Point2f(float(100-200*target),float(20+30*target)))<.002,
            "linear fit fails irregular timing/negative velocity");
    require(!linear_forecast(samples,-1),"linear negative horizon accepted");
    samples[2].time=samples[1].time;
    require(!linear_forecast(samples,.1),"linear duplicate timestamp accepted");
  }
  std::vector<TimedCenter> repeated;
  for(int i=0;i<5;++i)repeated.push_back({i/30.,{float(100-5*i),50}});
  repeated.back().center=repeated[3].center;
  const auto repeated_prediction=linear_forecast(repeated,.1);
  require(repeated_prediction&&repeated_prediction->x<85,"one repeated observation erased multi-frame velocity");
  std::vector<TimedCenter> still;
  for(int i=0;i<9;++i)still.push_back({i/30.,{100,50}});
  require(cv::norm(*linear_forecast(still,.1)-cv::Point2f(100,50))<.001,"linear stationary target moved");
  require(cv::norm(*quadratic_forecast(still,.1)-cv::Point2f(100,50))<.001,"stationary target moved");
  std::cout<<"diagnostic fit contracts passed\n";
}
