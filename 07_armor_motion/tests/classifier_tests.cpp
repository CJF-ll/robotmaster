#include "classifier_contract.hpp"
#include <iostream>
using namespace outpost;
void require(bool v){if(!v)throw std::runtime_error("classifier contract failed");}
int main(){try{
 cv::Mat image(100,100,CV_8UC3,cv::Scalar(255,255,255));
 std::array<cv::Point2f,4> p{cv::Point2f(20,30),{40,30},{40,50},{20,50}};
 auto pattern=classifier_pattern(image,p,1.125);require(!pattern.clipped&&pattern.image.rows==45&&pattern.image.cols==20);
 auto input=classifier_input(pattern.image);require(input.rows==32&&input.cols==32&&input.type()==CV_32F);
 require(input.at<float>(0,0)==1&&input.at<float>(0,31)==0);
 for(auto& q:p)q.y-=25;require(classifier_pattern(image,p,1.125).clipped);
 std::swap(p[1],p[3]);require(classifier_pattern(image,p,1.125).image.empty());
 // A valid geometric crop may be too thin for reference 32x32 letterboxing.
 std::array<cv::Point2f,4> thin{cv::Point2f(20,20),{21,20},{21,80},{20,80}};
 require(classifier_pattern(image,thin,.5).image.empty());
 std::array<cv::Point2f,4> flat{cv::Point2f(20,20),{80,20},{80,21},{20,21}};
 require(classifier_pattern(image,flat,.5).image.empty());
 std::array<cv::Point2f,4> boundary{cv::Point2f(20,20),{21,20},{21,52},{20,52}};
 auto edge=classifier_pattern(image,boundary,.5);require(!edge.image.empty());
 require(classifier_input(edge.image).cols==32);
 // Rejection must not poison processing of the next ordinary candidate.
 require(!classifier_pattern(image,{cv::Point2f(20,30),{40,30},{40,50},{20,50}},1.125).image.empty());
 float logits[9]={1000,0,0,0,0,0,0,0,0};auto probabilities=classifier_probabilities(logits);require(probabilities[0]==1);
 for(auto& x:logits)x=0;probabilities=classifier_probabilities(logits);require(std::abs(probabilities[0]-1./9)<1e-12);
 bool rejected=false;logits[0]=NAN;try{classifier_probabilities(logits);}catch(...){rejected=true;}require(rejected);
 std::cout<<"classifier preprocessing contracts passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
