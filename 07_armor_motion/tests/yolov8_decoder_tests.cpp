#include "yolov8_decoder.hpp"
#include <limits>
using namespace outpost;
void check(bool ok){if(!ok)throw std::runtime_error("neural decoder contract failed");}
int main(){
  std::vector<float> tensor{50,50,40,20,.8f,.1f,30,40,70,40,70,60,30,60};
  auto out=decode_yolov8(tensor.data(),14,1,{200,100},{100,50},.7,.3);
  // This first fixture lies outside resized image height and must be rejected.
  check(out.empty());
  out=decode_yolov8(tensor.data(),14,1,{200,200},{100,100},.7,.3);
  check(out.size()==1&&out[0].detection.center==cv::Point2f(100,100)&&out[0].raw_class==0);
  auto duplicate=tensor;std::vector<float> two(28);
  for(int k=0;k<14;++k){two[2*k]=tensor[k];two[2*k+1]=duplicate[k];}
  check(decode_yolov8(two.data(),14,2,{200,200},{100,100},.7,.3).size()==1);
  tensor[4]=.2;check(decode_yolov8(tensor.data(),14,1,{200,200},{100,100},.7,.3).empty());
  tensor[4]=.8;tensor[6]=std::numeric_limits<float>::quiet_NaN();
  check(decode_yolov8(tensor.data(),14,1,{200,200},{100,100},.7,.3).empty());
  bool rejected=false;try{decode_yolov8(two.data(),13,2,{200,200},{100,100},.7,.3);}catch(...){rejected=true;}
  check(rejected);
  check(std::abs(neural_box_iou({0,0,10,10},{5,0,10,10})-1./3)<1e-12);
  check(neural_box_iou({0,0,10,10},{10,0,10,10})==0);
  check(neural_box_iou({0,0,0,10},{0,0,10,10})==0);
  check(neural_box_iou({0,0,INFINITY,10},{0,0,10,10})==0);
  const float huge=std::numeric_limits<float>::max();
  check(neural_box_iou({0,0,huge,huge},{0,0,huge,huge})==1);
  // Finite raw widths can become infinite after coordinate rescaling.
  auto overflow=duplicate;overflow[2]=huge;
  check(decode_yolov8(overflow.data(),14,1,{200,200},{100,100},.7,.3).empty());
  // Finite boxes with overflowing float area must still suppress duplicates.
  for(int k=0;k<14;++k){two[2*k]=duplicate[k];two[2*k+1]=duplicate[k];}
  two[4]=two[5]=two[6]=two[7]=1e20f;
  check(decode_yolov8(two.data(),14,2,{100,100},{100,100},.7,.3).size()==1);
  // A bad high-confidence box must not suppress the remaining valid candidate.
  for(int k=0;k<14;++k){two[2*k]=overflow[k];two[2*k+1]=duplicate[k];}
  two[8]=.99f;
  check(decode_yolov8(two.data(),14,2,{200,200},{100,100},.7,.3).size()==1);
}
