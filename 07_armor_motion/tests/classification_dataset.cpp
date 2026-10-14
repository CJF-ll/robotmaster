#include "core.hpp"
#include "classifier_contract.hpp"
#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <iostream>
using namespace outpost;
struct Split {
 long train_end,validation_begin;
 void validate()const{if(train_end<0||validation_begin<=train_end+1)throw std::runtime_error("split requires a nonempty temporal gap");}
 std::string name(long frame)const{return frame<=train_end?"train":frame<validation_begin?"excluded_gap":"validation";}
};
struct Proposal{int index;std::array<cv::Point2f,4> points;};
void require(bool ok){if(!ok)throw std::runtime_error("dataset contract failed");}
void self_test(){
 Split s{599,660};s.validate();require(s.name(599)=="train"&&s.name(600)=="excluded_gap"&&s.name(659)=="excluded_gap"&&s.name(660)=="validation");
 bool rejected=false;try{Split{599,600}.validate();}catch(...){rejected=true;}require(rejected);
 cv::Mat image(100,100,CV_8UC3,cv::Scalar(1,2,3));
 std::array<cv::Point2f,4> points{cv::Point2f(20,30),{40,30},{40,50},{20,50}};
 auto p=classifier_pattern(image,points,1.125);require(!p.image.empty()&&!p.clipped);
 require(classifier_input(p.image).size()==cv::Size(32,32));
 YAML::Node sample;sample["reviewed_by_human"]=false;sample["human_label"]=YAML::Null;
 require(!sample["reviewed_by_human"].as<bool>()&&sample["human_label"].IsNull());
 std::cout<<"classification dataset contracts passed\n";
}
int main(int argc,char**argv){try{
 if(argc==2&&std::string(argv[1])=="--self-test"){self_test();return 0;}
 if(argc!=2)throw std::runtime_error("usage: classification_dataset dataset.yaml");
 auto y=YAML::LoadFile(argv[1]);auto config=Config::load(y["video_config"].as<std::string>());
 if(config.geometry.undistort)throw std::runtime_error("dataset requires raw coordinate inputs");
 Split split{y["train_end_frame"].as<long>(),y["validation_begin_frame"].as<long>()};split.validate();
 double scale=y["crop_axis_half_scale"].as<double>();
 if(!std::isfinite(scale)||scale<=0)throw std::runtime_error("invalid crop scale");
 const auto root=std::filesystem::path(y["output_dir"].as<std::string>());
 if(root.empty()||std::filesystem::exists(root))throw std::runtime_error("output exists or empty");
 const std::string candidate_path=y["candidates_csv"].as<std::string>();
 std::ifstream input(candidate_path);std::string line;
 if(!std::getline(input,line)||line!="frame,index,raw_class,confidence,cx,cy,tlx,tly,trx,try,brx,bry,blx,bly")throw std::runtime_error("invalid candidates header");
 std::map<long,std::vector<Proposal>> proposals;size_t expected=0;
 while(std::getline(input,line)){
   std::stringstream stream(line);std::vector<std::string> fields;std::string field;
   while(std::getline(stream,field,','))fields.push_back(field);
   if(fields.size()!=14)throw std::runtime_error("bad candidate row");
   const long frame=std::stol(fields[0]);Proposal p{};p.index=std::stoi(fields[1]);
   if(frame<0||p.index<0)throw std::runtime_error("negative proposal index");
   for(auto& previous:proposals[frame])if(previous.index==p.index)throw std::runtime_error("duplicate proposal");
   for(int i=0;i<4;++i){p.points[i]={std::stof(fields[6+2*i]),std::stof(fields[7+2*i])};
     if(!std::isfinite(p.points[i].x)||!std::isfinite(p.points[i].y))throw std::runtime_error("nonfinite proposal");}
   proposals[frame].push_back(p);++expected;
 }
 if(!expected)throw std::runtime_error("no proposals");
 if(!y["prediction_evidence"].IsSequence())throw std::runtime_error("prediction evidence must be a sequence");
 for(auto path:y["prediction_evidence"])if(!std::filesystem::is_regular_file(path.as<std::string>()))throw std::runtime_error("missing prediction evidence");
 std::filesystem::create_directories(root/"images");
 YAML::Node manifest;manifest["schema_version"]=1;manifest["purpose"]="classification_review";
 manifest["reviewed_by_human"]=false;manifest["video_path"]=config.input;manifest["coordinate_space"]="raw";
 manifest["proposal_source"]=candidate_path;manifest["crop_axis_half_scale"]=scale;
 manifest["split"]["train_end_frame"]=split.train_end;manifest["split"]["validation_begin_frame"]=split.validation_begin;
 manifest["split"]["independent_test_set"]=false;
 manifest["allowed_labels"]=std::vector<std::string>{"one","two","three","four","five","sentry","outpost","base","not_armor","uncertain"};
 manifest["prediction_evidence"]=y["prediction_evidence"];
 std::map<std::string,int> counts;size_t visited=0,valid=0;long last=-1;
 VideoSource source(config);Frame frame;
 while(source.next(frame)){
   last=frame.index;manifest["width"]=frame.image.cols;manifest["height"]=frame.image.rows;
   auto found=proposals.find(frame.index);if(found==proposals.end())continue;
   for(const auto& p:found->second){++visited;const auto partition=split.name(frame.index);++counts[partition];
     for(auto q:p.points)if(q.x<0||q.y<0||q.x>=frame.image.cols||q.y>=frame.image.rows)throw std::runtime_error("proposal point outside source image");
     auto pattern=classifier_pattern(frame.image,p.points,scale);YAML::Node row;
     const auto id="frame_"+std::to_string(frame.index)+"_"+std::to_string(p.index);
     row["sample_id"]=id;row["frame"]=frame.index;row["candidate_index"]=p.index;row["split"]=partition;
     row["reviewed_by_human"]=false;row["human_label"]=YAML::Null;row["notes"]="";
     for(auto q:p.points)row["proposal_points"].push_back(std::vector<float>{q.x,q.y});
     const bool usable=!pattern.image.empty()&&!pattern.clipped;row["crop_usable"]=usable;
     // Store a raw contextual patch; do not paint predictions onto label pixels.
     const auto bounds=cv::boundingRect(std::vector<cv::Point2f>(p.points.begin(),p.points.end()));
     const int padding=std::max(bounds.width,bounds.height);
     const auto roi=(cv::Rect(bounds.x-padding,bounds.y-padding,bounds.width+2*padding,bounds.height+2*padding)&cv::Rect(0,0,frame.image.cols,frame.image.rows));
     if(roi.empty())throw std::runtime_error("proposal outside source image");
     row["context_origin"]=std::vector<int>{roi.x,roi.y};
     row["context_path"]="images/"+id+"_context.png";
     if(!cv::imwrite((root/row["context_path"].as<std::string>()).string(),frame.image(roi)))throw std::runtime_error("context write failed");
     if(usable){++valid;row["crop_path"]="images/"+id+"_crop.png";row["input_path"]="images/"+id+"_input.png";
       cv::Mat pixels;classifier_input(pattern.image).convertTo(pixels,CV_8U,255);
       if(!cv::imwrite((root/row["crop_path"].as<std::string>()).string(),pattern.image)||
          !cv::imwrite((root/row["input_path"].as<std::string>()).string(),pixels))throw std::runtime_error("crop write failed");
     }
     manifest["samples"].push_back(row);
   }
 }
 if(visited!=expected||last<split.validation_begin)throw std::runtime_error("incomplete dataset/video split");
 manifest["export_complete"]=true;manifest["candidate_count"]=visited;
 std::ofstream out(root/"review.yaml");out<<manifest;out.flush();if(!out)throw std::runtime_error("manifest write failed");
 std::cout<<"candidates="<<visited<<" usable="<<valid;
 for(auto [name,n]:counts)std::cout<<' '<<name<<'='<<n;
 std::cout<<"\nUnreviewed proposal dataset, NOT ground truth or an independent test set.\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
