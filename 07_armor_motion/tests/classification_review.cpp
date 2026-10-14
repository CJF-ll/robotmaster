#include <yaml-cpp/yaml.h>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <map>
#include <chrono>
namespace fs=std::filesystem;
const std::set<std::string> classes{"one","two","three","four","five","sentry","outpost","base","not_armor","uncertain"};
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
struct Summary {int total=0,reviewed=0,pending=0,uncertain=0,gap=0,unusable=0,train=0,validation=0;};
std::string partition(long frame,long train_end,long validation_begin){
 return frame<=train_end?"train":frame<validation_begin?"excluded_gap":"validation";
}
fs::path image_path(const fs::path& root,const YAML::Node& n){
 require(n.IsScalar(),"missing image path");const fs::path relative(n.as<std::string>());
 require(!relative.empty()&&!relative.is_absolute(),"image path must be relative");
 for(const auto& part:relative)require(part!="..","image path escapes dataset");
 auto base=fs::canonical(root),path=fs::canonical(root/relative);
 auto suffix=path.lexically_relative(base);
 require(!suffix.empty()&&!suffix.is_absolute(),"invalid dataset path");
 for(const auto& part:suffix)require(part!="..","image symlink escapes dataset");
 require(fs::is_regular_file(path),"image missing");return path;
}
Summary validate(const YAML::Node& document,const fs::path& root,bool pixels){
 require(document["schema_version"].as<int>()==1&&document["purpose"].as<std::string>()=="classification_review","wrong review schema");
 require(document["export_complete"].as<bool>(false),"dataset export incomplete");
 require(document["coordinate_space"].as<std::string>()=="raw","unsupported coordinate space");
 require(!document["split"]["independent_test_set"].as<bool>(),"single-video dataset cannot claim independent test");
 const long train_end=document["split"]["train_end_frame"].as<long>(),validation_begin=document["split"]["validation_begin_frame"].as<long>();
 require(train_end>=0&&validation_begin>train_end+1,"invalid temporal split");
 const auto samples=document["samples"];require(samples.IsSequence()&&samples.size()>0,"no samples");
 require(samples.size()==document["candidate_count"].as<size_t>(),"sample count mismatch");
 Summary s;std::set<std::string> ids;std::set<std::pair<long,int>> keys;
 for(const auto& row:samples){++s.total;
   auto id=row["sample_id"].as<std::string>();const long frame=row["frame"].as<long>();const int index=row["candidate_index"].as<int>();
   require(frame>=0&&index>=0&&ids.insert(id).second&&keys.emplace(frame,index).second,"invalid/duplicate sample identity");
   require(id=="frame_"+std::to_string(frame)+"_"+std::to_string(index),"sample identity mismatch");
   auto split=row["split"].as<std::string>();require(split==partition(frame,train_end,validation_begin),"sample split changed");
   const bool usable=row["crop_usable"].as<bool>(),reviewed=row["reviewed_by_human"].as<bool>();
   if(split=="excluded_gap")++s.gap;if(!usable)++s.unusable;
   if(pixels){
     auto context=cv::imread(image_path(root,row["context_path"]).string(),cv::IMREAD_COLOR);
     require(!context.empty(),"unreadable context image");
     if(usable){auto crop=cv::imread(image_path(root,row["crop_path"]).string(),cv::IMREAD_COLOR);
       auto input=cv::imread(image_path(root,row["input_path"]).string(),cv::IMREAD_UNCHANGED);
       require(!crop.empty()&&input.type()==CV_8UC1&&input.size()==cv::Size(32,32),"invalid classifier images");}
   }
   if(!reviewed){require(!row["human_label"]||row["human_label"].IsNull(),"unreviewed sample has a label");
     if(usable&&split!="excluded_gap")++s.pending;continue;}
   ++s.reviewed;require(row["human_label"].IsScalar(),"reviewed label missing");
   auto label=row["human_label"].as<std::string>();require(classes.count(label),"unknown label");
   require(!row["reviewer"].as<std::string>("").empty()&&row["reviewed_at_unix"].as<long long>(0)>0,"review provenance missing");
   if(label=="uncertain"){++s.uncertain;continue;}
   if(!usable||split=="excluded_gap")continue;
   if(split=="train")++s.train;else ++s.validation;
 }
 require(!document["reviewed_by_human"].as<bool>(false)||s.reviewed==s.total,"global review flag inconsistent");
 return s;
}
void report(const Summary& s){std::cout<<"total="<<s.total<<" reviewed="<<s.reviewed<<" pending_eligible="<<s.pending<<" uncertain="<<s.uncertain<<" gap="<<s.gap<<" unusable="<<s.unusable<<" train_ready="<<s.train<<" validation_ready="<<s.validation<<'\n';}
std::string csv(const std::string& value){std::string result="\"";for(char c:value){if(c=='\"')result+='\"';result+=c;}return result+'\"';}
YAML::Node fixture(){
 YAML::Node d;d["schema_version"]=1;d["purpose"]="classification_review";d["coordinate_space"]="raw";d["export_complete"]=true;
 d["reviewed_by_human"]=false;d["candidate_count"]=3;d["split"]["train_end_frame"]=10;d["split"]["validation_begin_frame"]=20;d["split"]["independent_test_set"]=false;
 for(int f:{0,15,20}){YAML::Node r;r["sample_id"]="frame_"+std::to_string(f)+"_0";r["frame"]=f;r["candidate_index"]=0;r["split"]=partition(f,10,20);
   r["crop_usable"]=true;r["reviewed_by_human"]=false;r["human_label"]=YAML::Null;d["samples"].push_back(r);}
 return d;
}
void self_test(){
 auto d=fixture();auto s=validate(d,"",false);require(s.pending==2&&s.gap==1&&s.train==0,"pending accounting");
 auto rejects=[](const YAML::Node& n){try{validate(n,"",false);}catch(...){return true;}return false;};
 auto bad=YAML::Clone(d);bad["samples"][0]["human_label"]="outpost";require(rejects(bad),"pseudo-label accepted");
 bad=YAML::Clone(d);bad["samples"][0]["reviewed_by_human"]=true;bad["samples"][0]["human_label"]="outpost";require(rejects(bad),"missing reviewer accepted");
 bad=YAML::Clone(d);bad["samples"][1]["split"]="train";require(rejects(bad),"gap leaked into train");
 bad=YAML::Clone(d);bad["candidate_count"]=4;require(rejects(bad),"missing row accepted");
 for(int i:{0,2}){d["samples"][i]["reviewed_by_human"]=true;d["samples"][i]["human_label"]=i==0?"outpost":"uncertain";
   d["samples"][i]["reviewer"]="synthetic-test";d["samples"][i]["reviewed_at_unix"]=1;}
 s=validate(d,"",false);require(s.pending==0&&s.train==1&&s.validation==0&&s.uncertain==1,"uncertain entered export");
 bad=YAML::Clone(d);bad["samples"][0]["human_label"]="invented";require(rejects(bad),"unknown label accepted");
 std::cout<<"classification review contracts passed (synthetic labels only)\n";
}
int main(int argc,char**argv){try{
 if(argc==2&&std::string(argv[1])=="--self-test"){self_test();return 0;}
 require(argc>=3,"usage: classification_review --check review.yaml | --label review.yaml sample_id label reviewer new.yaml | --export review.yaml new_directory");
 const std::string mode=argv[1];require((mode=="--check"&&argc==3)||(mode=="--label"&&argc==7)||(mode=="--export"&&argc==4),"invalid mode/arguments");
 const auto source=fs::canonical(argv[2]),root=source.parent_path();auto d=YAML::LoadFile(source.string());auto s=validate(d,root,true);
 if(mode=="--check"){report(s);return 0;}
 if(mode=="--label"){
   const std::string id=argv[3],label=argv[4],reviewer=argv[5];const fs::path destination=fs::absolute(argv[6]);
   require(classes.count(label)&&!reviewer.empty(),"invalid label/reviewer");
   require(fs::canonical(destination.parent_path())==root&&!fs::exists(destination),"save a new manifest beside original; overwrite forbidden");
   bool found=false;for(auto row:d["samples"])if(row["sample_id"].as<std::string>()==id){
     require(!row["reviewed_by_human"].as<bool>(),"already reviewed; original decision preserved");
     row["human_label"]=label;row["reviewed_by_human"]=true;row["reviewer"]=reviewer;
     row["reviewed_at_unix"]=std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();found=true;}
   require(found,"sample not found");d["reviewed_by_human"]=false;
   d["parent_manifest"]=source.filename().string();s=validate(d,root,true);std::ofstream out(destination);out<<d;out.flush();require(bool(out),"cannot save reviewed copy");report(s);return 0;
 }
 report(s);require(s.pending==0,"unreviewed eligible samples: export blocked");
 require(s.train>0&&s.validation>0,"reviewed non-uncertain train and validation samples required");
 const fs::path destination=argv[3];require(!destination.empty()&&!fs::exists(destination),"export directory exists/empty");
 fs::create_directories(destination);std::ofstream train(destination/"train.csv"),validation(destination/"validation.csv");
 require(bool(train)&&bool(validation),"cannot create export");
 train<<"sample_id,input_path,label\n";validation<<"sample_id,input_path,label\n";
 for(const auto& r:d["samples"]){if(!r["crop_usable"].as<bool>()||!r["reviewed_by_human"].as<bool>()||r["split"].as<std::string>()=="excluded_gap")continue;
   auto label=r["human_label"].as<std::string>();if(label=="uncertain")continue;
   auto& out=r["split"].as<std::string>()=="train"?train:validation;
   out<<csv(r["sample_id"].as<std::string>())<<','<<csv(image_path(root,r["input_path"]).string())<<','<<csv(label)<<'\n';}
 train.flush();validation.flush();require(bool(train)&&bool(validation),"incomplete export");
 std::ofstream provenance(destination/"source.txt");provenance<<source.string()<<"\nSingle previously inspected video; not an independent test set.\n";provenance.flush();require(bool(provenance),"provenance write failed");
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
