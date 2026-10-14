#include "temporal_classification.hpp"
#include "detection_correspondence.hpp"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>
#include <map>
#include <filesystem>
#include <iostream>
using namespace outpost;
using Key=std::pair<long,int>;
std::vector<std::string> split(const std::string& line){std::vector<std::string> v;std::stringstream s(line);std::string x;while(std::getline(s,x,','))v.push_back(x);if(!line.empty()&&line.back()==',')v.emplace_back();return v;}
void require(bool ok){if(!ok)throw std::runtime_error("semantic test failed");}
void self_test(){
 require(split("1,2,,").size()==4);
 SemanticPolicy p;std::array<double,9> good{},weak{},bad{};good[6]=.95;good[8]=.05;weak[6]=.37;weak[8]=.63;bad[6]=.02;bad[8]=.98;
 SemanticHistory h;
 for(int i=0;i<3;++i){auto d=semantic_step(p,h,true,true,good);require(d.accepted);h=d.next;}
 auto d=semantic_step(p,h,true,true,weak);require(d.accepted&&!d.direct);
 require(!semantic_step(p,d.next,true,true,weak).accepted);
 require(!semantic_step(p,h,false,true,weak).accepted);
 require(!semantic_step(p,h,true,false,weak).accepted);
 require(!semantic_step(p,h,true,true,bad).accepted);
 auto invalid=good;invalid[0]=NAN;require(!semantic_step(p,h,true,true,invalid).accepted);
 auto reset=semantic_step(p,h,true,true,bad);require(!semantic_step(p,reset.next,true,true,weak).accepted);
 Detection a{};a.measurement_points={cv::Point2f(10,10),{30,10},{30,30},{10,30}};
 require(detection_correspondence({a},{a},.5)[0].traditional==0);
 require(detection_correspondence({a,a},{a},.5)[0].ambiguous);
 require(detection_correspondence({a},{a,a},.5)[0].ambiguous);
 auto shifted=a;for(auto& q:shifted.measurement_points)q.x+=100;
 require(detection_correspondence({shifted},{a},.5)[0].traditional<0);
 std::cout<<"semantic history contracts passed\n";
}
int main(int argc,char**argv){try{
 if(argc==2&&std::string(argv[1])=="--self-test"){self_test();return 0;}
 if(argc!=2)throw std::runtime_error("usage: temporal_classification_probe experiment.yaml");
 auto y=YAML::LoadFile(argv[1]);SemanticPolicy policy;
 policy.target=y["target_index"].as<int>();policy.confirm=y["confirm_frames"].as<int>();policy.max_hold=y["max_hold_frames"].as<int>();
 policy.accept=y["accept_threshold"].as<double>();policy.target_floor=y["target_floor"].as<double>();policy.negative_veto=y["negative_veto"].as<double>();policy.validate();
 const double fraction=y["endpoint_fraction"].as<double>();
 detection_correspondence({}, {},fraction);
 std::map<Key,std::array<double,9>> scores;std::map<Key,bool> support;
 std::map<long,std::vector<std::pair<int,Detection>>> frames;
 auto read=[&](const char* key,const std::string& header,auto consume){
   std::ifstream f(y[key].as<std::string>());std::string line;
   if(!std::getline(f,line)||line!=header)throw std::runtime_error(std::string("invalid CSV: ")+key);
   while(std::getline(f,line))consume(split(line));
 };
 read("probabilities_csv","frame,index,crop_width,crop_height,status,p0,p1,p2,p3,p4,p5,p6,p7,p8",[&](const auto& v){
   if(v.size()<5)throw std::runtime_error("bad probability row");
   Key k{std::stol(v[0]),std::stoi(v[1])};std::array<double,9> p;p.fill(NAN);
   if(v[4]=="classified"){if(v.size()!=14)throw std::runtime_error("missing scores");for(int i=0;i<9;++i)p[i]=std::stod(v[5+i]);}
   if(!scores.emplace(k,p).second)throw std::runtime_error("duplicate scores");
 });
 read("correspondence_csv","frame,neural_index,status,traditional_index,max_endpoint_error_px",[&](const auto& v){
   if(v.size()!=5)throw std::runtime_error("bad support row");
   if(!support.emplace(Key{std::stol(v[0]),std::stoi(v[1])},v[2]=="matched").second)throw std::runtime_error("duplicate support");
 });
 read("candidates_csv","frame,index,raw_class,confidence,cx,cy,tlx,tly,trx,try,brx,bry,blx,bly",[&](const auto& v){
   if(v.size()!=14)throw std::runtime_error("bad candidate row");long frame=std::stol(v[0]);int index=std::stoi(v[1]);
   if(frame<0||index<0)throw std::runtime_error("negative index");
   for(const auto& row:frames[frame])if(row.first==index)throw std::runtime_error("duplicate candidate");
   Detection d{};for(int i=0;i<4;++i)d.measurement_points[i]={std::stof(v[6+2*i]),std::stof(v[7+2*i])};
   frames[frame].push_back({index,d});
 });
 size_t total=0;for(const auto& f:frames)for(const auto& row:f.second){Key k{f.first,row.first};++total;if(!scores.count(k)||!support.count(k))throw std::runtime_error("missing joined row");}
 if(total!=scores.size()||total!=support.size())throw std::runtime_error("unjoined records");
 const auto path=y["output_csv"].as<std::string>();if(std::filesystem::exists(path))throw std::runtime_error("output exists");
 std::ofstream out(path);if(!out)throw std::runtime_error("cannot create output");
 out<<"frame,index,previous_index,continuous,measured_support,direct_accepted,experimental_accepted,reason\n";
 std::vector<Detection> previous;std::vector<int> ids;std::vector<SemanticHistory> histories;long previous_frame=-2;
 size_t direct=0,accepted=0,added=0;
 for(const auto& [frame,rows]:frames){
   std::vector<Detection> current;for(const auto& row:rows)current.push_back(row.second);
   const auto matches=detection_correspondence(current,previous,fraction);std::vector<SemanticHistory> next;std::vector<int> next_ids;
   for(size_t i=0;i<rows.size();++i){const auto j=matches[i].traditional;const bool continuous=frame==previous_frame+1&&j>=0;
     const Key k{frame,rows[i].first};auto decision=semantic_step(policy,continuous?histories[j]:SemanticHistory{},continuous,support.at(k),scores.at(k));
     const bool baseline=decision.direct&&support.at(k);direct+=baseline;accepted+=decision.accepted;added+=decision.accepted&&!baseline;
     out<<frame<<','<<rows[i].first<<','<<(continuous?ids[j]:-1)<<','<<continuous<<','<<support.at(k)<<','<<baseline<<','<<decision.accepted<<','<<decision.reason<<'\n';
     next.push_back(decision.next);next_ids.push_back(rows[i].first);
   }
   previous=current;histories=next;ids=next_ids;previous_frame=frame;
 }
 out.flush();if(!out)throw std::runtime_error("incomplete output");
 std::cout<<"candidates="<<total<<" direct="<<direct<<" experimental="<<accepted<<" added="<<added<<"\nDiagnostic only: no tracker updates; correspondence is not physical identity or truth.\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
