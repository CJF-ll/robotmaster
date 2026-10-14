#include "run_output.hpp"
#include <iostream>
#include <unistd.h>
void require(bool ok){if(!ok)throw std::runtime_error("output contract failed");}
int main(){try{
 char pattern[]="/tmp/outpost-output-tests-XXXXXX";auto name=mkdtemp(pattern);require(name);
 const std::filesystem::path root(name);
 struct Cleanup{std::filesystem::path p;~Cleanup(){std::error_code e;std::filesystem::remove_all(p,e);}} cleanup{root};
 outpost::RunOutput first({},root),second({},root);require(first.root()!=second.root());
 {std::ofstream f(first.path("keep.txt"));f<<"original";}
 bool rejected=false;try{outpost::RunOutput again(first.root());}catch(...){rejected=true;}require(rejected);
 std::string content;{std::ifstream f(first.path("keep.txt"));f>>content;}require(content=="original");
 for(auto bad:{"../escape","/tmp/escape",".","..",""}){rejected=false;try{first.path(bad);}catch(...){rejected=true;}require(rejected);}
 first.finish(925,false,false);second.finish(12,true,true);
 auto read=[](const std::string& p){std::ifstream f(p);return std::string(std::istreambuf_iterator<char>(f),{});};
 require(read(first.path("run_status.txt")).find("state=completed")!=std::string::npos);
 require(read(second.path("run_status.txt")).find("state=interrupted")!=std::string::npos);
 outpost::RunOutput empty(root/"empty");empty.finish(0,false,false);
 require(read(empty.path("run_status.txt")).find("state=empty_input")!=std::string::npos);
 std::cout<<"isolated output and completion contracts passed\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
