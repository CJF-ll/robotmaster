#include "prediction_fit.hpp"
#include <stdexcept>
using namespace outpost;
void require(bool ok){if(!ok)throw std::runtime_error("stationary evidence contract failed");}
int main(){
  std::vector<TimedCenter> h{{0,{0,0}},{.04,{.1f,.1f}},{.08,{0,0}}};
  require(stationary_history(h,.07,.75));
  require(!stationary_history(h,.1,.75));
  h={{0,{0,0}},{.04,{2,0}},{.08,{4,0}}};
  require(!stationary_history(h,.07,.75));
  h={{0,{0,0}},{.04,{2,0}},{.08,{2,0}}};
  require(!stationary_history(h,.07,.75));
  h={{0,{0,0}},{.04,{2,0}},{.08,{0,0}}};
  require(!stationary_history(h,.07,.75));
  h={{0,{0,0}},{.08,{0,0}}};
  require(!stationary_history(h,.07,.75));
  h={{0,{0,0}},{.02,{0,0}},{.04,{0,0}},{.06,{0,0}},{.08,{0,0}}};
  require(stationary_history(h,.07,.75));
  h.back().time=h[h.size()-2].time;
  require(!stationary_history(h,.07,.75));
  require(!stationary_history(h,-1,.75));
}
