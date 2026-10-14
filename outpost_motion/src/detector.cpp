#include "detector.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <tuple>
#include <stdexcept>
namespace outpost {
namespace {
bool gray_sample(const cv::Mat& gray,cv::Point2f p,double& value){
  if(p.x<0||p.y<0||p.x>=gray.cols-1||p.y>=gray.rows-1)return false;
  const int x=int(p.x),y=int(p.y);const double dx=p.x-x,dy=p.y-y;
  value=(1-dy)*((1-dx)*gray.at<uchar>(y,x)+dx*gray.at<uchar>(y,x+1))+
        dy*((1-dx)*gray.at<uchar>(y+1,x)+dx*gray.at<uchar>(y+1,x+1));
  return true;
}
void refine_endpoints(const cv::Mat& gray,const DetectorConfig& config,Lightbar& bar){
  bar.endpoint_evidence=EndpointEvidence::Uncertain;
  const cv::Point2f normal(bar.axis.y,-bar.axis.x);
  const double half=.5*bar.length;
  const double extent=half+std::max(2.,bar.length*config.endpoint_search_fraction);
  // Quarter-pixel sampling is interpolation, not a claim of quarter-pixel accuracy.
  const int intervals=int(std::ceil(2*extent/.25));const double step=2*extent/intervals;
  const double span=std::min(1.5,.25*bar.width);
  std::array<double,3> tops{},bottoms{},contrasts{};
  for(int line=0;line<3;++line){
    std::vector<double> profile(size_t(intervals+1));
    for(int k=0;k<=intervals;++k){
      const auto point=bar.center+normal*float((line-1)*span)+bar.axis*float(-extent+k*step);
      if(!gray_sample(gray,point,profile[size_t(k)])){
        bar.endpoint_evidence=EndpointEvidence::BoundaryUnsupported;return;
      }
    }
    // Demand actual dark support on BOTH ends of the local search. A crop or
    // continuing bright streak is not permission to extrapolate an endpoint.
    const double background=std::max((profile[0]+profile[1]+profile[2])/3,
      (profile[size_t(intervals)]+profile[size_t(intervals-1)]+profile[size_t(intervals-2)])/3);
    std::vector<double> core;
    for(int k=0;k<=intervals;++k)if(std::abs(-extent+k*step)<=.25*bar.length)
      core.push_back(profile[size_t(k)]);
    if(core.empty())return;
    std::sort(core.begin(),core.end());const double peak=core[core.size()/2];
    const double contrast=peak-background;contrasts[size_t(line)]=contrast;
    if(contrast<config.endpoint_min_contrast)return;
    const double level=background+.5*contrast;
    int first=-1,last=-1,runs=0;
    bool bright=false;
    for(int k=0;k<=intervals;++k){const bool now=profile[size_t(k)]>=level;
      if(now&&!bright){++runs;if(first<0)first=k;}
      if(now)last=k;
      bright=now;
    }
    if(runs!=1||first<=0||last>=intervals||first>intervals/2||last<intervals/2)return;
    const auto crossing=[&](int a,int b){
      const double u=(level-profile[size_t(a)])/(profile[size_t(b)]-profile[size_t(a)]);
      return -extent+(a+u*(b-a))*step;
    };
    tops[size_t(line)]=crossing(first-1,first);
    bottoms[size_t(line)]=crossing(last,last+1);
    // A contrast run entirely within the binary component's central half is
    // inconsistent with a single complete measured lamp, even if rectangular.
    if(tops[size_t(line)]>-.25*bar.length||bottoms[size_t(line)]<.25*bar.length)return;
  }
  std::sort(tops.begin(),tops.end());std::sort(bottoms.begin(),bottoms.end());
  bar.endpoint_spread_px=std::max(tops[2]-tops[0],bottoms[2]-bottoms[0]);
  bar.endpoint_contrast=*std::min_element(contrasts.begin(),contrasts.end());
  if(bar.endpoint_spread_px>config.endpoint_max_spread_px)return;
  const double length=bottoms[1]-tops[1];
  if(length<config.min_length||length>config.max_length)return;
  bar.center+=bar.axis*float(.5*(tops[1]+bottoms[1]));bar.length=float(length);
  const auto top=bar.center-bar.axis*bar.length*.5f,bottom=bar.center+bar.axis*bar.length*.5f;
  bar.vertices={top-normal*bar.width*.5f,top+normal*bar.width*.5f,
                bottom+normal*bar.width*.5f,bottom-normal*bar.width*.5f};
  bar.endpoint_evidence=EndpointEvidence::ContrastSupported;bar.endpoint_refined=true;
}
void describe_contour_quality(const std::vector<cv::Point>& contour,Lightbar& bar) {
  // Use distance to the nearest fitted rectangle boundary. Distance to the
  // centerline would confuse an ordinary wide lamp with an inaccurate fit.
  const cv::Point2d axis(bar.axis),normal(axis.y,-axis.x);
  if(bar.low_resolution){
    bar.quality_available=true;bar.fit_residual_px=0;
    bar.center_variance_scale={1+normal.x*normal.x,1+normal.y*normal.y};
    return;
  }
  double along_sq=0,across_sq=0;size_t along_count=0,across_count=0;
  for(const auto& point:contour){
    const cv::Point2d delta=cv::Point2d(point)-cv::Point2d(bar.center);
    const double along=std::abs(std::abs(delta.dot(axis))-.5*bar.length);
    const double across=std::abs(std::abs(delta.dot(normal))-.5*bar.width);
    if(along<across){along_sq+=along*along;++along_count;}
    else {across_sq+=across*across;++across_count;}
  }
  if(contour.empty())return;
  bar.quality_available=true;
  bar.fit_residual_px=std::sqrt((along_sq+across_sq)/contour.size());
  const double va=along_count?along_sq/along_count:0;
  // A one-column lamp has unresolved transverse extent. Retain its explicit
  // one-pixel support uncertainty in addition to its boundary-fit residual.
  const double vn=across_count?across_sq/across_count:0;
  // One pixel squared is the reference scale. Residual samples share a contour,
  // so do not shrink uncertainty by the number of contour points.
  bar.center_variance_scale={1+va*axis.x*axis.x+vn*normal.x*normal.x,
                             1+va*axis.y*axis.y+vn*normal.y*normal.y};
}
}
bool enclose(const Lightbar& a, const Lightbar& b, const DetectorConfig& c, Detection& d) {
  // Failed grayscale refinement does not invalidate the original measured
  // binary contour. It remains explicitly uncertain and unrefined. A missing
  // image/ROI boundary is different: do not invent its unseen endpoint.
  if(c.endpoint_refinement_enabled&&(a.endpoint_evidence==EndpointEvidence::BoundaryUnsupported||
                                    b.endpoint_evidence==EndpointEvidence::BoundaryUnsupported))return false;
  for(const auto* lamp:{&a,&b}){
    if(!std::isfinite(lamp->center.x)||!std::isfinite(lamp->center.y)||
       !std::isfinite(lamp->length)||!std::isfinite(lamp->width)||
       !std::isfinite(lamp->axis.x)||!std::isfinite(lamp->axis.y)||
       std::abs(cv::norm(lamp->axis)-1)>1e-3||lamp->axis.y<=0)return false;
    for(auto p:lamp->vertices)if(!std::isfinite(p.x)||!std::isfinite(p.y))return false;
  }
  if (a.center.x >= b.center.x || a.length <= 0 || b.length <= 0 ||
      a.width <= 0 || b.width <= 0) return false;
  const auto delta = b.center-a.center;
  const double length = (a.length+b.length)/2;
  if (std::max(a.length,b.length)/std::min(a.length,b.length)>c.max_length_ratio ||
      delta.x/length<c.min_separation || delta.x/length>c.max_separation ||
      std::abs(delta.y)/length>c.max_vertical_offset) return false;
  const double cosine = std::clamp(double(a.axis.dot(b.axis)), -1., 1.);
  if (std::acos(cosine)*180/CV_PI>c.max_angle_diff_deg) return false;
  // Independent side support lines parallel to each measured lamp, plus
  // upper/lower support lines aligned with measured endpoint pairs. These are
  // display geometry, not PnP keypoints or independently extrapolated corners.
  cv::Point2f na(a.axis.y,-a.axis.x), nb(b.axis.y,-b.axis.x);
  if (na.x<0) na=-na;
  if (nb.x<0) nb=-nb;
  if (na.x<.1 || nb.x<.1) return false;
  double lo=std::numeric_limits<double>::infinity(), hi=-lo;
  const auto lt=a.center-a.axis*a.length*.5f,rt=b.center-b.axis*b.length*.5f;
  const auto lb=a.center+a.axis*a.length*.5f,rb=b.center+b.axis*b.length*.5f;
  auto positive_y_normal=[](cv::Point2f edge){
    const double norm=cv::norm(edge);
    if(norm<1e-6)return cv::Point2f(0,0);
    return cv::Point2f(float(-edge.y/norm),float(edge.x/norm));
  };
  const auto nt=positive_y_normal(rt-lt),nbottom=positive_y_normal(rb-lb);
  if(nt.y<=0||nbottom.y<=0)return false;
  double top=lo,bottom=-lo;
  for (const auto* bar : {&a,&b}) for (const auto& p : bar->vertices) {
    lo=std::min(lo,double(na.dot(p)));
    hi=std::max(hi,double(nb.dot(p)));
    top=std::min(top,double(nt.dot(p)));
    bottom=std::max(bottom,double(nbottom.dot(p)));
  }
  const double side=std::max(c.side_margin_min_px,c.side_margin_ratio*(a.width+b.width)/2);
  lo-=side; hi+=side;
  const double end=std::max(c.end_margin_min_px,c.end_margin_ratio*length);
  top-=end; bottom+=end;
  auto intersect=[](cv::Point2f n,double k,cv::Point2f m,double j,cv::Point2f& p) {
    const double determinant=double(n.x)*m.y-double(n.y)*m.x;
    if(std::abs(determinant)<1e-6)return false;
    p={float((k*m.y-double(n.y)*j)/determinant),
       float((double(n.x)*j-k*m.x)/determinant)};
    return std::isfinite(p.x)&&std::isfinite(p.y);
  };
  if(!intersect(na,lo,nt,top,d.corners[0])||!intersect(nb,hi,nt,top,d.corners[1])||
     !intersect(nb,hi,nbottom,bottom,d.corners[2])||
     !intersect(na,lo,nbottom,bottom,d.corners[3]))return false;
  std::vector<cv::Point2f> poly(d.corners.begin(),d.corners.end());
  if (!cv::isContourConvex(poly) || cv::contourArea(poly)<4 ||
      d.corners[0].x>=d.corners[1].x || d.corners[3].x>=d.corners[2].x) return false;
  for (const auto* bar : {&a,&b}) for (const auto& p : bar->vertices)
    if (cv::pointPolygonTest(poly,p,false)<0) return false;
  d.center=(a.center+b.center)*.5f;
  d.endpoint_evidence={a.endpoint_evidence,b.endpoint_evidence};
  d.endpoint_refined=a.endpoint_refined&&b.endpoint_refined;
  d.endpoint_min_contrast=std::min(a.endpoint_contrast,b.endpoint_contrast);
  d.endpoint_max_spread_px=std::max(a.endpoint_spread_px,b.endpoint_spread_px);
  d.low_resolution=a.low_resolution||b.low_resolution;
  d.quality_available=a.quality_available&&b.quality_available;
  d.fit_residual_px=d.quality_available?std::sqrt(
      .5*(a.fit_residual_px*a.fit_residual_px+b.fit_residual_px*b.fit_residual_px)):0;
  d.center_variance_scale={1,1};
  if(d.quality_available)for(int axis=0;axis<2;++axis)
    d.center_variance_scale[axis]=.5*(a.center_variance_scale[axis]+b.center_variance_scale[axis]);
  d.measurement_points={a.center-a.axis*a.length*.5f,b.center-b.axis*b.length*.5f,
                        b.center+b.axis*b.length*.5f,a.center+a.axis*a.length*.5f};
  d.score=std::abs(std::log(a.length/b.length)) + std::abs(delta.y)/length;
  return true;
}
std::vector<Detection> Detector::detect(const cv::Mat& image) {
  CV_Assert(image.type()==CV_8UC3);
  cv::Mat gray;
  if(config_.endpoint_refinement_enabled){
    if(!std::isfinite(config_.endpoint_search_fraction)||config_.endpoint_search_fraction<=0||config_.endpoint_search_fraction>1||
       !std::isfinite(config_.endpoint_min_contrast)||config_.endpoint_min_contrast<=0||config_.endpoint_min_contrast>255||
       !std::isfinite(config_.endpoint_max_spread_px)||config_.endpoint_max_spread_px<=0||config_.endpoint_max_spread_px>5)
      throw std::runtime_error("invalid endpoint refinement configuration");
    cv::cvtColor(image,gray,cv::COLOR_BGR2GRAY);
  }
  cv::Mat mask(image.size(),CV_8UC1,cv::Scalar(0));
  for(int y=0;y<image.rows;++y) {
    const auto* src=image.ptr<cv::Vec3b>(y); auto* dst=mask.ptr<uchar>(y);
    for(int x=0;x<image.cols;++x)
      dst[x]=(src[x][2]>=config_.brightness &&
              int(src[x][2])-src[x][0]>=config_.red_difference)?255:0;
  }
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
  bars_.clear();
  border_rejected_contours_=0;
  thin_accepted_lamps_=0;
  for(const auto& contour:contours) {
    if(contour.size()<4)continue;
    const auto box=cv::boundingRect(contour);
    const double area=cv::contourArea(contour);
    const bool thin=config_.thin_vertical_enabled&&area==0&&box.width==1&&
                    box.height>=config_.thin_min_pixels&&cv::countNonZero(mask(box))==box.height;
    if(area<config_.min_area&&!thin)continue;
    // An image-boundary contour does not establish a complete lamp length.
    // Reject before fitting and before it can support a coasting track.
    const int margin=config_.border_margin_px;
    if(config_.reject_border_contours&&
       (box.x<=margin||box.y<=margin||
        image.cols-(box.x+box.width)<=margin||image.rows-(box.y+box.height)<=margin)){
      ++border_rejected_contours_;continue;
    }
    auto rect=cv::minAreaRect(contour);
    // A one-column component has zero polygon area, but nonzero pixel support.
    // Model that support only; do not claim to recover the unresolved true width.
    if(thin)rect=cv::RotatedRect(cv::Point2f(float(box.x),float(box.y+(box.height-1)*.5)),
                                cv::Size2f(1,float(box.height)),0);
    Lightbar bar;bar.low_resolution=thin;rect.points(bar.vertices.data()); bar.center=rect.center;
    auto e0=bar.vertices[1]-bar.vertices[0], e1=bar.vertices[2]-bar.vertices[1];
    const auto axis=cv::norm(e0)>=cv::norm(e1)?e0:e1;
    bar.length=float(cv::norm(axis)); bar.width=std::min(rect.size.width,rect.size.height);
    if(bar.length<config_.min_length || bar.length>config_.max_length || bar.width<=0 ||
       bar.length/bar.width<config_.min_aspect) continue;
    bar.axis=axis/bar.length; if(bar.axis.y<0) bar.axis=-bar.axis;
    if(std::acos(std::clamp(double(bar.axis.y),-1.,1.))*180/CV_PI>config_.max_tilt_deg) continue;
    describe_contour_quality(contour,bar);
    if(config_.endpoint_refinement_enabled)refine_endpoints(gray,config_,bar);
    bars_.push_back(bar);
    if(thin)++thin_accepted_lamps_;
  }
  std::sort(bars_.begin(),bars_.end(),[](const auto& a,const auto& b){return a.center.x<b.center.x;});
  pairs_.clear();pair_statuses_.clear();
  for(size_t i=0;i<bars_.size();++i) for(size_t j=i+1;j<bars_.size();++j) {
    Detection d; if(!enclose(bars_[i],bars_[j],config_,d)) continue;
    d.left=int(i);d.right=int(j);pairs_.push_back(d);
  }
  std::sort(pairs_.begin(),pairs_.end(),[](const auto& a,const auto& b){
    if(a.score!=b.score) return a.score<b.score;
    return std::tie(a.left,a.right)<std::tie(b.left,b.right);
  });
  // Only the two BEST alternatives for each lamp establish ambiguity. Two
  // equally poor alternatives must not invalidate a clearly better pairing.
  std::vector<bool> ambiguous(bars_.size(),false);
  for(size_t lamp=0;lamp<bars_.size();++lamp){
    double best=std::numeric_limits<double>::infinity();
    for(const auto& d:pairs_)if(d.left==int(lamp)||d.right==int(lamp)){
      if(!std::isfinite(best)){best=d.score;continue;}
      ambiguous[lamp]=d.score-best<=config_.pairing_ambiguity_margin;break;
    }
  }
  std::set<int> used; std::vector<Detection> accepted;
  for(const auto& d:pairs_) {
    if(ambiguous[d.left]||ambiguous[d.right]){
      pair_statuses_.push_back(PairStatus::AmbiguousLamp);continue;
    }
    if(used.count(d.left)||used.count(d.right)){
      pair_statuses_.push_back(PairStatus::SharedLamp);continue;
    }
    pair_statuses_.push_back(PairStatus::Accepted);
    used.insert(d.left);used.insert(d.right);accepted.push_back(d);
  }
  return accepted;
}
}
