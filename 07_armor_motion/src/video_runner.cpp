#include "core.hpp"
#include "detection_backend.hpp"
#include "target_model.hpp"
#include "run_output.hpp"
#include "video_runner.hpp"
#include "estimator.hpp"
#include "rigid_output.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>

int outpost::run_video(const std::string& config_path,bool headless,bool save_video,const std::string& output_dir) {
  try {
    bool interrupted=false;
    outpost::ConfigRuntime runtime(config_path);
    auto config=runtime.current();
    outpost::VideoSource source(config);
    if(save_video&&!config.timestamp_file.empty())throw std::runtime_error("timestamp sidecar mode requires --no-video; CFR writer cannot preserve VFR timing");
    auto detector=outpost::make_detector(config.detector.backend,config.detector);
    std::unique_ptr<outpost::Estimator> target;
    outpost::RunOutput run(output_dir);const auto artifact=[&](const std::string& name){return run.path(name);};
    std::cout<<"output_directory="<<run.root().string()<<'\n';
    std::unique_ptr<outpost::RigidOutput> rigid_output;
    if(config.geometry.mode=="calibrated_3d")rigid_output=std::make_unique<outpost::RigidOutput>(run,source.time_basis());
    {
      std::ofstream mode(artifact("estimator_mode.txt"));mode<<"mode="<<config.geometry.mode<<'\n';
      mode.flush();if(!mode)throw std::runtime_error("cannot record estimator mode");
    }
    using Clock=std::chrono::steady_clock;
    const auto milliseconds=[](auto begin,auto end){return std::chrono::duration<double,std::milli>(end-begin).count();};
    std::ofstream timing(artifact("timing.csv"));
    if(!timing)throw std::runtime_error("cannot create timing diagnostics");
    timing.precision(12);
    std::ofstream detector_timing(artifact("detector_timing.csv"));
    if(!detector_timing)throw std::runtime_error("cannot create detector timing diagnostics");
    detector_timing.precision(12);
    detector_timing<<"frame,traditional_ms,network_preprocess_ms,network_infer_ms,network_decode_ms,classification_ms,matching_ms,network_candidates,classifier_calls\n";
    timing<<"frame,read_ms,reload_ms,preprocess_ms,detect_ms,target_ms,encode_ms,snapshot_ms,display_ms,other_ms,total_ms\n";
    {
      std::ofstream input_info(artifact("input_metadata.txt"));input_info.precision(17);
      input_info<<"nominal_fps="<<source.fps()<<"\ndeclared_frames="<<source.reported_frame_count()
                <<"\ntime_basis="<<source.time_basis()<<"\nprefetch_capacity="<<config.prefetch_frames<<"\ncount_verification="
                <<(source.reported_frame_count()<0?"unavailable":"required_at_decoder_eof")<<'\n';
      input_info.flush();if(!input_info)throw std::runtime_error("cannot save input metadata");
      if(source.reported_frame_count()<0)std::cerr<<"warning: input frame count unavailable; EOF completeness cannot be checked\n";
    }
    std::ofstream config_audit(artifact("config_events.csv"));
    if(!config_audit)throw std::runtime_error("cannot create configuration audit");
    config_audit.precision(17);
    config_audit<<"frame,time,attempt,event,active_snapshot,message\n";
    auto csv_quote=[](const std::string& value){std::string q="\"";for(char c:value){if(c=='\"')q+='\"';q+=c;}return q+'\"';};
    std::uint64_t config_epoch=0,last_rejected_attempt=0;std::string last_rejected_message;
    std::string active_snapshot="config_0.yaml",active_text=config.source_text;
    auto save_snapshot=[&](const std::string& name,const std::string& text){
      std::ofstream snapshot(artifact(name),std::ios::binary);snapshot<<text;snapshot.flush();
      if(!snapshot)throw std::runtime_error("cannot save configuration snapshot");
    };
    save_snapshot(active_snapshot,active_text);
    config_audit<<"-1,0,0,startup,"<<active_snapshot<<",\"startup configuration\"\n";config_audit.flush();
    std::ofstream metrics(artifact("detection.csv"));
    std::ofstream association_log(artifact("association.csv")),health_log(artifact("health.csv"));
    if(!association_log||!health_log)throw std::runtime_error("cannot create estimator diagnostics");
    association_log.precision(17);health_log.precision(17);
    association_log<<"frame,time,local_id,candidate,reason,nis\n";
    health_log<<"frame,time,local_id,measurement_used,samples,ready,warning,mean_nis,failure_fraction,mean_zx,mean_zy\n";
    if(!metrics) throw std::runtime_error("cannot create detection metrics");
    metrics << "frame,time_seconds,lightbars,paired_candidates\n";
    std::ofstream quality(artifact("detector_quality.csv"));
    if(!quality)throw std::runtime_error("cannot create detector quality metrics");
    quality<<"frame,time_seconds,border_rejected_contours,thin_accepted_lamps\n";
    std::ofstream tracks(artifact("tracking.csv"));
    if(!tracks)throw std::runtime_error("cannot create tracking metrics");
    tracks<<"frame,time,id,measurement_used,state,future_valid,cx,cy,fx,fy,innovation,age_seconds,position_std,current_drawn,future_drawn,evidence\n";
    cv::VideoWriter writer;
    cv::Size video_size;
    std::ofstream motion(artifact("motion_state.csv"));
    std::ofstream reliability(artifact("measurement_reliability.csv"));
    if(!reliability)throw std::runtime_error("cannot create measurement reliability diagnostics");
    reliability.precision(17);
    reliability<<"frame,id,measurement_used,quality_available,fit_residual_px,variance_x,variance_y,roi_refined,clean_recovery_active,clean_recovery_used\n";
    if(!motion)throw std::runtime_error("cannot create motion state diagnostics");
    motion.precision(17);
    motion<<"frame,time,id,measurement_used,consecutive_hits,filtered_x,filtered_y,vx,vy,velocity_var_x,velocity_var_y,future_var_x,future_var_y,future_valid,future_drawn,measurement_low_resolution,measurement_variance,future_position_std,forecast_block_mask,forecast_block_reasons,future_display_reason,horizon_seconds\n";
    std::ofstream slot_log(artifact("slots.csv"));
    if(!slot_log)throw std::runtime_error("cannot create slot diagnostics");
    slot_log<<"frame,time,epoch,local_id,slot,measurement_used,sequence_consistent,recurrence_seconds\n";
    std::ofstream candidates(artifact("candidates.csv")),frames(artifact("frames.csv"));
    if(!candidates||!frames)throw std::runtime_error("cannot create measurement diagnostics");
    std::ofstream provenance(artifact("observation_sources.csv")),backend_audit(artifact("backend_audit.csv"));
    if(!provenance||!backend_audit)throw std::runtime_error("cannot create provenance logs");
    provenance<<"frame,index,kind,point_semantics,detector_confidence,class_index,classifier_confidence\n";
    backend_audit<<"frame,index,kind,decision,detector_confidence,class_index,classifier_confidence,raw_network_class,tlx,tly,trx,try,brx,bry,blx,bly\n";
    std::ofstream endpoints(artifact("endpoint_evidence.csv"));
    if(!endpoints)throw std::runtime_error("cannot create endpoint diagnostics");
    endpoints.precision(17);
    endpoints<<"frame,index,left_evidence,right_evidence,both_refined,min_contrast,max_spread_px\n";
    std::ofstream shape_log(artifact("shape_reliability.csv"));shape_log.precision(17);
    if(!shape_log)throw std::runtime_error("cannot create shape diagnostics");
    shape_log<<"frame,id,center_valid,shape_reliable,shape_drift_px,guard_enabled,future_valid,center_forecast_blended\n";
    std::ofstream tentative_log(artifact("tentative_observations.csv"));
    if(!tentative_log)throw std::runtime_error("cannot create tentative observation log");
    tentative_log<<"frame,id,detection_index,current_drawn,future_drawn\n";
    candidates<<"frame,time,index,left_lamp,right_lamp,cx,cy,m0x,m0y,m1x,m1y,m2x,m2y,m3x,m3y,d0x,d0y,d1x,d1y,d2x,d2y,d3x,d3y,low_resolution\n";
    std::ofstream pair_log(artifact("pairing.csv"));
    if(!pair_log)throw std::runtime_error("cannot create pairing diagnostics");
    pair_log<<"frame,time,left_lamp,right_lamp,score,status,cx,cy\n";
    frames<<"frame,time,candidates,confirmed_tracks,drawn_current,drawn_future\n";
    std::string last_config_message;
    std::ofstream coordinates(artifact("coordinates.csv"));
    if(!coordinates)throw std::runtime_error("cannot create coordinate diagnostics");
    coordinates.precision(17);
    coordinates<<"frame,space,width,height,fx,fy,cx,cy\n";
    outpost::Frame frame;
    long count = 0;
    std::ofstream timestamps(artifact("frame_timestamps.csv"));timestamps.precision(17);
    if(!timestamps)throw std::runtime_error("cannot create timestamp log");
    timestamps<<"frame,time_seconds\n";
    double last = -1;
    while (true) {
      const auto frame_begin=Clock::now();
      if(!source.next(frame))break;
      timestamps<<frame.index<<','<<frame.time_seconds<<'\n';
      const auto read_end=Clock::now();
      runtime.bind_input_size(frame.image.size());
      std::string config_message;
      const auto reload=runtime.poll(config_message);
      if(reload==outpost::ReloadResult::LiveUpdated||reload==outpost::ReloadResult::ResetRequired){
        const bool was_display=config.display;
        config=runtime.current();detector->reconfigure(config.detector);
        if(reload==outpost::ReloadResult::ResetRequired&&target)target->reconfigure(config,frame.image.size());
        if(was_display&&!config.display&&!headless)cv::destroyWindow("outpost_motion");
      }
      if(!config_message.empty()&&config_message!=last_config_message){
        std::cerr<<"frame "<<frame.index<<": "<<config_message<<'\n';last_config_message=config_message;
      }
      if(reload==outpost::ReloadResult::Rejected){
        if(runtime.update_attempt()!=last_rejected_attempt||config_message!=last_rejected_message){
          config_audit<<frame.index<<','<<frame.time_seconds<<','<<runtime.update_attempt()<<",rejected,"<<active_snapshot<<','<<csv_quote(config_message)<<'\n';
          last_rejected_attempt=runtime.update_attempt();last_rejected_message=config_message;
        }
      }else{
        last_rejected_message.clear();
        if((reload==outpost::ReloadResult::LiveUpdated||reload==outpost::ReloadResult::ResetRequired)&&config.source_text!=active_text){
          active_snapshot="config_"+std::to_string(++config_epoch)+".yaml";active_text=config.source_text;
          save_snapshot(active_snapshot,active_text);
          config_audit<<frame.index<<','<<frame.time_seconds<<','<<runtime.update_attempt()<<','
                      <<(reload==outpost::ReloadResult::ResetRequired?"reset":"live")<<','<<active_snapshot<<','<<csv_quote(config_message)<<'\n';
        }
      }
      config_audit.flush();if(!config_audit)throw std::runtime_error("incomplete configuration audit");
      if (frame.time_seconds <= last) throw std::runtime_error("non-monotonic frame time");
      last = frame.time_seconds;
      ++count;
      const auto reload_end=Clock::now();
      const auto& preprocessing=runtime.preprocessor();
      frame.image=preprocessing.process(frame.image);
      const auto preprocess_end=Clock::now();
      coordinates<<frame.index<<','<<(preprocessing.rectified()?"rectified":"raw")<<','
                 <<frame.image.cols<<','<<frame.image.rows;
      const auto& k=preprocessing.camera_matrix();
      if(preprocessing.rectified())coordinates<<','<<k.at<double>(0,0)<<','<<k.at<double>(1,1)
                                              <<','<<k.at<double>(0,2)<<','<<k.at<double>(1,2);
      else coordinates<<",,,,";
      coordinates<<'\n';
      const auto detect_begin=Clock::now();
      const auto detected=detector->detect(frame.image);
      const auto detect_end=Clock::now();
      const auto& dt=detected.timing;
      detector_timing<<frame.index<<','<<dt.traditional_ms<<','<<dt.network_preprocess_ms<<','
        <<dt.network_infer_ms<<','<<dt.network_decode_ms<<','<<dt.classification_ms<<','
        <<dt.matching_ms<<','<<dt.network_candidates<<','<<dt.classifier_calls<<'\n';
      const auto& detections=detected.observations;
      for(size_t i=0;i<detections.size();++i){const auto& d=detections[i];
        endpoints<<frame.index<<','<<i<<','<<outpost::endpoint_evidence_name(d.endpoint_evidence[0])
          <<','<<outpost::endpoint_evidence_name(d.endpoint_evidence[1])<<','<<d.endpoint_refined
          <<','<<d.endpoint_min_contrast<<','<<d.endpoint_max_spread_px<<'\n';
        provenance<<frame.index<<','<<i<<','<<outpost::observation_kind_name(d.kind)<<','
          <<(d.kind==outpost::ObservationKind::NeuralKeypoints?"learned_keypoints":"measured_lamp_centerlines")<<',';
        if(d.detection_confidence>=0)provenance<<d.detection_confidence;
        provenance<<',';if(d.classified_index>=0)provenance<<d.classified_index;
        provenance<<',';if(d.classification_confidence>=0)provenance<<d.classification_confidence;provenance<<'\n';
      }
      for(size_t i=0;i<detected.reviews.size();++i){const auto& r=detected.reviews[i];const auto& d=r.candidate;
        backend_audit<<frame.index<<','<<i<<','<<outpost::observation_kind_name(d.kind)<<','<<r.decision<<','<<d.detection_confidence<<',';
        if(d.classified_index>=0)backend_audit<<d.classified_index;
        backend_audit<<',';if(d.classification_confidence>=0)backend_audit<<d.classification_confidence;
        backend_audit<<','<<d.raw_network_class;
        for(auto point:d.measurement_points)backend_audit<<','<<point.x<<','<<point.y;
        backend_audit<<'\n';
      }
      quality<<frame.index<<','<<frame.time_seconds<<','<<detected.border_rejected<<','
             <<detected.thin_accepted<<'\n';
      const auto target_begin=Clock::now();
      if(!target)target=outpost::make_estimator(config,frame.image.size());
      const auto result=target->update(frame.time_seconds,detections,config.prediction_seconds);
      const auto& estimate=result.image;
      const auto target_end=Clock::now();
      const auto& tracked=estimate.tracks;
      for(const auto& d:estimate.associations){
        association_log<<frame.index<<','<<frame.time_seconds<<','<<d.id<<','<<d.detection<<','<<d.reason<<',';
        if(d.innovation_available)association_log<<d.nis;
        association_log<<'\n';
      }
      const auto& assignments=estimate.slots;
      const auto decisions=outpost::visibility_batch(tracked,detected.lamps,detected.pairs,config.visibility);
      for(size_t i=0;i<detected.pairs.size();++i){
        const auto& pair=detected.pairs[i];const auto state=detected.pair_status[i];
        pair_log<<frame.index<<','<<frame.time_seconds<<','<<pair.left<<','<<pair.right<<','<<pair.score<<','
                <<(state==outpost::Detector::PairStatus::Accepted?"accepted":
                   state==outpost::Detector::PairStatus::AmbiguousLamp?"ambiguous_lamp":"shared_lamp")<<','
                <<pair.center.x<<','<<pair.center.y<<'\n';
      }
      for(size_t i=0;i<detections.size();++i){const auto& d=detections[i];
        candidates<<frame.index<<','<<frame.time_seconds<<','<<i<<','<<d.left<<','<<d.right<<','<<d.center.x<<','<<d.center.y;
        for(auto p:d.measurement_points)candidates<<','<<p.x<<','<<p.y;
        for(auto p:d.corners)candidates<<','<<p.x<<','<<p.y;
        candidates<<','<<d.low_resolution<<'\n';
      }
      metrics << frame.index << ',' << frame.time_seconds << ',' << detected.lamps.size()
              << ',' << detections.size() << '\n';
      auto draw=[&](const auto& corners,cv::Scalar color){
        std::vector<cv::Point> points;
        for(const auto& p:corners)points.emplace_back(cvRound(p.x),cvRound(p.y));
        cv::polylines(frame.image,points,true,color,1,cv::LINE_AA);
      };
      size_t drawn_current=0,drawn_future=0;
      for(const auto& observation:estimate.tentative_observations){
        const auto& measured=detections.at(size_t(observation.detection_index));
        draw(measured.corners,{0,255,0});++drawn_current;
        cv::putText(frame.image,"T"+std::to_string(observation.id)+" tentative obs",
          measured.corners[0]-cv::Point2f(0,4),cv::FONT_HERSHEY_SIMPLEX,.35,{0,255,0},1);
        tentative_log<<frame.index<<','<<observation.id<<','<<observation.detection_index<<",1,0\n";
      }
      for(size_t i=0;i<tracked.size();++i) {const auto& t=tracked[i];
        shape_log<<frame.index<<','<<t.id<<','<<t.future_center_valid<<','<<t.future_shape_reliable<<','
          <<t.shape_drift_px<<','<<config.tracker.shape_guard_enabled<<','<<t.future_valid<<','<<t.center_forecast_blended<<'\n';
        const auto& h=t.health;
        health_log<<frame.index<<','<<frame.time_seconds<<','<<t.id<<','<<t.measurement_used<<','
                  <<h.samples<<','<<h.ready<<','<<h.warning<<','<<h.mean_nis<<','<<h.failure_fraction<<','
                  <<h.mean_standardized_residual.x<<','<<h.mean_standardized_residual.y<<'\n';
        const auto decision=decisions[i];drawn_current+=decision.current;drawn_future+=decision.future;
        const outpost::Detection* measurement=t.measurement_used?&detections.at(size_t(t.detection_index)):nullptr;
        reliability<<frame.index<<','<<t.id<<','<<t.measurement_used<<','<<(measurement&&measurement->quality_available)<<',';
        if(measurement&&measurement->quality_available)reliability<<measurement->fit_residual_px;
        reliability<<','<<t.measurement_variances[0]<<','<<t.measurement_variances[1]<<','
          <<(measurement&&measurement->roi_refined)<<','<<t.clean_recovery_active<<','<<t.clean_recovery_used<<'\n';
        motion<<frame.index<<','<<frame.time_seconds<<','<<t.id<<','<<t.measurement_used<<','
              <<t.consecutive_hits<<','<<t.filtered_center.x<<','<<t.filtered_center.y<<','
              <<t.velocity.x<<','<<t.velocity.y<<','<<t.velocity_variance[0]<<','<<t.velocity_variance[1]<<','
              <<t.future_position_variance[0]<<','<<t.future_position_variance[1]<<','
              <<t.future_valid<<','<<decision.future<<','<<t.measurement_low_resolution<<','
              <<t.measurement_variance<<','<<t.forecast.position_std<<','<<t.forecast.blocks<<','
              <<outpost::forecast_block_names(t.forecast.blocks)<<','
              <<(decision.future?"drawn":!t.measurement_used?"no_accepted_observation":
                 !t.future_valid?"model_forecast_blocked":"visibility_gate")<<','
              <<config.prediction_seconds<<'\n';
        const auto slot=assignments[i];
        slot_log<<frame.index<<','<<frame.time_seconds<<','<<estimate.epoch<<','<<t.id<<','
                <<slot.slot<<','<<t.measurement_used<<','<<slot.sequence_consistent<<','
                <<estimate.recurrence_seconds<<'\n';
        if(decision.current)draw(t.current,t.measurement_used?cv::Scalar(0,255,0):cv::Scalar(0,180,255));
        if(decision.future)draw(t.future,{0,0,255});
        const std::string label=slot.slot>=0?"S"+std::to_string(slot.slot+1)+(slot.sequence_consistent?"":"?"):
                                               "T"+std::to_string(t.id)+" unknown";
        if(decision.current)cv::putText(frame.image,label+(t.measurement_used?" obs":" lamp-supported"),
                    t.current[0]-cv::Point2f(0,4),cv::FONT_HERSHEY_SIMPLEX,.35,{0,255,0},1);
        const auto c=t.current_center,f=t.future_center;
        tracks<<frame.index<<','<<frame.time_seconds<<','<<t.id<<','<<t.measurement_used<<','
              <<(t.measurement_used?"tracking":"coasting")<<','<<t.future_valid<<','
              <<c.x<<','<<c.y<<','<<f.x<<','<<f.y<<','<<t.innovation<<','
              <<t.age_seconds<<','<<t.position_std<<','<<decision.current<<','<<decision.future<<','
              <<(decision.evidence==outpost::Evidence::AcceptedPair?"pair":
                 decision.evidence==outpost::Evidence::CompatibleSingleLamp?"single_lamp":"none")<<'\n';
      }
      if(result.rigid){
        const auto drawn=rigid_output->record_and_draw(frame,*result.rigid,result.epoch,detections);
        drawn_current=drawn.first;drawn_future=drawn.second;
      }
      frames<<frame.index<<','<<frame.time_seconds<<','<<detections.size()<<','<<(result.rigid?size_t(result.rigid->valid?3:0):tracked.size())<<','
            <<drawn_current<<','<<drawn_future<<'\n';
      if(!result.rigid){
      cv::putText(frame.image,"GREEN=observed ORANGE=lamp-supported RED=future",{10,25},
                  cv::FONT_HERSHEY_SIMPLEX,.42,{255,255,255},1);
      cv::putText(frame.image,"lead="+std::to_string(config.prediction_seconds)+"s; S=online slot hypothesis",
                  {10,45},cv::FONT_HERSHEY_SIMPLEX,.4,{255,255,255},1);
      cv::putText(frame.image,estimate.sequence_consistent?"slot sequence consistent; forecast=2D translation":
                                                "slot order unconfirmed; forecast=2D translation",
                  {10,65},cv::FONT_HERSHEY_SIMPLEX,.4,{255,255,255},1);
      }
      if(save_video&&!writer.isOpened()) {
        // The default rendering is unchanged; experimental runs are explicitly
        // marked below so exported video cannot be mistaken for the baseline.
        writer.open(artifact("tracking-preview.mp4"),cv::VideoWriter::fourcc('m','p','4','v'),
                    source.fps(),frame.image.size());
        if(!writer.isOpened())throw std::runtime_error("cannot create preview video");
        video_size=frame.image.size();
      }
      if(!result.rigid&&config.tracker.one_sided_guard_enabled){
        bool rejected=false;for(const auto& d:estimate.associations)
          rejected|=std::string(d.reason)=="one_sided_length_drop";
        cv::putText(frame.image,rejected?"EXPERIMENT: one-side guard / measurement skipped":
                                        "EXPERIMENT: one-side guard enabled",
                    {10,85},cv::FONT_HERSHEY_SIMPLEX,.4,{0,220,255},1);
      }
      if(config.detector.backend!="traditional")cv::putText(frame.image,"EXPERIMENT: neural-supported measured lamps",
                    {10,105},cv::FONT_HERSHEY_SIMPLEX,.4,{0,220,255},1);
      const auto encode_begin=Clock::now();
      if(save_video)writer.write(frame.image);
      const auto encode_end=Clock::now();
      if(frame.index==0 || frame.index==180 || frame.index==360 || frame.index==480 || frame.index==720)
        if(!cv::imwrite(artifact("frame_"+std::to_string(frame.index)+".png"),frame.image))
          throw std::runtime_error("cannot save diagnostic image");
      const auto snapshot_end=Clock::now();
      if (config.display && !headless) {
        cv::imshow("outpost_motion", frame.image);
        const double interval_ms=1000*source.frame_interval(frame.index);
        do{
          const double remaining=interval_ms-milliseconds(frame_begin,Clock::now());
          if(cv::waitKey(int(std::clamp(std::ceil(remaining),1.,50.)))==27){interrupted=true;break;}
        }while(milliseconds(frame_begin,Clock::now())<interval_ms);
      }
      const auto frame_end=Clock::now();
      const double read_ms=milliseconds(frame_begin,read_end),reload_ms=milliseconds(read_end,reload_end),
        preprocess_ms=milliseconds(reload_end,preprocess_end),detect_ms=milliseconds(detect_begin,detect_end),
        target_ms=milliseconds(target_begin,target_end),encode_ms=milliseconds(encode_begin,encode_end),
        snapshot_ms=milliseconds(encode_end,snapshot_end),display_ms=milliseconds(snapshot_end,frame_end),
        total_ms=milliseconds(frame_begin,frame_end);
      const double other_ms=std::max(0.,total_ms-read_ms-reload_ms-preprocess_ms-detect_ms-target_ms-encode_ms-snapshot_ms-display_ms);
      timing<<frame.index<<','<<read_ms<<','<<reload_ms<<','<<preprocess_ms<<','<<detect_ms<<','<<target_ms<<','
            <<encode_ms<<','<<snapshot_ms<<','<<display_ms<<','<<other_ms<<','<<total_ms<<'\n';
      if(interrupted)break;
    }
    for(auto* stream:{&metrics,&quality,&tracks,&motion,&slot_log,&candidates,&frames,&pair_log,
                      &coordinates,&association_log,&health_log,&provenance,&backend_audit,&endpoints,&timing,&detector_timing,&reliability,&timestamps,&shape_log,&tentative_log}){
      stream->flush();if(!*stream)throw std::runtime_error("incomplete output log");
    }
    if(rigid_output)rigid_output->finish();
    {std::ofstream pipeline(artifact("input_pipeline.txt"));
      pipeline<<"capacity="<<config.prefetch_frames<<"\nqueue_high_water="<<source.queue_high_water()<<"\nconsumed_frames="<<count<<"\ndrop_policy=never\n";
      pipeline.flush();if(!pipeline)throw std::runtime_error("cannot save pipeline diagnostics");}
    writer.release();
    if(save_video&&count>0){
      // VideoWriter has no per-write success result. Decode the completed file
      // before marking the run complete, so truncated output is not success.
      cv::VideoCapture check(artifact("tracking-preview.mp4"));
      if(!check.isOpened())throw std::runtime_error("cannot verify preview video");
      cv::Mat decoded;long verified=0;
      while(check.read(decoded)){
        if(decoded.size()!=video_size)throw std::runtime_error("preview resolution mismatch");
        ++verified;
      }
      if(verified!=count)throw std::runtime_error("incomplete preview video frame count");
    }
    run.finish(count,interrupted,save_video);
    std::cout << "frames=" << count << " fps=" << source.fps()
              << " last_time_seconds=" << last << " hardware_initialized=0\n";
    return count ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
