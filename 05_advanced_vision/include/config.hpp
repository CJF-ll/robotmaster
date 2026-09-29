#pragma once

#include <string>

struct Config {
  std::string input_video;
  std::string output_video;
  std::string output_csv;
  int queue_capacity = 8;
  int h_min = 90;
  int h_max = 140;
  int s_min = 80;
  int s_max = 255;
  int v_min = 50;
  int v_max = 255;
  double min_area = 100.0;
  double max_area = 10000.0;
  int morphology_kernel = 5;
  int max_missed_frames = 5;
  double max_match_distance = 80.0;
  double tracker_alpha = 0.65;
  double tracker_beta = 0.08;
};

Config load_config(const std::string& path);
