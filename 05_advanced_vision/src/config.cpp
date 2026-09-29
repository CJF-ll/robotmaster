#include "config.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace {

std::string trim(std::string text) {
  const auto is_not_space = [](unsigned char c) { return !std::isspace(c); };
  text.erase(text.begin(), std::find_if(text.begin(), text.end(), is_not_space));
  text.erase(std::find_if(text.rbegin(), text.rend(), is_not_space).base(), text.end());
  return text;
}

std::string resolve_path(const std::filesystem::path& config_path,
                         const std::string& value) {
  std::filesystem::path path(value);
  if (path.is_relative()) {
    path = config_path.parent_path() / path;
  }
  return std::filesystem::weakly_canonical(path).string();
}

template <typename T>
T parse_number(const std::unordered_map<std::string, std::string>& values,
               const std::string& key, T fallback) {
  const auto it = values.find(key);
  if (it == values.end()) {
    return fallback;
  }
  try {
    if constexpr (std::is_integral_v<T>) {
      return static_cast<T>(std::stoll(it->second));
    } else {
      return static_cast<T>(std::stod(it->second));
    }
  } catch (const std::exception&) {
    throw std::runtime_error("invalid number for config key: " + key);
  }
}

}  // namespace

Config load_config(const std::string& path_string) {
  const std::filesystem::path path = std::filesystem::absolute(path_string);
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open config file: " + path.string());
  }

  std::unordered_map<std::string, std::string> values;
  std::string line;
  int line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    line = trim(line);
    if (line.empty() || line.front() == '#') {
      continue;
    }
    const auto equal = line.find('=');
    if (equal == std::string::npos) {
      throw std::runtime_error("invalid config line " + std::to_string(line_number));
    }
    values[trim(line.substr(0, equal))] = trim(line.substr(equal + 1));
  }

  Config config;
  const auto require_path = [&](const std::string& key) {
    const auto it = values.find(key);
    if (it == values.end() || it->second.empty()) {
      throw std::runtime_error("missing config key: " + key);
    }
    return resolve_path(path, it->second);
  };

  config.input_video = require_path("input_video");
  config.output_video = require_path("output_video");
  config.output_csv = require_path("output_csv");
  config.queue_capacity = parse_number(values, "queue_capacity", config.queue_capacity);
  config.h_min = parse_number(values, "h_min", config.h_min);
  config.h_max = parse_number(values, "h_max", config.h_max);
  config.s_min = parse_number(values, "s_min", config.s_min);
  config.s_max = parse_number(values, "s_max", config.s_max);
  config.v_min = parse_number(values, "v_min", config.v_min);
  config.v_max = parse_number(values, "v_max", config.v_max);
  config.min_area = parse_number(values, "min_area", config.min_area);
  config.max_area = parse_number(values, "max_area", config.max_area);
  config.morphology_kernel =
      parse_number(values, "morphology_kernel", config.morphology_kernel);
  config.max_missed_frames =
      parse_number(values, "max_missed_frames", config.max_missed_frames);
  config.max_match_distance =
      parse_number(values, "max_match_distance", config.max_match_distance);
  config.tracker_alpha = parse_number(values, "tracker_alpha", config.tracker_alpha);
  config.tracker_beta = parse_number(values, "tracker_beta", config.tracker_beta);

  if (config.queue_capacity <= 0 || config.min_area < 0.0 ||
      config.max_area < config.min_area || config.max_missed_frames < 0 ||
      config.max_match_distance <= 0.0 || config.tracker_alpha < 0.0 ||
      config.tracker_alpha > 1.0 || config.tracker_beta < 0.0 ||
      config.tracker_beta > 1.0) {
    throw std::runtime_error("config values are out of range");
  }
  if (config.morphology_kernel <= 0) {
    throw std::runtime_error("morphology_kernel must be positive");
  }
  if (config.morphology_kernel % 2 == 0) {
    ++config.morphology_kernel;
  }
  return config;
}
