#pragma once
#include <string>
namespace outpost {
int run_calibrated_video(const std::string& config_path,bool headless,bool save_video,
                         const std::string& output_directory);
}
