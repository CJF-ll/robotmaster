cmake_minimum_required(VERSION 3.16)

project(sensor_sync LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

add_executable(sensor_sync src/main.cpp)
target_include_directories(sensor_sync PRIVATE include)
target_link_libraries(sensor_sync PRIVATE Threads::Threads)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(sensor_sync PRIVATE -Wall -Wextra -Wpedantic)
endif()
