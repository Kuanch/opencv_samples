# project name
PROJECT(cascade_inference)
# requirement of cmake version
cmake_minimum_required(VERSION 3.5)

# set the directory of executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

# find required opencv
find_package(OpenCV REQUIRED)
# directory of opencv headers
set(OpenCV_INCLUDE_DIRS "/usr/local/include")
include_directories(${OpenCV_INCLUDE_DIRS})
# name of executable file and path of source file
add_executable(cascade_inference src/cascade_inference.cpp)
# directory of opencv library
set(OpenCV_LIBRARY_DIRS "/usr/local/cuda/lib64")
link_directories(${OpenCV_LIBRARY_DIRS})
# opencv libraries
target_link_libraries(cascade_inference ${OpenCV_LIBS})
