project(styletransfer-ncnn-vulkan)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.5.3-windows-vs2019/x64/x64/vc16/staticlib)
find_package(OpenCV REQUIRED)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20210720-windows-vs2019/x64/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_executable(styletransfer-ncnn-vulkan main.cpp)

target_link_libraries(styletransfer-ncnn-vulkan ncnn ${OpenCV_LIBS})
