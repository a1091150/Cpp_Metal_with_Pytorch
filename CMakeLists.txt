cmake_minimum_required(VERSION 3.21)
project(CPP_TORCH_METAL)
set(CMAKE_CXX_STANDARD 17)

add_executable(CPP_TORCH_METAL main.cpp)
add_subdirectory(metal-cpp)
target_link_libraries(CPP_TORCH_METAL METAL_CPP)
