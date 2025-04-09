cmake_minimum_required(VERSION 3.21)
project(CPP_TORCH_METAL)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# set(CMAKE_CXX_COMPILER, "/usr/bin/g++")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --no-as-needed")

add_executable(CPP_TORCH_METAL main.cpp)

set(CMAKE_PREFIX_PATH "/opt/homebrew/anaconda3/envs/align3r/lib/python3.11/site-packages/torch")
find_package(Torch REQUIRED)
target_link_libraries(CPP_TORCH_METAL "${TORCH_LIBRARIES}")

# add_subdirectory(/opt/homebrew/anaconda3/envs/align3r/include/python3.11)
add_subdirectory(metal-cpp)
target_link_libraries(CPP_TORCH_METAL METAL_CPP)
