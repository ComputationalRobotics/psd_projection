cmake_minimum_required(VERSION 3.10)

project(
    psd_projection_MATLAB
    LANGUAGES CXX CUDA
)

# NOTE -- host compiler choice matters here.  The MEX file is dlopen'd into
# MATLAB's own process, and MATLAB ships its own libstdc++.  If the MEX is built
# with a newer GCC than MATLAB's libstdc++ supports, it links successfully but
# fails at load time with e.g. "version `GLIBCXX_3.4.32' not found".
# MATLAB R2024b bundles a libstdc++ providing at most GLIBCXX_3.4.30, so build
# this target with GCC <= 12 (gcc 12 -> 3.4.30, gcc 13 -> 3.4.32).  Check a given
# MATLAB install with:
#   strings <matlabroot>/sys/os/glnxa64/libstdc++.so.6 | grep -o 'GLIBCXX_3\.4\.[0-9]*' | sort -V -u | tail -1
# The core library (../CMakeLists.txt) has no such constraint.

# C++17 is required by the adaptive method's headers
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# find MATLAB
find_package(Matlab REQUIRED)

set(TOP_SOURCE_NAME psd_projection_MATLAB.cu)
set(TARGET_NAME psd_projection_MATLAB)

file(
    GLOB_RECURSE MEX_SOURCES
    ${TOP_SOURCE_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cu
)

set(
    INCLUDE_DIRS
    ${Matlab_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

include_directories(${INCLUDE_DIRS})

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
matlab_add_mex(
    NAME ${TARGET_NAME}
    SRC ${MEX_SOURCES}
    OUTPUT_NAME ${TARGET_NAME}
    LINK_TO
    ${Matlab_LIBRARIES}
    cublas cusolver
)

set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
