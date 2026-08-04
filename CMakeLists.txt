cmake_minimum_required(VERSION 3.10)

project(
    psd_projection
    LANGUAGES CXX CUDA
)

# C++17 is required by the adaptive method's headers (filter_bank.h declares the
# composite filter bank as inline constexpr aggregates).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA Toolkit and use imported targets for linking
find_package(CUDAToolkit REQUIRED)

set(PSD_LIB psd_lib)

set(
    INCLUDE_DIRS
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    include
)

file(
    GLOB_RECURSE LIB_SOURCES
    src/*.cu
)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)

### Create library
# src/ and include/ contain the spectrum-aware adaptive projection method plus
# eig_FP64_psd, the exact cuSOLVER FP64 reference projection.  The benchmark harness
# and its vendored fixed-composite baseline (which carries its own copy of
# eig_FP64_psd) live in the untracked benchmark/ development tree and are configured
# and built separately -- this file never references them.
add_library(${PSD_LIB} SHARED ${LIB_SOURCES})
# Force the CUDA architecture off so nvcc emits only its own default cubin+PTX and
# the library stays forward-portable.  NOTE: setting the TARGET property below
# OVERRIDES -DCMAKE_CUDA_ARCHITECTURES from the command line (the property is only
# initialised from that variable at add_library time).  To emit native SASS for a
# specific GPU (e.g. 90 for H100/H200), replace OFF with the architecture, or drop
# the set_property line so the cache variable takes effect.
set_property(TARGET ${PSD_LIB} PROPERTY CUDA_ARCHITECTURES OFF)
target_include_directories(${PSD_LIB} PUBLIC ${INCLUDE_DIRS})
target_compile_features(${PSD_LIB} PUBLIC cxx_std_17)
target_link_libraries(
    ${PSD_LIB}
    PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)

install(
    TARGETS ${PSD_LIB}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Option to control building tests
option(PSD_PROJECTION_BUILD_TESTS "Build psd_projection tests" OFF)

# Only build tests if enabled.  The GoogleTest download below does run at configure
# time, but it sits inside this guard and the option is OFF by default, so a plain
# configure needs no network.
if(PSD_PROJECTION_BUILD_TESTS)
  ### GoogleTest for unit testing
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  ### Configure unit tests
  enable_testing()

  add_executable(
    tests
    test/tests.cpp
  )
  target_link_libraries(
    tests
    GTest::gtest_main
    ${PSD_LIB}
  )

  include(GoogleTest)
  gtest_discover_tests(tests)
endif()
