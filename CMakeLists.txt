cmake_minimum_required(VERSION 3.10)

project(
    psd_projection
    LANGUAGES CXX CUDA
)

### GoogleTest for unit testing
# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Find CUDA Toolkit and use imported targets for linking
find_package(CUDAToolkit REQUIRED)

set_property(TARGET ${ROUTINE} PROPERTY CUDA_ARCHITECTURES OFF)

set(PSD_LIB psd_lib)

set(CMAKE_INSTALL_PREFIX ../)

set(
    INCLUDE_DIRS
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${SUITESPARSE_INCLUDE_DIRECTORIES}
    include
)

file(
    GLOB_RECURSE LIB_SOURCES
    src/*.cu
)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)

### Create library
add_library(${PSD_LIB} SHARED ${LIB_SOURCES})
set_property(TARGET ${PSD_LIB} PROPERTY CUDA_ARCHITECTURES OFF)
target_include_directories(${PSD_LIB} PUBLIC ${INCLUDE_DIRS})
target_link_libraries(
    ${PSD_LIB}
    PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    CUDA::cublasLt
)

install(
    TARGETS ${PSD_LIB}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Option to control building tests
option(PSD_PROJECTION_BUILD_TESTS "Build psd_projection tests" ON)

# Only build tests if enabled
if(PSD_PROJECTION_BUILD_TESTS)
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