cmake_minimum_required(VERSION 3.9)

project(
    cuADMM_MATLAB
    LANGUAGES CXX CUDA
)

# find MATLAB, LAPACK, and BLAS
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

include_directories(${TARGET_NAME} PUBLIC ${INCLUDE_DIRS})

matlab_add_mex(
    NAME ${TARGET_NAME}
    SRC ${MEX_SOURCES}
    OUTPUT_NAME ${TARGET_NAME}
    LINK_TO 
    ${Matlab_LIBRARIES} 
    cublas cusolver 
    ${CMAD_LIB} ${CCOLAMD_LIB} ${CHOLMOD_LIB}
)

set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
