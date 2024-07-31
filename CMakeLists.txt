cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(parallel_algo LANGUAGES C CUDA)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(CUDAToolkit_ROOT /usr/local/cuda)
set(CMAKE_CUDA_ARCHITECTURES 61)

include(CTest)
enable_testing()

message(STATUS "Building rust lib")
execute_process(
    COMMAND cargo build 
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/rust
    COMMAND_ERROR_IS_FATAL ANY
)
find_library(RUST_LIB parallel_algo_rs PATHS ${CMAKE_SOURCE_DIR}/rust/target/debug REQUIRED)

add_subdirectory(examples)
add_subdirectory(c_cuda)