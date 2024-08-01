cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(parallel_algo LANGUAGES C CUDA)

include(CTest)
enable_testing()

message(STATUS "CMAKE_CUDA_SEPARABLE_COMPILATION=" ${CMAKE_CUDA_SEPARABLE_COMPILATION})
message(STATUS "CUDAToolkit_ROOT=" ${CUDAToolkit_ROOT})
message(STATUS "DCMAKE_CUDA_ARCHITECTURES=" ${CMAKE_CUDA_ARCHITECTURES})

message(STATUS "Building rust lib")
execute_process(
    COMMAND cargo build 
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/rust
    COMMAND_ERROR_IS_FATAL ANY
)
find_library(RUST_LIB parallel_algo_rs PATHS ${CMAKE_SOURCE_DIR}/rust/target/debug REQUIRED)
message(STATUS "RUST_LIB=" ${RUST_LIB})

add_subdirectory(examples)
add_subdirectory(c_cuda)