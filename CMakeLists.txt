cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(parallel_algo LANGUAGES C CUDA)

include(CTest)
enable_testing()

add_subdirectory(examples)