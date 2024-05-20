cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(parallel-algo LANGUAGES CXX CUDA)

include(CTest)
enable_testing()

add_subdirectory(src)