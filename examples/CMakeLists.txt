add_executable(show_gpu_info gpu_info.cu)

add_executable(convolution_0 convolution_0.c)
target_include_directories(convolution_0 PRIVATE ${CMAKE_SOURCE_DIR}/rust/src ${CUDAToolkit_ROOT}/include)
target_link_directories(convolution_0 PRIVATE ${CUDAToolkit_ROOT}/lib64)
target_link_libraries(convolution_0 ${RUST_LIB} convolution cudart)

add_executable(convolution_1 convolution_1.c)
target_include_directories(convolution_1 PRIVATE ${CMAKE_SOURCE_DIR}/rust/src ${CUDAToolkit_ROOT}/include)
target_link_directories(convolution_1 PRIVATE ${CUDAToolkit_ROOT}/lib64)
target_link_libraries(convolution_1 ${RUST_LIB} convolution cudart)