add_library(convolution SHARED
    conv_2d_basic.cu
    hello_cuda.cu
)
target_include_directories(convolution PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})