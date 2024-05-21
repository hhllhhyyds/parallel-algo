mkdir build
cd build
cmake -GNinja .. -DCMAKE_CXX_STANDARD=17 -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc --preset conan-release
ninja -j8
cd ..