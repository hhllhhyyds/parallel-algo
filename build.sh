mkdir build
cd build
cmake -GNinja .. -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
ninja -j8
cd ..