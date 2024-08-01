cmake -DCMAKE_CUDA_SEPARABLE_COMPILATION=ON -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_CUDA_ARCHITECTURES=61 -S ./ -B ./build
cmake --build ./build
