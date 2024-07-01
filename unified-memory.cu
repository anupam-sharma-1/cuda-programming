#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <device_launch_parameters.h>
//#include <opencv2/opencv.hpp>

__global__ void vectorAdd(int* a, int* b, int* c, int n){
    int thread_id = ( blockIdx.x * blockDim.x ) + threadIdx.x;

    if (thread_id < n){
        c[thread_id] = a[thread_id] + b[thread_id];
    }

}

//cv::Mat readImage()
void matrix_init(int* a, int n){
    for(int i = 0; i < n; i++){
        a[i] = rand() % 100;
    }
}

void error_check(int* a, int* b, int* c, int n){
    for(int i=0; i < n; i++){
        assert(c[i] == a[i] + b[i]);
    }
}

int main(){
    //setting up n = 2^16
    int n = 1 << 16;

    int *a, *b, *c;

    //Threadblock size
    int NUM_THREADS = 256;
    int NUM_BLOCKS = (n + NUM_THREADS - 1) / NUM_THREADS; // Round up division
    size_t bytes = sizeof(int) * n;
    //Allocate host memory
    // h_a = (int*)malloc(bytes);
    // h_b = (int*)malloc(bytes);
    // h_c = (int*)malloc(bytes);

    //Allocate device memory
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    matrix_init(a, n);
    matrix_init(b, n);


    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(a, b, c, n);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        // Additional handling or debugging here
    }

    error_check(a, b, c, n);
    std::cout<<"done\n";

    return 0;
}
