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
    //Host vector pointer
    int *h_a, *h_b, *h_c;
    //Device vector pointer
    int *d_a, *d_b, *d_c;

    //Threadblock size
    int NUM_THREADS = 256;
    int NUM_BLOCKS = (n + NUM_THREADS - 1) / NUM_THREADS; // Round up division
    size_t bytes = sizeof(int) * n;
    //Allocate host memory
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    //Allocate device memory
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    matrix_init(h_a, n);
    matrix_init(h_b, n);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        // Additional handling or debugging here
    }
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    error_check(h_a, h_b, h_c, n);
    std::cout<<"done\n";
    for(int i=0; i < n; i++){
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
