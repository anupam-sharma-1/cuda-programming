#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <math.h>

__global__ void matrix_mul(int* a, int* b, int* c, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int temp = 0;
    if(row < n && col < n){
       for(int k =0 ; k <n ; k++){
        temp += a[row * n + k]* b[k * n + col];
       }
       c[row * n + col] = temp;
    }
}

void init_matrix(int *a, int *b, int n){
    
    for(int i = 0; i < n; i++){
        for(int j=0; j < n; j++){
            a[i*n + j] = rand() % 100;
            b[i*n + j] = rand() % 100;
        }
    }
}

void verify_results(int *a, int *b, int *c, int n){
    int *verify_c = (int*)calloc(n * n, sizeof(int));
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            for(int k = 0; k < n; k++){
                verify_c[i*n + j] += a[i*n + k] * b[k*n + j];
            }
        }
    }
    for(int i=0; i < n; i++){
        for (int j=0; j < n; j++){
            assert(c[i*n+j] == verify_c[i*n+j]);
        }
    }
}

int main(){
    //initialinzing size of 2^10
    int n = 1 << 10;
    size_t bytes = n * n * sizeof(int);

    int *d_a, *d_b, *d_c;
    int *h_a, *h_b, *h_c;

    //Host memory initialization
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    //gpu memory initialization
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    init_matrix(d_a, d_b, n);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    const int BLOCK_SIZE = 16;
    const int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);

    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    matrix_mul<<<grid, threads>>>(d_a, d_b, d_c, n);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cout << "Found error: "<<cudaGetErrorString(err);
    }
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    verify_results(h_a, h_b, h_c, n);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);



    return 0;
}