# CUDA Programming Practice Repository

This repository contains my practice projects for CUDA programming, focusing on GPU-accelerated computing using NVIDIA's CUDA platform.

## Program Overview

### Vector Addition Program

#### Description
The `vector_addition.cu` program demonstrates basic CUDA kernel invocation to perform element-wise addition of two vectors on the GPU.

#### Files
- `vector_addition.cu`: CUDA C++ source code implementing vector addition.
- `README.md`: This file explaining the repository and program.

#### Features
- **CUDA Kernel (`vectorAdd`)**:
  - Utilizes GPU threads to concurrently add corresponding elements of two input vectors (`a` and `b`) and store the result in vector `c`.

- **Memory Management**:
  - Allocates memory on both the host and device.
  - Transfers data between host and device using `cudaMemcpy`.
  - Handles error checking using `cudaGetLastError` and `cudaError_t`.

- **Verification**:
  - Checks correctness of the GPU computation against CPU computation using `error_check` function.
  - Outputs results showing each element addition from vectors `a`, `b`, and `c`.

#### Execution
To compile and run the program:
1. Ensure CUDA Toolkit is installed and `nvcc` (NVIDIA CUDA Compiler) is accessible.
2. Navigate to the directory containing `vector_addition.cu`.
3. Compile using `nvcc`:
   ```bash
   nvcc vector_addition.cu -o vector_addition
