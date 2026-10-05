#include <cstdio>
#include <cuda_runtime.h>

__global__ void memoryExample(const int* input, int* output)
{
    int tid = threadIdx.x;

    // Read from global memory into a thread-local variable.
    // This value is typically stored in a register.
    int value = input[tid];

    // Shared memory accessible by all threads in the block.
    __shared__ int sharedData[8];

    // Each thread stores its value into shared memory
    sharedData[tid] = value;

    // Wait until all threads in the block finish writing to shared memory
    __syncthreads();

    // Compute the index of the thread on the opposite side
    int reverseIndex = blockDim.x - 1 - tid;

    // Read from shared memory and write the result to global memory
    output[tid] = sharedData[reverseIndex];
}

int main()
{
    const int N = 8;
    const int bytes = N * sizeof(int);

    int h_input[N]  = {0,1,2,3,4,5,6,7};
    int h_output[N] = {0};

    // Device memory pointers
    int* d_input;
    int* d_output;

    // Allocate memory on the GPU
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // Copy input data from host memory to device memory
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Launch one block with eight threads
    memoryExample<<<1, 8>>>(d_input, d_output);

    // Copy the result from device memory back to host memory
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // Print the input data
    printf("Input : ");

    for (int i = 0; i < N; i++)
        printf("%d ", h_input[i]);

    printf("\n");

    // Print the output data
    printf("Output: ");

    for (int i = 0; i < N; i++)
        printf("%d ", h_output[i]);
    
    printf("\n");

    // Release GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}