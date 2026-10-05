#include <cstdio>
#include <cuda_runtime.h>

__global__ void coalescedAccess(const int* input, int* output)
{
    int tid = threadIdx.x;

    // Neighboring threads access neighbouring memory locations
    output[tid] = input[tid];
}

__global__ void stridedAccess(const int* input, int* output, int stride)
{
    int tid = threadIdx.x;

    // Neighbouring thread access memory locations seperated by a stride
    output[tid] = input[tid * stride];
}

int main()
{
    const int N = 64;
    const int OUTPUT_SIZE = 32;
    const int stride = 2;

    const int inputBytes = N * sizeof(int);
    const int outputBytes = OUTPUT_SIZE * sizeof(int);

    int h_input[N];
    int h_coalesced[OUTPUT_SIZE];
    int h_strided[OUTPUT_SIZE];

    // initialize host input data
    for (int i = 0; i < N; i++)
    {
        h_input[i] = i;
    }

    int* d_input;
    int* d_coalesced;
    int* d_strided;

    // allocate device memory
    cudaMalloc(&d_input, inputBytes);
    cudaMalloc(&d_coalesced, outputBytes);
    cudaMalloc(&d_strided, outputBytes);

    // Copy input data from host to device.
    cudaMemcpy(d_input, h_input, inputBytes, cudaMemcpyHostToDevice);

    // Launch one Warp (32 threads)
    coalescedAccess<<<1, 32>>>(d_input, d_coalesced);
    stridedAccess<<<1, 32>>>(d_input, d_strided, stride);

    // Copy the result back to host
    cudaMemcpy(h_coalesced, d_coalesced, outputBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_strided, d_strided, outputBytes, cudaMemcpyDeviceToHost);

    printf("Coalesced:\n");
    for (int i = 0; i < OUTPUT_SIZE ; i++)
    {
        printf("%d ", h_coalesced[i]);
    }
    printf("\n\n");

    printf("Strided (stride = %d):\n", stride);
    for (int i = 0; i < OUTPUT_SIZE ; i++)
    {
        printf("%d ", h_strided[i]);
    }
    printf("\n");

    // Release memory
    cudaFree(d_input);
    cudaFree(d_coalesced);
    cudaFree(d_strided);

    return 0;
}