#include <cstdio>
#include <cuda_runtime.h>

__global__ void vectorAdd(const int* a, const int* b, int* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    const int N = 1000;
    const int bytes = N * sizeof(int);

    int h_a[N];
    int h_b[N];
    int h_c[N];

    // CPU memory initialization
    for (int i = 0; i < N; i++)
    {
        h_a[i] = i;
        h_b[i] = i * 10;
    }

    // GPU memory pointers
    int* d_a;
    int* d_b;
    int* d_c;

    // Allocate GPU memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // CPU -> GPU
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("N = %d\n", N);
    printf("Threads per block = %d\n", threadsPerBlock);
    printf("Blocks per grid = %d\n", blocksPerGrid);
    printf("Total threads = %d\n\n", threadsPerBlock * blocksPerGrid);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // GPU -> CPU
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < 10; i++)
    {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}