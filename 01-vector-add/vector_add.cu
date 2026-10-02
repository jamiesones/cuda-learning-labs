#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n",           \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)

__global__ void vectorAdd(const float* a,
                          const float* b,
                          float* c,
                          int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        c[i] = a[i] + b[i];
}

int main()
{
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];

    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    printf("N       : %d\n", N);
    printf("Blocks  : %d\n", blocks);
    printf("Threads : %d\n", threads);
    printf("Result  : %.1f + %.1f = %.1f\n",
           h_a[0], h_b[0], h_c[0]);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
