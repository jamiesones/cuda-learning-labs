#include <cstdio>
#include <cuda_runtime.h>

__global__ void helloKernel()
{
    printf("Hello from block %d, thread %d\n",
            blockIdx.x,
            threadIdx.x);
}

int main()
{
    printf("Hello from CPU\n");
    
    helloKernel<<<2, 4>>>();

    cudaDeviceSynchronize();

    return 0;
}