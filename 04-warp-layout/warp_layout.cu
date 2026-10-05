#include <cstdio>
#include <cuda_runtime.h>

__global__ void showWarpLayout()
{
    int threadId = threadIdx.x;

    int warpId = threadId / warpSize;
    int laneId = threadId % warpSize;

    printf("Thread %2d -> Warp %d, Lane %2d\n",
            threadId, warpId, laneId);
}

int main()
{
    showWarpLayout<<<1, 64>>>();

    cudaDeviceSynchronize();
    
    return 0;
}