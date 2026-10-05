#include <cstdio>
#include <cuda_runtime.h>

__global__ void divergenceBranch()
{
    int tid = threadIdx.x;
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;

    if (tid % 2 == 0)
    {
        printf("Warp %d Lane %2d Thread %2d -> Path A\n", warpId, laneId, tid);
    }
    else
    {
        printf("Warp %d Lane %2d Thread %2d -> Path B\n", warpId, laneId, tid);
    }
}

__global__ void warpAlignedBranch()
{
    int tid = threadIdx.x;
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;

    if (warpId == 0)
    {
        printf("Warp %d Lane %2d Thread %2d -> Path A\n", warpId, laneId, tid);
    }
    else
    {
        printf("Warp %d Lane %2d Thread %2d -> Path B\n", warpId, laneId, tid);
    }
}

int main()
{
    printf("=== Divergent Branch ===\n");

    divergenceBranch<<<1, 64>>>();
    cudaDeviceSynchronize();

    printf("\n=== Warp-Aligned Branch ===\n");

    warpAlignedBranch<<<1, 64>>>();
    cudaDeviceSynchronize();

    return 0;
}