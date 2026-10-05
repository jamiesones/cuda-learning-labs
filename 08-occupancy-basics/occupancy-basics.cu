#include <cstdio>
#include <cuda_runtime.h>

__global__ void simpleKernel(float* data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        data[i] = data[i] * 2.0f;
    }
}

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    printf("GPU: %s\n", prop.name);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max warps per SM: %d\n\n", maxWarpsPerSM);

    int blockSizes[] = {128, 256, 512, 1024};

    for (int blockSize: blockSizes)
    {
        int activeBlocksPerSM = 0;

        // Ask the CUDA runtime how many blocks of this kernel can be resident on one SM at the same time.
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocksPerSM, simpleKernel, blockSize, 0);

        int warpsPerBlock = (blockSize + prop.warpSize - 1) / prop.warpSize;

        int activeWarpsPerSM = activeBlocksPerSM * warpsPerBlock;

        float occupancy = static_cast<float>(activeWarpsPerSM) / maxWarpsPerSM * 100.0f;

        printf("Block size: %4d\n", blockSize);
        printf("  Warps per block:      %d\n", warpsPerBlock);
        printf("  Active blocks per SM: %d\n", activeBlocksPerSM);
        printf("  Active warps per SM:  %d\n", activeWarpsPerSM);
        printf("  Occupancy:             %.1f%%\n\n", occupancy);
    }

    return 0;
}