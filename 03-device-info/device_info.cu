#include <cstdio>
#include <cuda_runtime.h>

int main()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    printf("CUDA Device Count: %d\n\n", deviceCount);

    for (int device = 0; device < deviceCount; device++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        printf("=== GPU %d ===\n", device);
        printf("Name: %s\n", prop.name);

        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

        printf("Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

        printf("SM Count: %d\n", prop.multiProcessorCount);

        printf("Warp Size: %d\n", prop.warpSize);

        printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);

        printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);

        printf("Max Block Dimensions: %d x %d x %d\n",
            prop.maxThreadsDim[0],
            prop.maxThreadsDim[1],
            prop.maxThreadsDim[2]);

        printf("Max Grid Dimensions: %d x %d x %d\n",
            prop.maxGridSize[0],
            prop.maxGridSize[1],
            prop.maxGridSize[2]);

        printf("Shared Memory per Block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);

        printf("\n");
    }

    return 0;
}
