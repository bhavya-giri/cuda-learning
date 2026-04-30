#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

static void printCudaError(cudaError_t err, const char *what) {
    if (err != cudaSuccess)
        fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
}

int main(void) {
    int devCount = 0;
    cudaError_t err = cudaGetDeviceCount(&devCount);
    printCudaError(err, "cudaGetDeviceCount");
    if (err != cudaSuccess)
        return EXIT_FAILURE;
    if (devCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    printf("Found %d CUDA device(s).\n\n", devCount);

    for (int i = 0; i < devCount; i++) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        printCudaError(err, "cudaGetDeviceProperties");
        if (err != cudaSuccess)
            return EXIT_FAILURE;

        printf("=== Device %d ===\n", i);
        printf("  name:                    %s\n", prop.name);
        printf("  compute capability:       %d.%d\n", prop.major, prop.minor);
        printf("  totalGlobalMem:           %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  multiprocessorCount:      %d\n", prop.multiProcessorCount);
        printf("  maxThreadsPerBlock:       %d\n", prop.maxThreadsPerBlock);
        printf("  maxThreadsDim:            [%d, %d, %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1],
               prop.maxThreadsDim[2]);
        printf("  maxGridSize:              [%d, %d, %d]\n", prop.maxGridSize[0], prop.maxGridSize[1],
               prop.maxGridSize[2]);
        printf("  warpSize:                 %d\n", prop.warpSize);
        printf("  memPitch:                 %zu\n", prop.memPitch);
        printf("  clockRate:                %d kHz\n", prop.clockRate);
        printf("  memoryClockRate:          %d kHz\n", prop.memoryClockRate);
        printf("  memoryBusWidth:           %d bits\n", prop.memoryBusWidth);
        printf("  maxSharedMemPerBlock:     %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  regsPerBlock:             %d\n", prop.regsPerBlock);
        printf("\n");
    }

    return EXIT_SUCCESS;
}
