#include <iostream>
#include <cuda.h>
#include <assert.h>

#define CEIL(A,B) (1+(A-1)/B)
// #define THREADS_PER_BLOCK 32
#define THREADS_PER_BLOCK 256
// #define ARRAY_LENGTH 128
#define ARRAY_LENGTH 10'000'000
#define NUMBER_OF_BLOCKS CEIL(ARRAY_LENGTH, THREADS_PER_BLOCK)

__global__ void
kernel_daxpy(double scalarA, double* x, double* y){
	unsigned int localIdx = threadIdx.x;
	unsigned int globalIdx = localIdx + (blockDim.x * blockIdx.x);

	if(globalIdx < ARRAY_LENGTH)
        y[globalIdx] += scalarA * x[globalIdx];
}

void daxpy(double scalarA, double* arrayX, double* arrayY, unsigned int length)
{
    for(unsigned int i = 0;i < length;i++)
    {
        arrayY[i] += scalarA * arrayX[i];
    }
}

template <typename T>
void printArray(T* array, unsigned int length)
{
    unsigned int i;
    for(i = 0;i < length - 1;i++)
    {
        std::cout << array[i] << " ";
    }
    std::cout << array[i] << " " << std::endl;
}

int main()
{
    double scalarA = 2;
    const unsigned int arrayBytes = ARRAY_LENGTH * sizeof(double);

    double* host_x = NULL;
    host_x = (double*) malloc(arrayBytes);
    
    double* host_y = NULL;
    double* host_y_default = NULL;
    host_y = (double*) malloc(arrayBytes);
    host_y_default = (double*) malloc(arrayBytes);
    for(int i = 0;i < ARRAY_LENGTH;i++)
    {
        host_x[i] = 1.0;
        host_y[i] = 1.0 + i;
        host_y_default[i] = 1.0 + i;
    }
    
    // printf("host_x: "); printArray(host_x, ARRAY_LENGTH);
    // printf("host_y: "); printArray(host_y, ARRAY_LENGTH);

    double* device_x = NULL;
    cudaMalloc(&device_x, arrayBytes);
    
    double* device_y = NULL;
    cudaMalloc(&device_y, arrayBytes);

    cudaMemcpy(device_x, host_x, arrayBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, host_y, arrayBytes, cudaMemcpyHostToDevice);

    kernel_daxpy
    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
    (scalarA, device_x, device_y);

    cudaDeviceSynchronize();
    cudaMemcpy(host_y, device_y, arrayBytes, cudaMemcpyDeviceToHost);

    // printf("host_y*: "); printArray(host_y, ARRAY_LENGTH);
    
    daxpy(scalarA, host_x, host_y_default, ARRAY_LENGTH);
    // printf("host_y_default*: "); printArray(host_y_default, ARRAY_LENGTH);

    for(int i = 0;i < ARRAY_LENGTH;i++)
    {
        assert(host_y[i] == host_y_default[i]);
    }

    cudaFree(device_x);
    cudaFree(device_y);
    free(host_x);
    free(host_y);
    
    return 0;
}