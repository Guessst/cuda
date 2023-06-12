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
kernel_dotprod(double* x, double* y, double* resultArray){
	__shared__ double tempResult[THREADS_PER_BLOCK];
    unsigned int localIdx = threadIdx.x;
	unsigned int globalIdx = localIdx + (blockDim.x * blockIdx.x);

    if(globalIdx < ARRAY_LENGTH)
        tempResult[localIdx] = x[globalIdx] * y[globalIdx];
    else
        tempResult[localIdx] = 0.0;
    
    __syncthreads();
	
    for (unsigned int stride = THREADS_PER_BLOCK/2; stride > 0; stride /= 2){
		if (localIdx < stride)
			tempResult[localIdx] += tempResult[localIdx+stride];
		__syncthreads();
	}
	if (localIdx==0)
		resultArray[blockIdx.x] = tempResult[0];
    
    resultArray[blockIdx.x] = tempResult[0];
}

double dotprod(double* x, double* y)
{
    double result = 0;
    for(int i = 0;i < ARRAY_LENGTH;i++)
    {
        result += x[i] * y[i];
    }
    return result;
}

template <typename T>
T reduce(T* array, unsigned int length)
{
    T result = 0;
    for(int i = 0; i < length; i++)
    {
        result += array[i];
    }
    return result;
}

template <typename T>
void printArray(T* array, unsigned int length)
{
    unsigned int i;
    for(i = 0; i < length-1; i++)
    {
        std::cout << array[i] << " ";
    }
    std::cout << array[i] << " " << std::endl;
}

int main()
{
    const unsigned int arrayBytes = ARRAY_LENGTH * sizeof(double);
    const unsigned int resultArrayBytes = NUMBER_OF_BLOCKS * sizeof(double);

    double* host_x = NULL;
    host_x = (double*) malloc(arrayBytes);
    
    double* host_y = NULL;
    host_y = (double*) malloc(arrayBytes);
    for(int i = 0;i < ARRAY_LENGTH;i++)
    {
        host_x[i] = 1.0;
        host_y[i] = 1.0 + i;
    }
    double* host_result_array = NULL;
    host_result_array = (double*) malloc(resultArrayBytes);

    
    // printf("host_x: "); printArray(host_x, ARRAY_LENGTH);
    // printf("host_y: "); printArray(host_y, ARRAY_LENGTH);

    double* device_x = NULL;
    cudaMalloc(&device_x, arrayBytes);
    
    double* device_y = NULL;
    cudaMalloc(&device_y, arrayBytes);

    double* device_result_array = NULL;
    cudaMalloc(&device_result_array, resultArrayBytes);

    cudaMemcpy(device_x, host_x, arrayBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, host_y, arrayBytes, cudaMemcpyHostToDevice);

    kernel_dotprod
    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
    (device_x, device_y, device_result_array);

    cudaDeviceSynchronize();
    cudaMemcpy(
        host_result_array, device_result_array,
        resultArrayBytes,
        cudaMemcpyDeviceToHost
    );
    
    printf("host_result: "); printf("%lf\n", dotprod(host_x, host_y));
    printf("device_result: "); printf("%lf\n", reduce(host_result_array, NUMBER_OF_BLOCKS));

    cudaFree(device_x);
    cudaFree(device_y);
    free(host_x);
    free(host_y);
    
    return 0;
}