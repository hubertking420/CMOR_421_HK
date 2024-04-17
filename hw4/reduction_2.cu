#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 128

__global__ void partial_reduction(const int N, float *x_reduced, const float *x){
    __shared__ float s_x[BLOCKSIZE / 2];

    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int tid = threadIdx.x;
  
    // Initialize shared memory to zero
    s_x[tid] = 0.f;
    // Ensure each thread reads and sums two elements from global memory
    if (i * 2 < N) {
        int index1 = i * 2;
        int index2 = index1 + 1;
        float value1 = (index1 < N) ? x[index1] : 0.f;  // Check if within bounds
        float value2 = (index2 < N) ? x[index2] : 0.f;  // Check if within bounds
        s_x[tid] = value1 + value2;
    }

    __syncthreads();  // Synchronize before starting reduction

    // number of "live" threads per block
    int alive = blockDim.x; 

    while (alive > 1){
    __syncthreads(); 
    alive /= 2; // update the number of live threads    
    if (tid < alive){
        s_x[tid] += s_x[tid + alive];
    }
    }

    // write out once we're done reducing each block
    if (tid==0){
        x_reduced[blockIdx.x] = s_x[0];
    }
}
    
int main(int argc, char * argv[]){
    int N = 4096;
    if (argc > 1){
        N = atoi(argv[1]);
    }

    int blockSize = BLOCKSIZE;

    // Next largest multiple of blockSize
    int numBlocks = (N + blockSize - 1) / blockSize;

    printf("N = %d, blockSize = %d, numBlocks = %d\n", N, blockSize, numBlocks);

    float * x = new float[N];
    float * x_reduced = new float[numBlocks];  

    for (int i = 0; i < N; ++i){
        x[i] = i + 1.f;
    }



    // allocate memory and copy to the GPU
    float * d_x;
    float * d_x_reduced;  
    int size_x = N * sizeof(float);
    int size_x_reduced = numBlocks * sizeof(float);
    cudaMalloc((void **) &d_x, size_x);
    cudaMalloc((void **) &d_x_reduced, size_x_reduced);
  
    // copy memory over to the GPU
    cudaMemcpy(d_x, x, size_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_reduced, x_reduced, size_x_reduced, cudaMemcpyHostToDevice);

    partial_reduction <<< numBlocks, blockSize >>> (N, d_x_reduced, d_x);

    // copy memory back to the CPU
    cudaMemcpy(x_reduced, d_x_reduced, size_x_reduced, cudaMemcpyDeviceToHost);

    float sum_x = 0.f;
    for (int i = 0; i < numBlocks; ++i){
        sum_x += x_reduced[i];
    }

    float target = N * (N+1) / 2.f;
    printf("error = %f\n", fabs(sum_x - target));

#if 1
    int num_trials = 10;
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int i = 0; i < num_trials; ++i){
        partial_reduction <<< numBlocks, blockSize >>> (N, d_x_reduced, d_x);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
  
    printf("Time to run kernel 10x: %6.2f ms.\n", time);
#endif

    return 0;
}