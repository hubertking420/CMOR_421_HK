#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 128
__global__ void stencil_shared(const float *x, float *y, int N){
  __shared__ float s_x[BLOCKSIZE+2];
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int tid = threadIdx.x; 
  const int local_index = tid+1; // shift indices over to leave room for left halo
  // Load interior data
  s_x[local_index] = 0.f;
  if(i<N){
    s_x[local_index] = x[i];
  }

  // Load left halo element if it exists
  if(tid==0 && i>0){
      s_x[0] = x[i-1];
  }

  // Load right halo element if it exists
  if(tid==blockDim.x-1 && i<N-1){
      s_x[blockDim.x+1] = x[i+1];
  }
  
  // Compute stencil
  __syncthreads();

  // Interior elements
  y[i]=0.f;
  if(i>0 && i<N-1){
    y[i] = 2*s_x[i]-s_x[i-1]-s_x[i+1];
  }

  // Overall boundary elements
  if(i==0){
    y[i] = y[i+1];
  }
  if(i==N-1){
    y[i] = y[i-1];
  }
}
    
int main(int argc, char * argv[]){
  int N = 4194304;
  if (argc > 1){
    N = atoi(argv[1]);
  }

  int blockSize = BLOCKSIZE;

  // Next largest multiple of blockSize
  int numBlocks = (N + blockSize - 1) / blockSize;

  printf("Stencil operation with N = %d, blockSize = %d, numBlocks = %d\n", N, blockSize, numBlocks);

  float * x = new float[N];
  float * y = new float[N];  
  float * y_target = new float[N];

  // Define x for the stencil
  for (int i = 0; i < N; ++i){
    x[i] = 1.f;
  }

  // allocate memory and copy to the GPU
  float * d_x;
  float * d_y;  
  int size_x = N * sizeof(float);
  int size_y = N * sizeof(float);
  cudaMalloc((void **) &d_x, size_x);
  cudaMalloc((void **) &d_y, size_y);
  
  // copy memory over to the GPU
  cudaMemcpy(d_x, x, size_x, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, size_y, cudaMemcpyHostToDevice);

  stencil_shared <<< numBlocks, blockSize >>> (d_x, d_y, N);
  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess){
  printf("GPUassert: %s\n", cudaGetErrorString(code));
  }

  // copy memory back to the CPU
  cudaMemcpy(y, d_y, size_y, cudaMemcpyDeviceToHost);
  
  // test accuracy
  float sum_y = 0.f;
  for(int i = 0; i < N; ++i) sum_y += y[i];
  float target = 0.f;
  printf("error = %f\n", fabs(sum_y - target)); 

#if 1
  int num_trials = 10;
  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  for (int i = 0; i < num_trials; ++i){
    stencil_shared <<< numBlocks, blockSize >>> (d_x, d_y, N);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  
  printf("Time to run kernel 10x: %6.2f ms.\n", time);
  
#endif

  return 0;
}