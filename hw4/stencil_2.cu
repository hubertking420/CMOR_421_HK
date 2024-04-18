#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 128

__global__ void stencil_global(const float *x, float *y, int N, float bc_initial, float bc_final){
  __shared__ float s_x[BLOCKSIZE+2];
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int tid = threadIdx.x; 
  const int local_index = tid+1; // shift indices over to leave room for left halo

  // Load interior data
  if(i<N){
    s_x[local_index] = x[i];
  }

  // Load initial value or last element from previous block
  if(tid==0){
    if(blockIDx.x == 0){
      s_x[0] = bc_initial;
    }
    else if(blockIDx.x > 0){
      s_x[0] = x[i-1];
    }
  }

  // Load final value or first element from next block
  if(tid==blockDim.x-1){
    if(blockIDx.x==blockDim.x-1){
      s_x[blockDim.x-1] = bc_final;
    }
    else if(blockIDx.x<blockDim.x-1){
      s_x[0] = x[i+1];
    }
  }
  
  // Compute stencil
  __syncthreads();

  // Interior elements
  if(i>0 && i<N-1){
    y[i] = 2*s_x[local_index]-s_x[local_index-1]-s_x[local_index+1];
  }
  // Boundary conditions
  if(i==0){
    y[i] = bc_initial;
  }
  else if(i==N-1){
    y[i] = bc_final;
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
  float * y = new float[N];  

  // Define boundary conditions
  float bc_initial = 50.f;
  float bc_final = 50.f;

  // Define x for the stencil
  for (int i = 0; i < N; ++i){
    if(i==0) x[i]=bc_initial;
    else if(i==N-1) x[i]=bc_final;
    else x[i] = 10.f;
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

  stencil_global <<< numBlocks, blockSize+2 >>> (d_x, d_y, N, bc_initial, bc_final);

  // copy memory back to the CPU
  cudaMemcpy(y, d_y, size_y, cudaMemcpyDeviceToHost);

  // Sum up the values
  float sum_y = 0.f;
  for(int i = 0; i < N; ++i){
    sum_y += y[i];
  }
  // Compute target for stencil and check for accuracy
  float target = bc_initial+bc_final;
  printf("error = %f\n", fabs(sum_y - target));

#if 0
  int num_trials = 10;
  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  for (int i = 0; i < num_trials; ++i){
    stencil_global <<< numBlocks, blockSize >>> (d_x, d_y, N, initial, final);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  
  printf("Time to run kernel 10x: %6.2f ms.\n", time);
  
#endif

  return 0;
}