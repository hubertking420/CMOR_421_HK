#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 128

__global__ void stencil_global(const float *x, float *y, int N, long bc_initial, long bc_final){
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  // Boundary conditions
  if(i == 0){
    y[i] = bc_initial;
    return;
  }
  if(i == N-1){
    y[i] = bc_final;
    return;
  }
  // Interior elements
  y[i] = 2*x[i]-x[i-1]-x[i+1];
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

  stencil_global <<< numBlocks, blockSize >>> (x, y, N, bc_initial, bc_:wfinal);

  // copy memory back to the CPU
  cudaMemcpy(y, d_y, size_y, cudaMemcpyDeviceToHost);

  // Sum up the values
  float sum_y = 0.f;
  for(int i = 0; i < N; ++i){
    sum_y += y[i];
  }
  // Compute target for stencil and check for accuracy
  float target = x[0]+x[N-1];
  printf("error = %f\n", fabs(sum_y - target));

#if 0
  int num_trials = 10;
  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  for (int i = 0; i < num_trials; ++i){
    stencil_global <<< numBlocks, blockSize >>> (x, y, N, initial, final);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  
  printf("Time to run kernel 10x: %6.2f ms.\n", time);
  
#endif

  return 0;
}