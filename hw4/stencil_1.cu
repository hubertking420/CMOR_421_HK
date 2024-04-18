#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 128
void check(const float *x, float *y_target, float *y, int N) {
    // Compute solution
    y_target[0] = x[0]; // Assuming some boundary condition or direct assignment
    y_target[N-1] = x[N-1]; // Assuming some boundary condition or direct assignment
    for (int i = 1; i < N-1; ++i) {
        y_target[i] = 2*x[i] - x[i-1] - x[i+1];
    }

    // Check element-wise
    bool isCorrect = true;
    float tol = 1e-9;
    for (int i = 0; i < N; ++i) {
        float diff = fabs(y[i] - y_target[i]);
        if (diff >= tol) {
            printf("y is not accurate to machine precision.\n");
            printf("At index %d, incorrect element = %f, correct element = %f\n", i, y[i], y_target[i]);
            isCorrect = false;
            break; // Exiting early on first error
        }
    }

    if (isCorrect) {
        printf("y is accurate to machine precision.\n");
    }
}

__global__ void stencil_global(const float *x, float *y, int N, float bc_initial, float bc_final){
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  y[i]=0.f;

  // Boundary conditions
  if(i == 0){
    y[i] = bc_initial;
  }
  if(i == N-1){
    y[i] = bc_final;
  }
  
  // Rest of stencil relies on BCs being applied
  __syncthreads();

  // Interior elements
  if(i>0 && i<N-1){
    y[i] = 2*x[i]-x[i-1]-x[i+1];
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
  float * y_target = new float[N];

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

  stencil_global <<< numBlocks, blockSize >>> (d_x, d_y, N, bc_initial, bc_final);

  // copy memory back to the CPU
  cudaMemcpy(y, d_y, size_y, cudaMemcpyDeviceToHost);

  
  // Compute target for stencil and check for accuracy
  check(x, y, y_target, N);

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