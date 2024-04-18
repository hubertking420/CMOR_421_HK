#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 128
void check(int N, const float *A, float *x, float *y, float *y_target) {
    // Perform matrix-vector multiplication
    for (int i = 0; i < N; ++i) {
        float sum = 0.0f;  // Initialize the sum for the i-th element of y
        for (int j = 0; j < N; ++j) {
            sum += A[i * N + j] * x[j];  // Accumulate the dot product of the i-th row of A and x
        }
        y_target[i] = sum;  // Store the result in y_target
    }

    // Check element-wise for accuracy
    bool isCorrect = true;
    float tol = 1e-9;
    for (int i = 0; i < N; ++i) {
        float diff = fabs(y[i] - y_target[i]);  // Calculate the difference between computed and target
        if (diff >= tol) {  // If the difference exceeds the tolerance, the result is incorrect
            printf("y is not accurate to machine precision.\n");
            printf("At index %d, incorrect element = %f, correct element = %f\n", i, y[i], y_target[i]);
            isCorrect = false;
            break;  // Exit early on the first error
        }
    }
    if (isCorrect) {
        printf("y is accurate to machine precision.\n");
    }
}


__global__ void stencil_global(int N, const float *A, float *x, float *y, float bc_initial, float bc_final){
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N){
    float val = y[i];
    for (int j = 0; j < N; ++j){
      val += A[i + j * N] * x[j];  // coalesced
    }
    y[i] = val;
  }
  if(i==0) y[i] += bc_initial;
  else if(i==N-1) y[i] += bc_final;
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
  
  float * A = new float[N * N];
  float * x = new float[N];
  float * y = new float[N];
  float * y_target = new float[N];
  for (int i = 0; i < N; ++i) {
    // Initialize x and y
    x[i] = 1.f;
    y[i] = 0.f;    
    y_target[i] = 0.f;
    // Set indices for A
    int main = i*N+i;
    int super = i*N+(i+1);
    int sub = i*N+(i-1);
    // Set elements
    A[main] = 2.f;
    if (i < N - 1) {
        A[sub] = -1.f; 
    }
    if (i > 0) {
        A[super] = -1.f; 
    }
  } 

  // Define boundary conditions
  float bc_initial = 50.f;
  float bc_final = 50.f;

  // allocate memory and copy to the GPU
  float * d_A;
  float * d_x;
  float * d_y;
  int size_A = N*N*sizeof(float);  
  int size_x = N*sizeof(float);
  int size_y = N*sizeof(float);
  cudaMalloc((void **) &d_x, size_x);
  cudaMalloc((void **) &d_y, size_y);
  cudaMalloc((void **) &d_A, size_A);
  
  // copy memory over to the GPU
  cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, size_x, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, size_y, cudaMemcpyHostToDevice);
  stencil_global <<< numBlocks, blockSize >>> (N, d_A, d_x, d_y, bc_initial, bc_final);

  // copy memory back to the CPU
  cudaMemcpy(y, d_y, size_y, cudaMemcpyDeviceToHost);
  
  // Compute target for stencil and check for accuracy
  check(N, A, x, y, y_target);

#if 0
  int num_trials = 10;
  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  for (int i = 0; i < num_trials; ++i){
    stencil_global <<< numBlocks, blockSize >>> (N, A, x, y, bc_initial, bc_final);
  }


  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  
  printf("Time to run kernel 10x: %6.2f ms.\n", time);
  
#endif

  return 0;
}