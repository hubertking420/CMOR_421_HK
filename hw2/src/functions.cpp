#include "functions.h"
#include <omp.h>

void matmul_blocked_serial(double *C, double *A, double *B, int n){
    for (int i = 0; i < n; i += BLOCK_SIZE){
        for (int j = 0; j < n; j += BLOCK_SIZE){
            for (int k = 0; k < n; k += BLOCK_SIZE){

                // small matmul
                for (int ii = i; ii < i + BLOCK_SIZE; ii++){
                    for (int jj = j; jj < j + BLOCK_SIZE; jj++){
                        double Cij = C[jj + ii * n];
                        for (int kk = k; kk < k + BLOCK_SIZE; kk++){
                            Cij += A[kk + ii * n] * B[jj + kk * n]; // Aik * Bkj
                        }
                        C[jj + ii * n] = Cij;
                    }
                }

            }
        }
    }
}

void matmul_blocked_parallel(double *C, double *A, double *B, int n){
    #pragma omp parallel for
    for (int i = 0; i < n; i += BLOCK_SIZE){
        #pragma omp parallel for
        for (int j = 0; j < n; j += BLOCK_SIZE){
            for (int k = 0; k < n; k += BLOCK_SIZE){

                // small matmul
                for (int ii = i; ii < i + BLOCK_SIZE; ii++){
                    for (int jj = j; jj < j + BLOCK_SIZE; jj++){
                        double Cij = C[jj + ii * n];
                        for (int kk = k; kk < k + BLOCK_SIZE; kk++){
                            Cij += A[kk + ii * n] * B[jj + kk * n]; // Aik * Bkj
                        }
                        C[jj + ii * n] = Cij;
                    }
                }

            }
        }
    }
}

void matmul_blocked_parallel_collapse(double *C, double *A, double *B, int n){
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i += BLOCK_SIZE){
        for (int j = 0; j < n; j += BLOCK_SIZE){
            for (int k = 0; k < n; k += BLOCK_SIZE){

                // small matmul
                for (int ii = i; ii < i + BLOCK_SIZE; ii++){
                    for (int jj = j; jj < j + BLOCK_SIZE; jj++){
                        double Cij = C[jj + ii * n];
                        for (int kk = k; kk < k + BLOCK_SIZE; kk++){
                            Cij += A[kk + ii * n] * B[jj + kk * n]; // Aik * Bkj
                        }
                        C[jj + ii * n] = Cij;
                    }
                }

            }
        }
    }
}

void back_solve_serial(double *A, double *b, double *x, int n){
    for(int i = 0; i < n; ++i){
        x[i] = b[i];
    }
    
    for (int j = n - 1; j >= 0; --j) {
        for (int i = 0; i < j; ++i) {
            x[i] -= A[i * n + j] * x[j];
        }
    }
}

void back_solve_static(double *A, double *b, double *x, int n){
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < n; ++i){
    x[i] = b[i];
    }
    for (int j = n - 1; j >= 0; --j) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < j; ++i) {
            x[i] -= A[i * n + j] * x[j];
        }
    }
}


void back_solve_dynamic(double *A, double *b, double *x, int n){
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < n; ++i){
    x[i] = b[i];
    }
    for (int j = n - 1; j >= 0; --j) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < j; ++i) {
            x[i] -= A[i * n + j] * x[j];
        }
    }
}