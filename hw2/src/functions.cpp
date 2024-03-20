#include "functions.h"
#include <omp.h>


void back_solve_static(double *A, double *b, double *x, int n){
    omp_set_num_threads(NUM_THREADS);
    x[n-1] = b[n-1];
    for (int i = n-2; i >= 0; --i) {
        double sum = 0.0; 
        #pragma omp parallel for schedule(static) reduction(+:sum)
        {
            for (int j = i+1; j < n; ++j) { 
                sum += A[i*n+j] * x[j];
            }
            x[i] = b[i] - sum; 
        }
    }
}


void back_solve_dynamic(double *A, double *b, double *x, int n){
    omp_set_num_threads(NUM_THREADS);
    x[n-1] = b[n-1];
    for (int i = n-2; i >= 0; --i) {
        double sum = 0.0; 
        #pragma omp parallel for schedule(dynamic) reduction(+:sum)
        {
            for (int j = i+1; j < n; ++j) { 
                sum += A[i*n+j] * x[j];
            }
            x[i] = b[i] - sum; 
        }
    }
}

void back_solve_serial(double *A, double *b, double *x, int n){
    x[n-1] = b[n-1];
    for (int i = n-2; i >= 0; --i) {
        double sum = 0.0; 
        for (int j = i+1; j < n; ++j) { 
            sum += A[i*n+j] * x[j];
        }
        x[i] = b[i] - sum; 
    }
}