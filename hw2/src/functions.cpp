#include "functions.h"
#include <omp.h>
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
    {
        for(int i = 0; i < n; ++i){
        x[i] = b[i];
    
        }
    }
    for (int j = n - 1; j >= 0; --j) {
        #pragma omp parallel for schedule(static)
        {
            for (int i = 0; i < j; ++i) {
                x[i] -= A[i * n + j] * x[j];
            }
        }

    }
}


void back_solve_dynamic(double *A, double *b, double *x, int n){
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel for schedule(dynamic)
    {
        for(int i = 0; i < n; ++i){
        x[i] = b[i];
    
        }
    }
    for (int j = n - 1; j >= 0; --j) {
        #pragma omp parallel for schedule(dynamic)
        {
            for (int i = 0; i < j; ++i) {
                x[i] -= A[i * n + j] * x[j];
            }
        }

    }
}