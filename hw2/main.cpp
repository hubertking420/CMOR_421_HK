#include "functions.h"
#include <iostream>
#include <omp.h>

using namespace std;

int main(int argc, char * argv[]){
    int n = atoi(argv[1]);
    int trials = 5;
   
    // Part 1
    cout << "Matrix size n = " << n << ", block size = " << BLOCK_SIZE << endl;

    double * A = new double[n * n];
    double * B = new double[n * n];
    double * C = new double[n * n];

    // Initialize multiplication I x I = C
    for (int i = 0; i < n; ++i){
        A[i + i * n] = 1.0;
        B[i + i * n] = 1.0;
    }
    for (int i = 0; i < n * n; ++i){
        C[i] = 0.0;
    }

    // Measure serial blocked performance
    double elapsed_time_serial_matmul = omp_get_wtime();
    for (int i = 0; i < trials; ++i){
        matmul_blocked_serial(C, A, B, n);
    }
    elapsed_time_serial_matmul = omp_get_wtime() - elapsed_time_serial_matmul;
    double sum_C_serial = 0.0;
    for (int i = 0; i < n * n; ++i){
        sum_C_serial += C[i];
    }


    // reset C
    for (int i = 0; i < n * n; ++i){
        C[i] = 0.0;
    }

    // Measure parallel blocked performance 
    double elapsed_time_parallel_matmul = omp_get_wtime();
    for (int i = 0; i < trials; ++i){
        matmul_blocked_parallel(C, A, B, n);
    }
    elapsed_time_parallel_matmul = omp_get_wtime() - elapsed_time_parallel_matmul;
    double sum_C_parallel = 0.0;
    for (int i = 0; i < n * n; ++i){
        sum_C_parallel += C[i];
    }


    // reset C
    for (int i = 0; i < n * n; ++i){
        C[i] = 0.0;
    }

    // Measure parallel blocked performance 
    double elapsed_time_parallel_collapse = omp_get_wtime();
    for (int i = 0; i < trials; ++i){
        matmul_blocked_parallel_collapse(C, A, B, n);
    }
    elapsed_time_parallel_collapse = omp_get_wtime() -elapsed_time_parallel_collapse;
    double sum_C_collapse = 0.0;
    for (int i = 0; i < n * n; ++i){
        sum_C_collapse += C[i];
    }


    cout << "Serial sum_C = " << sum_C_serial/trials << endl;
    cout << "Parallel sum_C = " << sum_C_parallel/trials << endl;
    cout << "Parallel collapse sum_C = " << sum_C_collapse/trials << endl;
    cout << "Serial elapsed time = " << elapsed_time_serial_matmul << endl;  
    cout << "Parallel elapsed time = " << elapsed_time_parallel_matmul << endl;
    cout << "Parallel collapse elapsed time= " << elapsed_time_parallel_collapse << endl;

    delete[] A;
    delete[] B;
    delete[] C;  

    // Part 2
    // Allocate memory for A, x, and b
    cout << "Matrix size n = " << n << endl;
    double* A = new double[n * n];
    double* x = new double[n];
    double* b = new double[n];

    // Initialize linear system
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i <= j) { // Upper triangular region
                A[i * n + j] = 1;
            } else { // Lower triangular region
                A[i * n + j] = 0;
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        b[i] = n - i;
        x[i] = 0;
    }

    // Run time trials for serial
    int sum_x_serial = 0;
    double elapsed_time_serial = omp_get_wtime();
    for(int i = 0; i < trials; ++i){
        back_solve_serial(A, b, x, n);
    }
    elapsed_time_serial = omp_get_wtime() - elapsed_time_serial;
    for(int i = 0; i < n; ++i){
        sum_x_serial += x[i];
    }

    
    // Run time trials for static
    int sum_x_static = 0;
    double elapsed_time_static = omp_get_wtime();
    for(int i = 0; i < trials; ++i){
        back_solve_static(A, b, x, n);
    }
    elapsed_time_static = omp_get_wtime() - elapsed_time_static;
    for(int i = 0; i < n; ++i){
        sum_x_static += x[i];
    }

    // Run time trials for dynamic
    int sum_x_dynamic = 0;
    double elapsed_time_dynamic = omp_get_wtime();
    for(int i = 0; i < trials; ++i){
        back_solve_dynamic(A, b, x, n);
    }
    elapsed_time_dynamic = omp_get_wtime() - elapsed_time_dynamic;
    for(int i = 0; i < n; ++i){
        sum_x_dynamic += x[i];
    }

    // Display results
    cout << "Serial sum_x = " << sum_x_serial << endl;
    cout << "Static Scheduling sum_x = " << sum_x_static << endl;
    cout << "Dynamic Scheduling sum_x = " << sum_x_dynamic << endl;
    cout << "Serial elapsed Time = " << elapsed_time_serial << " seconds." << endl;
    cout << "Static elapsed Time = " << elapsed_time_static << " seconds." << endl;
    cout << "Dynamic elapsed Time = " << elapsed_time_dynamic << " seconds." << endl;
    
    // Clean up memory
    delete[] A;
    delete[] x;
    delete[] b;
}