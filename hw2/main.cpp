#include "functions.h"
#include <iostream>
#include <omp.h>

using namespace std;

int main(int argc, char * argv[]){
    int n = atoi(argv[1]);
    int trials = 5;
   
    // Part 1
    cout << "Matrix size n = " << n << ", block size = " << BLOCK_SIZE << endl;

    double * A_1 = new double[n * n];
    double * B_1 = new double[n * n];
    double * C_1 = new double[n * n];

    // Initialize multiplication I x I = C
    for (int i = 0; i < n; ++i){
        A_1[i + i * n] = 1.0;
        B_1[i + i * n] = 1.0;
    }
    for (int i = 0; i < n * n; ++i){
        C_1[i] = 0.0;
    }

    // Measure serial blocked performance
    double elapsed_time_serial_matmul = omp_get_wtime();
    for (int i = 0; i < trials; ++i){
        matmul_blocked_serial(C_1, A_1, B_1, n);
    }
    elapsed_time_serial_matmul = omp_get_wtime() - elapsed_time_serial_matmul;
    double sum_C_serial = 0.0;
    for (int i = 0; i < n * n; ++i){
        sum_C_serial += C_1[i];
    }


    // reset C
    for (int i = 0; i < n * n; ++i){
        C_1[i] = 0.0;
    }

    // Measure parallel blocked performance 
    double elapsed_time_parallel_matmul = omp_get_wtime();
    for (int i = 0; i < trials; ++i){
        matmul_blocked_parallel(C_1, A_1, B_1, n);
    }
    elapsed_time_parallel_matmul = omp_get_wtime() - elapsed_time_parallel_matmul;
    double sum_C_parallel = 0.0;
    for (int i = 0; i < n * n; ++i){
        sum_C_parallel += C_1[i];
    }


    // reset C
    for (int i = 0; i < n * n; ++i){
        C_1[i] = 0.0;
    }

    // Measure parallel blocked performance 
    double elapsed_time_parallel_collapse = omp_get_wtime();
    for (int i = 0; i < trials; ++i){
        matmul_blocked_parallel_collapse(C_1, A_1, B_1, n);
    }
    elapsed_time_parallel_collapse = omp_get_wtime() -elapsed_time_parallel_collapse;
    double sum_C_collapse = 0.0;
    for (int i = 0; i < n * n; ++i){
        sum_C_collapse += C_1[i];
    }

    cout << "Blocked Matrix-Matrix Multiplication Algorithm:" << endl;
    cout << "Serial sum_C = " << sum_C_serial/trials << endl;
    cout << "Parallel sum_C = " << sum_C_parallel/trials << endl;
    cout << "Parallel collapse sum_C = " << sum_C_collapse/trials << endl;
    cout << "Serial elapsed time = " << elapsed_time_serial_matmul << endl;  
    cout << "Parallel elapsed time = " << elapsed_time_parallel_matmul << endl;
    cout << "Parallel collapse elapsed time= " << elapsed_time_parallel_collapse << endl;

    // Part 2
    // Allocate memory for A, x, and b
    double* A_2 = new double[n * n];
    double* x_2 = new double[n];
    double* b_2 = new double[n];

    // Initialize linear system
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i <= j) { // Upper triangular region
                A_2[i * n + j] = 1;
            } else { // Lower triangular region
                A_2[i * n + j] = 0;
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        b_2[i] = n - i;
        x_2[i] = 0;
    }

    // Run time trials for serial
    int sum_x_serial = 0;
    double elapsed_time_serial = omp_get_wtime();
    for(int i = 0; i < trials; ++i){
        back_solve_serial(A_2, b_2, x_2, n);
    }
    elapsed_time_serial = omp_get_wtime() - elapsed_time_serial;
    for(int i = 0; i < n; ++i){
        sum_x_serial += x_2[i];
    }

    
    // Run time trials for static
    int sum_x_static = 0;
    double elapsed_time_static = omp_get_wtime();
    for(int i = 0; i < trials; ++i){
        back_solve_static(A_2, b_2, x_2, n);

    }
    elapsed_time_static = omp_get_wtime() - elapsed_time_static;
    for(int i = 0; i < n; ++i){
        sum_x_static += x_2[i];
    }

    // Run time trials for dynamic
    int sum_x_dynamic = 0;
    double elapsed_time_dynamic = omp_get_wtime();
    for(int i = 0; i < trials; ++i){
        back_solve_dynamic(A_2, b_2, x_2, n);
    }
    elapsed_time_dynamic = omp_get_wtime() - elapsed_time_dynamic;
    for(int i = 0; i < n; ++i){
        sum_x_dynamic += x_2[i];
    }

    // Display results
    cout << "Back-Solve Algorithm:" << endl; 
    cout << "Serial sum_x = " << sum_x_serial << endl;
    cout << "Static Scheduling sum_x = " << sum_x_static << endl;
    cout << "Dynamic Scheduling sum_x = " << sum_x_dynamic << endl;
    cout << "Serial elapsed Time = " << elapsed_time_serial << " seconds." << endl;
    cout << "Static elapsed Time = " << elapsed_time_static << " seconds." << endl;
    cout << "Dynamic elapsed Time = " << elapsed_time_dynamic << " seconds." << endl;
    
    // Clean up memory 
    delete[] A_1;
    delete[] B_1;
    delete[] C_1;  
    delete[] A_2;
    delete[] x_2;
    delete[] b_2;
}