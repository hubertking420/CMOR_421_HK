#include "functions.h"
#include <iostream>
#include <omp.h>

using namespace std;

int main(int argc, char * argv[]){
    int n = atoi(argv[1]);
    int trials = 5;
    cout << "Matrix size n = " << n << endl;

    // Allocate memory for A, x, and b
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

    // // Display linear system
    // std::cout << "Matrix A:" << std::endl;
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         std::cout << A[i * n + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "\nVector b:" << std::endl;
    // for (int i = 0; i < n; ++i) {
    //     std::cout << b[i] << std::endl;
    // }

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

    // Reset x
    for(int i = 0; i < n; ++i){
        x[i] = 0;
    }
    
    // Run time trials for static
    int sum_x_static = 0;
    double elapsed_time_static = omp_get_wtime();
    for(int i = 1; i < trials; ++i){
        back_solve_static(A, b, x, n);
    }
    elapsed_time_static = omp_get_wtime() - elapsed_time_static;
    for(int i = 0; i < n; ++i){
        sum_x_static += x[i];
    }
    
    // Reset x
    for(int i = 0; i < n; ++i){
        x[i] = 0;
    }

    // Run time trials for dynamic
    int sum_x_dynamic = 0;
    double elapsed_time_dynamic = omp_get_wtime();
    for(int i = 1; i < trials; ++i){
        back_solve_dynamic(A, b, x, n);
    }
    elapsed_time_dynamic = omp_get_wtime() - elapsed_time_dynamic;
    for(int i = 0; i < n; ++i){
        sum_x_dynamic += x[i];
    }

    // Display results
    cout << "Serial sum_x: " << sum_x_serial << endl;
    cout << "Average Elapsed Time:" << elapsed_time_serial << " seconds." << endl;
    cout << "Static Scheduling sum_x: " << sum_x_static << endl;
    cout << "Average Elapsed Time:" << elapsed_time_static << " seconds." << endl;
    cout << "Dynamic Scheduling sum_x: " << sum_x_dynamic << endl;
    cout << "Average Elapsed Time:" << elapsed_time_dynamic << " seconds." << endl;
    
    // Clean up memory
    delete[] A;
    delete[] x;
    delete[] b;
}