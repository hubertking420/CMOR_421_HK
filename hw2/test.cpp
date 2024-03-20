#include <iostream>

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

    // Display linear system
    std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << A[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "\nVector b:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << b[i] << std::endl;
    }

    // Run time trials for serial
    int sum_x_serial = 0;
    for(int i = 0; i < trials; ++i){
        x[n-1] = b[n-1];
        for (int i = n-2; i >= 0; --i) {
            double sum = 0.0; 
            for (int j = i+1; j < n; ++j) { 
                sum += A[i*n+j] * x[j];
            }
            x[i] = b[i] - sum; 
        }
    }

    std::cout << "\nVector x:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << x[i] << std::endl;
    }

    for(int i = 0; i < n; ++i){
        sum_x_serial += x[i];
    }

    cout << "Serial sum_x: " << sum_x_serial << endl;

    // Clean up memory
    delete[] A;
    delete[] x;
    delete[] b;
}