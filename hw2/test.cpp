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
                A[i * n + j] = 1.0;
            } else { // Lower triangular region
                A[i * n + j] = 0.0;
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        b[i] = n - i;
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

    // // Back Solve
    int sum_x_serial = 0;
    for(int i = 0; i < n; ++i){
        x[i] = b[i];
    }
    for (int j = n - 1; j >= 0; --j) {
        for (int i = 0; i < j; ++i) {
            x[i] -= A[i * n + j] * x[j];
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