#include <iostream>
#include "functions.h"
#include <random>

using namespace std;

int main(int arc, char* argv[]){ 
    int n = atoi(argv[1]);
    int num_trials = 5;
    cout << "Matrix size n = " << n << endl;

    // make A, B = random
    double * A = new double[n * n];
    double * B = new double[n * n];
    double * C = new double[n * n];    
    random_device rd;
    default_random_engine eng(rd());
    uniform_real_distribution<double> distr(0.0, 10.0);
    for (int i = 0; i < n; ++i){
        A[i + i * n] = distr(eng);
        B[i + i * n] = distr(eng);
    }
    for (int i = 0; i < n * n; ++i){
        C[i] = 0.0;
    }

    bool verbose = false;
    bool display_A = false;
    bool display_B = false;
    bool display_C = false;

    // Serial
    matmul_naive(n, C, A, B);
    reset(n, C);

    // SUMMA
    summa(n, C, A, B, verbose, display_A, display_B, display_C); 
    reset(n, C);

    // Cannon's
    cannon(n, C, A, B, verbose, display_A, display_B, display_C);
    reset(n, C);

    delete[] A;
    delete[] B;
    delete[] C;  
  
    return 0;
}