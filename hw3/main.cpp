#include <iostream>
#include <chrono>
#include <random>
#include "functions.h"
// #include "mpi.h"

using namespace std;
using namespace std::chrono;

int main(int arc, char* argv[]){ 
    int n = atoi(argv[1]);
    cout << "Matrix size n = " << n << endl;
  
    double * A = new double[n * n];
    double * B = new double[n * n];
    double * C = new double[n * n];
    
    // Seed with a real random value, if available
    random_device rd;
    default_random_engine eng(rd());
    uniform_real_distribution<double> distr(0.0, 10.0);
    
    // make A, B = random
    for (int i = 0; i < n; ++i){
        A[i + i * n] = distr(eng);
        B[i + i * n] = distr(eng);
    }
    for (int i = 0; i < n * n; ++i){
        C[i] = 0.0;
    }

    int num_trials = 5;

    // Measure performance
    high_resolution_clock::time_point start = high_resolution_clock::now();
    for (int i = 0; i < num_trials; ++i){
        matmul_naive(n, C, A, B);
    }
    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double> elapsed_naive = (end - start) / num_trials;

    // Calculate sum
    double sum_C = 0.0;
    for (int i = 0; i < n * n; ++i){
        sum_C += C[i];
    }
    cout << "Serial sum_C = " << sum_C << endl;
    
    
    
    // // Calculate sum
    // double sum_C = 0.0;
    // for (int i = 0; i < n * n; ++i){
    //     sum_C += C[i];
    // }
    // cout << "SUMMA sum_C = " << sum_C << endl;
    
    
    
    // cout << "Cannon's sum_C = " << sum_C << endl;  
    cout << "Serial elapsed time (ms) = " << elapsed_naive.count() * 1000 << endl;
    // cout << "SUMMA elapsed time (ms) = " << elapsed_summa << endl;
    // cout << "Cannon's elapsed time (sec) = " << elapsed_cannon << endl;  

    delete[] A;
    delete[] B;
    delete[] C;  
  
    return 0;
}