#include <iostream>
#include "functions.h"
#include <random>
#include "mpi.h"

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
   
    
    // Initialize mpi environment
    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    MPI_Status status;
    MPI_Barrier(MPI_COMM_WORLD);


    // Serial
    matmul_naive(n, C, A, B);
    if(rank == 0){
        reset(n, C);
    }

    // SUMMA
    summa(n, rank, size C, A, B, verbose, display_A, display_B, display_C); 
    if(rank == 0){
        reset(n, C);
    }

    // Cannon's
    cannon(n, rank, size, C, A, B, verbose, display_A, display_B, display_C);
    if(rank == 0){
        reset(n, C);
    }

    // Kill mpi environment
    MPI_Finalize();

    delete[] A;
    delete[] B;
    delete[] C;  
  
    return 0;
}