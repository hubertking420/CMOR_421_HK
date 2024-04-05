#include <iostream>
#include "functions.h"
#include <random>
#include "mpi.h"

using namespace std;

int main(int arc, char* argv[]){ 
    // Initialize mpi environment
    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    MPI_Status status;
    MPI_Barrier(MPI_COMM_WORLD);

    int n = atoi(argv[1]);

    if(rank == 0){
        cout << "Matrix size n = " << n << endl;
    }

    // make A, B = random
    double *A = new double[n*n];
    double *B = new double[n*n];
    double *C_1 = new double[n*n];    
    double *C_2 = new double[n*n];    
    random_device rd;
    default_random_engine eng(rd());
    uniform_real_distribution<double> distr(0.0, 10.0);
    for(int i = 0; i < n*n; ++i){
        A[i] = distr(eng);
        B[i] = distr(eng);
    }
    for(int i = 0; i < n*n; ++i){
        C_1[i] = 0.0;
        C_2[i] = 0.0;
    } 


    bool verbose = false;
    bool display_A = false;
    bool display_B = false;
    bool display_C = false;
   
    
    // Serial
    if(rank == 0){
        matmul_naive(n, C_1, A, B);
    }

    // Cannon's
    cannon(n, rank, size, C_2, A, B, verbose, display_A, display_B, display_C);
    if(rank == 0){
        if(check_equal(n, C_1, C_2)){
            cout << "Serial product and Cannon's product are equal to machine precision." << endl;
        } else{
            cout << "Serial product and Cannon's product not equal."
        }
    }


    // Kill mpi environment
    MPI_Finalize();

    delete[] A;
    delete[] B;
    delete[] C;  
  
    return 0;
}