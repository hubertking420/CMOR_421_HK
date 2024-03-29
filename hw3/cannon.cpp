#include <iostream>
#include <cmath>
#include "mpi.h"
using namespace std;

int main(int argc, char * argv[]){
    int n = atoi(argv[1]);
    int p = atoi(argv[2]);
    int trials = 5;

    // Initialize multiplication I x I = C
    cout << "Matrix size n = " << n << ", block size = " << BLOCK_SIZE << endl;
    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status; 
    int block_size = n/p; 

    // Split nxn matrix into blocked matrix with pxp blocks
    if (rank == 0){
        double * A_1 = new double[n * n];
        double * B_1 = new double[n * n];
        double * C_1 = new double[n * n];
        for (int i = rank; i < n; i += size){
            A_1[i] = 1.0;
            B_1[i] = 1.0;
        }
        for (int i = 0; i < n * n; ++i){
            C_1[i] = 0.0;
        }
    }    
    double * A_ij = new double[block_size * block_size];
    for(int i = 0; i < n*n; ++i){

    }
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row_index, rank, &row_comm); // Split by row
    MPI_Comm_split(MPI_COMM_WORLD, col_index, rank, &col_comm); // Split by column   
  
    int row_color = rank / p;
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row_color, rank, &row_comm);

    int col_color = rank % p;
    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, col_color, rank, &col_comm);   
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
 
    delete A_1;
    delete B_1;
    delete C_1;
    
}