#include <iostream>
#include <cmath>
#include <cstring>
#include "mpi.h"
using namespace std;

int main(int argc, char * argv[]){
    int n = atoi(argv[1]);
    int trials = 5;

    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    MPI_Status status;

    if(rank == 0){
        cout << "Matrix size n = " << n << endl;
    }

    // Dimensions of mesh
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    int s = size;
    int p = (int)sqrt(s);
    int block_size = n / p;

    // Allocate memory for storing partitions
    double * A_ij = new double[block_size * block_size];
    double * B_ij = new double[block_size * block_size];
    double * C_ij = new double[block_size * block_size];
    // Construct system on root rank i
    if (rank == 0) {
        for (int k = size-1; k > 0; --k) {
            // Calculate start and end index
            int row_start_p = (k / p) * block_size;
            int col_start_p = (k % p) * block_size;
            for(int i = 0; i < row_start_p; ++i){
                for(int j = 0; j < col_start_p; ++j){
                    A_ij[i*n + j] = 1.0;
                    B_ij[i*n + j] = 1.0;
                }
            }
            // Rank 0 sends the blocks to other processes
            if(k > 0){
                MPI_Send(A_ij, block_size*block_size, MPI_DOUBLE, k, 0, MPI_COMM_WORLD);
                MPI_Send(B_ij, block_size*block_size, MPI_DOUBLE, k, 0, MPI_COMM_WORLD);
            } 
        } 
    } else {
        MPI_Recv(A_ij, block_size*block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
        MPI_Recv(B_ij, block_size*block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
    }   
    MPI_Barrier(MPI_COMM_WORLD);
    cout << "Paritions recieved on rank: " << rank << endl;

    // Map 2D coordinates (x, y) to 1D processor labels
    int x = rank / p;
    int y = rank % p;
    // Initial skew for A
    // Shift each row of A left by 'x' positions
    int origin_A = x * p + (y - x + p) % p; // Source rank for A submatrix
    int dest_A = x * p + (y + x) % p;       // Destination rank for A submatrix
    MPI_Sendrecv_replace(A_ij, block_size * block_size, MPI_DOUBLE, dest_A, 0,
                        origin_A, 0, MPI_COMM_WORLD, &status);

    // Initial skew for B
    // Shift each column of B up by 'y' positions
    int origin_B = ((x - y + p) % p) * p + y; // Source rank for B submatrix
    int dest_B = ((x + y) % p) * p + y;       // Destination rank for B submatrix
    MPI_Sendrecv_replace(B_ij, block_size * block_size, MPI_DOUBLE, dest_B, 0,
                        origin_B, 0, MPI_COMM_WORLD, &status);
    MPI_Barrier(MPI_COMM_WORLD);
    cout << "Inital skew of A and B completed on rank: " << rank << endl;

    // Main computational loop
    for(int k = 0; k < p; ++k){ 
        // Accumulation
        for(int i = 0; i < block_size; ++i){
            for(int j = 0; j < block_size; ++j){
                for(int k = 0; k < block_size; ++k){
                    C_ij[i * block_size + j] += A_ij[i * block_size + k] * B_ij[k * block_size + j];
                }
            }
        }
        // Distribute data for next ieration 
        int dest_A = x * p + (y - 1 + p) % p;
        int dest_B = ((x - 1 + p) % p) * p + y;
        int origin_A = x * p + (y + 1) % p;
        int origin_B = ((x + 1) % p) * p + y;
        MPI_Sendrecv_replace(A_ij, block_size*blocksize, MPI_DOUBLE, dest_A, 0, A_ij, block_size*blocksize, MPI_DOUBLE, origin_A, 0, MPI_COMM_WORLD, status);
        MPI_Sendrecv_replace(B_ij, block_size*blocksize, MPI_DOUBLE, dest_B, 0, B_ij, block_size*blocksize, MPI_DOUBLE, origin_B, 0, MPI_COMM_WORLD, status); 
    }
    MPI_Barrier(MPI_COMM_WORLD);
    cout << "Paritions of C computed on rank: " << rank << endl;
 

    // Gather blocks of C
    double *C = new double[n*n];

    MPI_Barrier(MPI_COMM_WORLD);
    cout << "Computation completed on rank: " << rank << endl;
 
    // Clear local memory
    delete[] A_ij;
    delete[] B_ij;
    delete[] C_ij;
    if(rank == 0){
        delete[] C;
        cout << "Elapsed time: " << endl;
    }
    MPI_Finalize();


}
