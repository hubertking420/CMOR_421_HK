#include <iostream>
#include <cmath>
#include "mpi.h"
using namespace std;

int main(int argc, char * argv[]){
    int n = atoi(argv[1]);
    int trials = 5;

    // Initialize multiplication I x I = C
    cout << "Matrix size n = " << n << endl;
    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    MPI_Status status;
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    int block_size = n / sqrt_p;
    int sqrt_p = (int)sqrt(size); 
    // Allocate memory for storing partitions
    double * A_ij = new double[block_size * block_size];
    double * B_ij = new double[block_size * block_size];
    double * C_ij = new double[block_size * block_size];
    // Allocate memory for recieving broadcasted partitions
    double * A_recv = new double[block_size * block_size];
    double * B_recv = new double[block_size * block_size];
    // Construct system on root rank i
    if (rank == 0) {
        for (int p = size-1; p > 0; --p) {
            // Calculate start and end index
            int row_start_p = (p / sqrt_p) * block_size;
            int col_start_p = (p % sqrt_p) * block_size;
            for(int i = 0; i < row_start_p; ++i){
                for(int j = 0; j < col_start_p; ++j){
                    A_ij[i*n + j] = 1.0;
                    B_ij[i*n + j] = 1.0;
                }
            }
            // Rank 0 sends the blocks to other processes
            if(p > 0){
                MPI_Send(A_ij, block_size*block_size, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                MPI_Send(B_ij, block_size*block_size, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            } 
        } 
    } else {
        MPI_Recv(A_ij, block_size*block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
        MPI_Recv(B_ij, block_size*block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
    }   
    // Row and column communicator 
    int row_color = rank / p; 
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row_color, rank, &row_comm);
    int col_color = rank % p;
    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, col_color, rank, &col_comm);
    // Compute p local outer products 
    for(int rank = 0; rank < p; ++rank){
        // Broadcast Aij and Bij across rows/cols
        MPI_Bcast(A_recv, block_size*block_size, MPI_DOUBLE, rank, row_comm);
        MPI_Bcast(B_recv, block_size*block_size, MPI_DOUBLE, rank, col_comm);  
        // Accumulate blocked outer product in Cij 
        for(int i = 0; i < block_size; ++i){
            for(int j = 0; j < block_size; ++j){
                for(int k = 0; k < block_size; ++k){
                    C_ij[i * block_size + j] += A_recv[i * block_size + k] * B_recv[k * block_size + j];
                }
            }
        }
    } 
    double *C = nullptr;
    if(rank == 0) {
        C = new double[n * n];  // Allocate space for the full matrix on rank 0

        // Directly place rank 0's data into C
        // This step requires correct calculation of where rank 0's data belongs in the full matrix
        // Assuming a simple scheme where rank 0 holds the top-left block
        for(int i = 0; i < block_size; ++i){
            for(int j = 0; j < block_size; ++j){
                C[i * n + j] = C_ij[i * block_size + j];
            }
        }

        // Receive blocks from other ranks
        for(int p = 1; p < size; ++p) {
            // Calculate the source rank's block's correct position in C
            int row_start_p = (p / sqrt(size)) * block_size;
            int col_start_p = (p % (int)sqrt(size)) * block_size;
        
            MPI_Recv(C_ij, block_size * block_size, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Place received data into C
            for(int i = 0; i < block_size; ++i){
                for(int j = 0; j < block_size; ++j){
                    C[(row_start_p+i)*n + (col_start_p+j)] = C_ij[i*block_size+j];
                }
            }
        }
    }
    else {
        // Non-root ranks send their C_ij block to rank 0
        MPI_Send(C_ij, block_size * block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    // Clear local memory
    delete[] A_recv;
    delete[] B_recv;
    delete[] A_ij;
    delete[] B_ij;
    delete[] C_ij;
    delete[] C;
    double elapsed = MPI_Wtime() - start;
    MPI_Finalize();
    cout << "SUMMA elapsed time = " << elapsed << endl; 
}
