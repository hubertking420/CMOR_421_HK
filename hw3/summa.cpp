#include <iostream>
#include <cmath>
#include "mpi.h"
using namespace std;

int main(int argc, char * argv[]){
    int n = atoi(argv[1]);
    int trials = 5;

    // Initialize multiplication I x I = C
    cout << "Matrix size n = " << n << ", block size = " << BLOCK_SIZE << endl;
    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    MPI_Status status;
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    int block_size = n / (int)sqrt(size); 
    // Allocate final matrix product
    if(rank == 0){
        double * C = new double[n * n];
    }
    if (rank == 0) {
        for(int i = 0; i < block_size; ++i){
            for(int j = 0; j < block_size; ++j){
                C[i*n + j] = C_ij[i*n + j];
            }
        }
        for (int p = 1; p < size; ++p) { 
            MPI_Send(C_ij, block_size*block_size, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            MPI_Recv(C_ij, block_size*block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
            // Calculate start and end index
            int row_start_p = (p / 2) * block_size;
            int col_start_p = (p % 2) * block_size;
            for(int i = row_start_p; i < row_start_p+block_size; ++i){
                for(int j = 0; j < col_start_p; ++j){
                    C[i*n + j] = C_ij[i*n + j];
                }
            } 
        } 
    }
    }
    // Allocate memory for storing partitions
    double * A_ij = new double[block_size * block_size];
    double * B_ij = new double[block_size * block_size];
    double * C_ij = new double[block_size * block_size];
    // Allocate memory for recieving broadcasted partitions
    double * A_recv = new double[block_size * block_size];
    double * B_recv = new double[block_size * block_size];
    // Construct system on root rank i
    // TODO: use MPI_scatter
    if (rank == 0) {
        for (int p = 0; p < size; ++p) {
            // Calculate start and end index
            int row_start_p = (p / 2) * block_size;
            int col_start_p = (p % 2) * block_size;
            for(int i = 0; i < row_start_p; ++i){
                for(int j = 0; j < col_start_p; ++j){
                    A[i*n + j] = 1.0;
                    B[i*n + j] = 1.0;
                }
            }
            // Rank 0 sends the blocks to other processes
            if(p > 0){
                MPI_Send(A_ij, block_size*block_size, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                MPI_Send(B_ij, block_size*block_size, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                MPI_Send(C_ij, block_size*block_size, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
           } 
        } 
    }
    else {
        // Rank > 0 recieves partitions
        for(int p = 1; p < size; ++p){
            MPI_Recv(A_ij, block_size*block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(B_ij, block_size*block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
            MPI_Recv(C_ij, block_size*block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }   
    }
    // Row and column communicator 
    int row_color = rank / p; 
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row_color, rank, &row_comm);
    int col_color = rank % p;
    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, col_color, rank, &col_comm);
    // Compute p local outer products (op) 
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
    // Send all partitions to rank 0
    if(rank != 0){
        for(int p = 1; p < size; ++p){
            MPI_Send(C_ij, block_size*block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        } 
    }
    // Construct C using partitions Cij on root rank
    else{
        for(int p = 0; p < size; ++p) {         
            if(p > 0){
                MPI_Recv(C_ij, block_size*block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
            }    
            // Calculate start and end index
            int row_start_p = (p / 2) * block_size;
            int col_start_p = (p % 2) * block_size;
            for(int i = row_start_p; i < row_start_p+block_size; ++i){
                for(int j = 0; j < col_start_p; ++j){
                    C[i*n + j] = C_ij[i*n + j];
                }
            } 
        } 
    }
    // Clear local memory
    delete[] A_recv;
    delete[] B_recv;
    delete[] A_ij;
    delete[] B_ij;
    delete[] C_ij;

    double elapsed = MPI_Wtime()-start;
    MPI_Finalize();
    delete A_1;
    delete B_1;
    delete C_1;
    
}
