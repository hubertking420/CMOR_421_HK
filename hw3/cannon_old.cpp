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
    MPI_Barrier(MPI_COMM_WORLD);

    // Construct system (make this function param)
    double * A = new double[n * n];
    double * B = new double[n * n];
    for (int i = 0; i < n * n; ++i){
        A[i] = 1.0;
        B[i] = 1.0;
    }

    if(rank == 0){
        cout << "Matrix size n = " << n << endl;
    }

    // Start time
    double start = MPI_Wtime();

    // Dimensions of mesh
    int s = size;
    int p = (int)sqrt(s);
    int block_size = n / p;

    // Allocate memory for storing partitions
    double * A_ij = new double[block_size * block_size];
    double * B_ij = new double[block_size * block_size];
    double * C_ij = new double[block_size * block_size];

    // Construct system on root rank
    if (rank == 0) {
        for (int k = size-1; k > 0; --k) {
            // Calculate the starting indices for parition of C
            int row_start = (k/p)*block_size;
            int col_start = (k%p)*block_size;
            
            // Place received data into C
            for(int i = 0; i < block_size; ++i){
                for(int j = 0; j < block_size; ++j){
                    A_ij[i*block_size+j] = A[(row_start+i)*n + (col_start+j)];
                    B_ij[i*block_size+j] = B[(row_start+i)*n + (col_start+j)];
               }
            }
            
            // Rank 0 sends the blocks to other processes
            if(k > 0){
                MPI_Send(A_ij, block_size*block_size, MPI_DOUBLE, k, 0, MPI_COMM_WORLD);
                MPI_Send(B_ij, block_size*block_size, MPI_DOUBLE, k, 0, MPI_COMM_WORLD);
            } 
        } 
    } else {
        // All other processes receive blocks
        MPI_Recv(A_ij, block_size*block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
        MPI_Recv(B_ij, block_size*block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
    }

    // Progress check
    MPI_Barrier(MPI_COMM_WORLD);
    bool check_paritions_recieved = false;
    if(check_paritions_recieved){
        cout << "Paritions recieved on rank: " << rank << endl;
    }

    // Map 2D coordinates (x, y) to 1D processor labels
    int x = rank / p;
    int y = rank % p;
    // Initial skew for A
    int origin_A = x * p + (y - x + p) % p; // Source rank for A submatrix
    int dest_A = x * p + (y + x) % p;       // Destination rank for A submatrix
    MPI_Sendrecv_replace(A_ij, block_size*block_size, MPI_DOUBLE, dest_A, 0,
                        origin_A, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Initial skew for B
    int origin_B = ((x - y + p) % p) * p + y; // Source rank for B submatrix
    int dest_B = ((x + y) % p) * p + y;       // Destination rank for B submatrix
    MPI_Sendrecv_replace(B_ij, block_size*block_size, MPI_DOUBLE, dest_B, 0,
                        origin_B, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


    // Progress check
    MPI_Barrier(MPI_COMM_WORLD);
    bool check_initial_skew = false;
    if(check_initial_skew){
        cout << "Inital skew of A and B completed on rank: " << rank << endl;
    }


    // Main computational loop
    for(int k = 0; k < p; ++k){ 
        // Accumulation into C
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

        MPI_Sendrecv_replace(A_ij, block_size*block_size, MPI_DOUBLE, dest_A, 0, origin_A, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(B_ij, block_size*block_size, MPI_DOUBLE, dest_B, 0, origin_B, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
    }


    // Progress check
    MPI_Barrier(MPI_COMM_WORLD);
    bool check_paritions_computed = true;
    if(check_paritions_computed){
        cout << "Partitions of C computed on rank: " << rank << endl;
    }


    // Gather blocks of C
    double *C = nullptr;
    if(rank == 0) {
        // Allocate space for the full matrix on rank 0
        C = new double[n * n];
        for(int i = 0; i < block_size; ++i){
            for(int j = 0; j < block_size; ++j){
                C[i * n + j] = C_ij[i * block_size + j];
            }
        }
        for(int k = 1; k < size; ++k) {
            // Calculate the starting indices for parition of C
            int row_start_p = (k/p)*block_size;
            int col_start_p = (k%p)*block_size;

            // Receive blocks from other ranks        
            MPI_Recv(C_ij, block_size*block_size, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

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
        MPI_Send(C_ij, block_size*block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }


    // Progress check
    MPI_Barrier(MPI_COMM_WORLD);
    bool check_gather = false;
    if(check_gather){
        cout << "Paritions written into C from rank = " << rank << endl;
    }


    // Stop clock
    double elapsed = MPI_Wtime()-start;
    
    // Display the matrix A
    bool display_A = false;
    if(rank == 0){
        if(display_A){
            cout << "Matrix A:" << endl;
            for (int i = 0; i < n * n; ++i){
                cout << A[i] << " ";
                if((i+1) % block_size == 0){
                    cout << "\n";
                }
            }
        }
    }
   // Display the matrix B
   bool display_B = false;
    if(rank == 0){
        if(display_B){
            cout << "Matrix B:" << endl;
            for (int i = 0; i < n * n; ++i){
                cout << B[i] << " ";
                if((i+1) % block_size == 0){
                    cout << "\n";
                }
            }
        }
    }
    // Display the matrix C
    bool display_C = true;
    if(rank == 0){
        // Display the matrix C
        cout << "Matrix C:" << endl;
        if(display_C){
            for (int i = 0; i < n * n; ++i){
                cout << C[i] << " ";
                if((i+1) % n == 0){
                    cout << "\n";
                }
            }
        }
        double sum_C = 0.0;
        for (int i = 0; i < n * n; ++i){           
            sum_C += C[i];
        }
        cout << "Cannon's sum_C = " << sum_C << endl;    
        delete[] C;
        cout << "Elapsed time: " << elapsed << endl;
    }


    // Clear memory
    delete[] A_ij;
    delete[] B_ij;
    delete[] C_ij; 
    MPI_Finalize();

}