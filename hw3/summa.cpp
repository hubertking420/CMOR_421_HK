#include <iostream>
#include <cmath>
#include <cstring>
#include "mpi.h"

using namespace std;

int main(int argc, char * argv[]){
    int n = atoi(argv[1]);
    int trials = 5;
    bool verbose = true;
    bool display_A = true;
    bool display_B = true;
    bool display_C = true;

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
    // Allocate memory for recieving broadcasted partitions
    double * A_recv = new double[block_size * block_size];
    double * B_recv = new double[block_size * block_size]; 
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


    // Display the matrix A
    if(display_A){
        cout << "Parition of A on rank = " << rank << endl;
        for (int i = 0; i < n * n; ++i){
            cout << A_ij[i] << " ";
            if((i+1) % block_size == 0){
                cout << "\n";
            }
        }
    }
    // Display the matrix B
    if(display_B){
        cout << "Parition of B on rank = " << rank << endl;
        for (int i = 0; i < n * n; ++i){
            cout << B_ij[i] << " ";
            if((i+1) % block_size == 0){
                cout << "\n";
            }
        }
    }


    // Progress check
    MPI_Barrier(MPI_COMM_WORLD);
    if(verbose){
        cout << "Paritions recieved on rank: " << rank << endl;
    }

    // Row and column communicator 
    int row_color = rank / p; 
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row_color, rank, &row_comm);
    int col_color = rank % p;
    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, col_color, rank, &col_comm);    // Initial skew for A

    // Main computational loop
    for(int r = 0; r < p; ++r){
    	if(rank%p == r){
			memcpy(A_recv, A_ij, block_size*block_size*sizeof(double));
		}
		if(rank/p == r){
			memcpy(B_recv, B_ij, block_size*block_size*sizeof(double));
		}
        // Broadcast Aij and Bij across rows/cols
        MPI_Bcast(A_recv, block_size*block_size, MPI_DOUBLE, r, row_comm);
        MPI_Bcast(B_recv, block_size*block_size, MPI_DOUBLE, r, col_comm);  
        // Accumulate blocked outer product in Cij 
        for(int i = 0; i < block_size; ++i){
            for(int j = 0; j < block_size; ++j){
                for(int k = 0; k < block_size; ++k){
                    C_ij[i * block_size + j] += A_recv[i * block_size + k] * B_recv[k * block_size + j];
                }
            }
        }
    } 

    // Progress check
    MPI_Barrier(MPI_COMM_WORLD);
    if(verbose){
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
    if(verbose){
        cout << "Paritions written into C from rank = " << rank << endl;
    }


    // Stop clock
    double elapsed = MPI_Wtime()-start; 
    
    if(rank == 0){
        // Display the matrix C
        if(display_C){
            cout << "Matrix C: " << endl;
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
        cout << "SUMMA's sum_C = " << sum_C << endl;    
        delete[] C;
        cout << "Elapsed time: " << elapsed << endl; 
    }

    // Clear memory
    delete[] A_ij;
    delete[] B_ij;
    delete[] C_ij; 
    MPI_Finalize();

}
