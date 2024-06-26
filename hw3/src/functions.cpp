#include "functions.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <chrono>
#include "mpi.h"

using namespace std;
using namespace std::chrono;

void matmul_naive(int n, double* C, double* A, double* B, bool display_C){
    high_resolution_clock::time_point start = high_resolution_clock::now(); 
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            double Cij = C[j + i * n];
            for (int k = 0; k < n; ++k){
	            double Aij = A[k + i * n];
	            double Bjk = B[j + k * n];	
	            Cij += Aij * Bjk;
            }
            C[j + i * n] = Cij;
        }
    }
    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double> elapsed_serial = end-start;
    

    // Display the matrix C
    if(display_C){
        cout << "Matrix C:" << endl;
        for (int i = 0; i < n*n; ++i){
            cout << C[i] << " ";
            if((i+1) % n == 0){
                cout << "\n";
            }
        }
    }   
    
    double sum_C = 0.0;
    for(int i = 0; i < n*n; ++i){
        sum_C += C[i];
    }
    cout << "Serial elapsed time = " << elapsed_serial.count()*1000 << endl;
}

void summa(int n, int rank, int size, double *C, double *A, double *B, bool verbose, bool display_A, bool display_B, bool display_C){
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
        for (int k = size-1; k >= 0; --k) {
            // Calculate the starting indices for parition of C
            int row_start = (k/p)*block_size;
            int col_start = (k%p)*block_size;
            
            // Write in partitions
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
        for (int i = 0; i < block_size*block_size; ++i){
            cout << A_ij[i] << " ";
            if((i+1) % block_size == 0){
                cout << "\n";
            }
        }
    }

    // Display the matrix B
    if(display_B){
        cout << "Parition of B on rank = " << rank << endl;
        for (int i = 0; i < block_size*block_size; ++i){
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


    // Gather blocks of C onto root rank
    if(rank == 0) {
        // Write in root rank partition
        for(int i = 0; i < block_size; ++i){
            for(int j = 0; j < block_size; ++j){
                C[i * n + j] = C_ij[i * block_size + j];
            }
        }
        MPI_Status status;
        for(int k = 1; k < size; ++k) {
            // Receive blocks from other ranks        
            MPI_Recv(C_ij, block_size*block_size, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        
            // Calculate the starting indices for parition of C
            int origin_rank = status.MPI_SOURCE;
            int row_start_p = (origin_rank/p)*block_size;
            int col_start_p = (origin_rank%p)*block_size;

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
        cout << "Elapsed time = " << elapsed << endl; 
    }

    // Clear memory
    delete[] A_recv;
    delete[] B_recv;
    delete[] A_ij;
    delete[] B_ij;
    delete[] C_ij; 
}

void cannon(int n, int rank, int size, double *C, double *A, double *B, bool verbose, bool display_A, bool display_B, bool display_C){
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
        for (int k = size-1; k >= 0; --k) {
            // Calculate the starting indices for parition
            int row_start = (k/p)*block_size;
            int col_start = (k%p)*block_size;
            
            // Write in paritions
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
        cout << "Partition of A on rank = " << rank << endl;
        for (int i = 0; i < block_size*block_size; ++i){
            cout << A_ij[i] << " ";
            if((i+1) % block_size == 0){
                cout << "\n";
            }
        }
    }
    // Display the matrix B
    if(display_B){
        cout << "Partition of B on rank = " << rank << endl;
        for (int i = 0; i < block_size*block_size; ++i){
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

    // Map 2D coordinates (x, y) to 1D processor labels
    int x = rank / p;
    int y = rank % p;

    // Initial skew for A
    int origin_A = x * p + (y - x + p) % p; // Source rank for A submatrix
    int dest_A = x * p + (y + x) % p;       // Destination rank for A submatrix
    MPI_Sendrecv_replace(A_ij, block_size*block_size, MPI_DOUBLE, dest_A, 0, origin_A, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Initial skew for B
    int origin_B = ((x - y + p) % p) * p + y; // Source rank for B submatrix
    int dest_B = ((x + y) % p) * p + y;       // Destination rank for B submatrix
    MPI_Sendrecv_replace(B_ij, block_size*block_size, MPI_DOUBLE, dest_B, 0, origin_B, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Progress check
    MPI_Barrier(MPI_COMM_WORLD);
    if(verbose){
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
    if(verbose){
        cout << "Partitions of C computed on rank: " << rank << endl;
    }


    // Gather blocks of C
    if(rank == 0) {
        // Write in root rank partition
        for(int i = 0; i < block_size; ++i){
            for(int j = 0; j < block_size; ++j){
                C[i * n + j] = C_ij[i * block_size + j];
            }
        }
        MPI_Status status;
        for(int k = 1; k < size; ++k) {
            // Receive blocks from other ranks        
            MPI_Recv(C_ij, block_size*block_size, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

            // Calculate the starting indices for parition of C
            int origin_rank = status.MPI_SOURCE;
            int row_start_p = (origin_rank/p)*block_size;
            int col_start_p = (origin_rank%p)*block_size;

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
            cout << "Matrix C:" << endl;
            for (int i = 0; i < n*n; ++i){
                cout << C[i] << " ";
                if((i+1) % n == 0){
                    cout << "\n";
                }
            }
        }   
        double sum_C = 0.0;
        for (int i = 0; i < n*n; ++i){           
            sum_C += C[i];
        }
        cout << "Elapsed time = " << elapsed << endl; 
    }

    // Clear memory
    delete[] A_ij;
    delete[] B_ij;
    delete[] C_ij; 
}

bool check_equal(int n, double *C_1, double *C_2) {
    double tolerance = 1e-9;
    bool equal = true;

    for (int i = 0; i < n*n; ++i) {
        double diff = fabs(C_1[i] - C_2[i]);
        double denom = std::max(1.0, std::max(fabs(C_1[i]), fabs(C_2[i])));
        if (diff/denom > tolerance) {
            equal = false;
            // Print the differing values and their indices
            std::cout << "Mismatch at index " << i << ": C_1[" << i << "] = " << C_1[i] 
                      << ", C_2[" << i << "] = " << C_2[i] << ", diff = " << diff << std::endl;
        }
    }
    return equal;
}