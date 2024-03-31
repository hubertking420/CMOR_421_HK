#include "functions.h"
// #include "mpi.h"
// computes C = C + A*B
void matmul_naive(int n, double* C, double* A, double* B){
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
}

void matmul_mpi(int n, const int rank, const int size, double* C, double *A, double* B){

}
