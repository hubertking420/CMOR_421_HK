#include <iostream>

using namespace std;
#define BLOCK_SIZE 16
int main(int argc, char * argv[]){
    int n = atoi(argv[1]);
    int trials = 5;
    
    // Part 1
    cout << "Matrix size n = " << n << ", block size = " << BLOCK_SIZE << endl;

    double * A = new double[n * n];
    double * B = new double[n * n];
    double * C = new double[n * n];

    // Initialize multiplication I x I = C
    for (int i = 0; i < n; ++i){
        A[i + i * n] = 1.0;
        B[i + i * n] = 1.0;
    }
    for (int i = 0; i < n * n; ++i){
        C[i] = 0.0;
    }

    for (int i = 0; i < n; i += BLOCK_SIZE){
        for (int j = 0; j < n; j += BLOCK_SIZE){
            for (int k = 0; k < n; k += BLOCK_SIZE){

                // small matmul
                for (int ii = i; ii < i + BLOCK_SIZE; ii++){
                    for (int jj = j; jj < j + BLOCK_SIZE; jj++){
                        double Cij = C[jj + ii * n];
                        for (int kk = k; kk < k + BLOCK_SIZE; kk++){
                            Cij += A[kk + ii * n] * B[jj + kk * n]; // Aik * Bkj
                        }
                        C[jj + ii * n] = Cij;
                    }
                }
            }
        }
    }

    double sum_C_serial = 0.0;
    for (int i = 0; i < n * n; ++i){
        sum_C_serial += C[i];
    }

    // reset C
    for (int i = 0; i < n * n; ++i){
        C[i] = 0.0;
    }

    // 

    delete[] A;
    delete[] B;
    delete[] C;  
}