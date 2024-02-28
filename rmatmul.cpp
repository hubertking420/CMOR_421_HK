#include <iostream>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

#define BLOCK_SIZE 8

// Forward declaration of the RMM function
void recursive_matrix_multiply(int n, double* C, double* A, double* B, int rowC, int colC, int rowA, int colA, int rowB, int colB, int originalSize);

// Microkernel for base case matrix multiplication
void microkernel(int n, double* C, double* A, double* B, int rowC, int colC, int rowA, int colA, int rowB, int colB, int originalSize) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[(rowA + i) * originalSize + colA + k] * B[(rowB + k) * originalSize + colB + j];
            }
            C[(rowC + i) * originalSize + colC + j] += sum;
        }
    }
}

// Recursive matrix multiplication function
void recursive_matrix_multiply(int n, double* C, double* A, double* B, int rowC, int colC, int rowA, int colA, int rowB, int colB, int originalSize) {
    if (n <= BLOCK_SIZE) {
        microkernel(n, C, A, B, rowC, colC, rowA, colA, rowB, colB, originalSize);
        return;
    }
    int newSize = n / 2;
    // Recurse for each quadrant of the result matrix
	// Top Left Quadrant
    recursive_matrix_multiply(newSize, C, A, B, rowC, colC, rowA, colA, rowB, colB, originalSize);
	recursive_matrix_multiply(newSize, C, A, B, rowC, colC, rowA, colA + newSize, rowB + newSize, colB, originalSize);

	// Top Right Quadrant
    recursive_matrix_multiply(newSize, C, A, B, rowC, colC + newSize, rowA, colA, rowB, colB + newSize, originalSize);
   	recursive_matrix_multiply(newSize, C, A, B, rowC, colC + newSize, rowA, colA + newSize, rowB + newSize, colB + newSize, originalSize);

	
 	// Bottom Left Quadrant
 	recursive_matrix_multiply(newSize, C, A, B, rowC + newSize, colC, rowA + newSize, colA, rowB, colB, originalSize);
	recursive_matrix_multiply(newSize, C, A, B, rowC + newSize, colC, rowA + newSize, colA + newSize, rowB + newSize, colB, originalSize);

	// Bottom Right Quadrant
	recursive_matrix_multiply(newSize, C, A, B, rowC + newSize, colC + newSize, rowA + newSize, colA, rowB, colB + newSize, originalSize);
	recursive_matrix_multiply(newSize, C, A, B, rowC + newSize, colC + newSize, rowA + newSize, colA + newSize, rowB + newSize, colB + newSize, originalSize);

	  
}

bool almost_equal(double x, double y, double tolerance){
	double diff = fabs(x - y);
	double denom = max(1.0, max(fabs(x), fabs(y)));
	return (diff/denom) <= tolerance;
}

int main(int argc, char * argv[]){
    int n = atoi(argv[1]);
    cout << "Matrix size n = " << n << ", block size = " << BLOCK_SIZE << endl;
    
    double * A = new double[n * n];
    double * B = new double[n * n];
    double * C = new double[n * n];

    // make A, B = I
    for (int i = 0; i < n; ++i){
        A[i + i * n] = 1.0;
        B[i + i * n] = 1.0;
    }
    for (int i = 0; i < n * n; ++i){
        C[i] = 0.0;
    }

    int num_trials = 5;

    // Measure performance
    high_resolution_clock::time_point start = high_resolution_clock::now();
    for (int i = 0; i < num_trials; ++i){
		recursive_matrix_multiply(n, C, A, B, 0, 0, 0, 0, 0, 0, n);
	}
    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double> elapsed_recursive = (end - start) / num_trials;

    double sum_C = 0.0;
    for (int i = 0; i < n * n; ++i){
        sum_C += C[i];
    }
    cout << "Recursive sum_C = " << sum_C << endl; 
    cout << "Recursive elapsed time (ms) = " << elapsed_recursive.count() * 1000 << endl;
	
	// Reset C and perform matmul once
 	for (int i = 0; i < n * n; ++i){
    	C[i] = 0.0;
	}
	recursive_matrix_multiply(n, C, A, B, 0, 0, 0, 0, 0, 0, n);
	// Check C elementwise
	double tolerance = numeric_limits<double>::epsilon();
	bool is_identity = true;
	for(int i = 0; i < n; ++i){
		for(int j = 0; j < n; ++j){
			double expected = (i == j) ? 1.0 : 0.0;
			if(i == j){
				if(!almost_equal(C[i*n + j], expected, tolerance)) {
					is_identity = false;
					break;
				}
			} 
			if(!is_identity){
				break;
			}	
		}
	}
	if(is_identity){
		cout << "Test passed. C = I to machine precision." << endl;
	} else{
		cout << "Test failed. C != I to machine precision." << endl;
	}	

    delete[] A;
    delete[] B;
    delete[] C;  
  
    return 0;
}

