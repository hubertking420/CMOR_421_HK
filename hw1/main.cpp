#include <iostream>
#include <chrono>
#include "include/functions.h"

using namespace std;
using namespace std::chrono;


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

