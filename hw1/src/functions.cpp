#include <iostream>
#include <cmath>
#include "functions.h"



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
	double denom = std::max(1.0, std::max(fabs(x), fabs(y)));
	return (diff/denom) <= tolerance;
}