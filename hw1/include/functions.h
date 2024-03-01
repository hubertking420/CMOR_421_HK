#pragma once
#define BLOCK_SIZE 8

void recursive_matrix_multiply(int n, double* C, double* A, double* B, int rowC, int colC, int rowA, int colA, int rowB, int colB, int originalSize);
void microkernel(int n, double* C, double* A, double* B, int rowC, int colC, int rowA, int colA, int rowB, int colB, int originalSize);
bool almost_equal(double x, double y, double tolerance);
