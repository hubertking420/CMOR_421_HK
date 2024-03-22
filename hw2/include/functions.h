#define NUM_THREADS 8
#define BLOCK_SIZE 16
void matmul_blocked_serial(double *C, double *A, double *B, int n);
void matmul_blocked_parallel(double *C, double *A, double *B, int n);
void matmul_blocked_parallel_collapse(double *C, double *A, double *B, int n);
void back_solve_static(double *A, double *b, double *x, int n);
void back_solve_dynamic(double *A, double *b, double *x, int n);
void back_solve_serial(double *A, double *b, double *x, int n);
