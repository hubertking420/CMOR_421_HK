void matmul_naive(int n, double *C, double *A, double *B, bool display_C);
void summa(int n, int rank, int size, double *C, double *A, double *B, bool verbose, bool display_A, bool display_B, bool display_C);
void cannon(int n, int rank, int size, double *C, double *A, double *B, bool verbose, bool display_A, bool display_B, bool display_C);
bool check_equal(int n, double *C_1, double *C_2);
