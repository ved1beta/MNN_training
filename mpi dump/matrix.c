#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4 // Set N to the desired matrix size

void fill_matrix(double *m) {
	for (int i = 0; i < N * N; i++) {
		m[i] = (double)(rand() % 10);
	}
}

void print_matrix(double *m){
	for(int i = 0; i < N ; i++) {
		for(int j =0; j < N; j++){
			printf("%lf ", m[i * N + j]);
		}
		printf("\n");
	}
}


int main(int argc , char **argv){
    int rank, size; 
    double *A = NULL , *B = NULL , *C = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(N % size != 0){
       if(rank == 0){
       printf("n should be divisible by no. of cores");
       }
       MPI_Finalize();
       return 1;
    }

    int rows_per_proc = N / size;

    B = (double *)malloc(N * N * sizeof(double));
    double *local_A = (double *)malloc(rows_per_proc * N * sizeof(double));
    double *local_C = (double *)calloc(rows_per_proc * N, sizeof(double));
    
    if(rank == 0){
       A = (double *)malloc(N * N * sizeof(double));
       C = (double *)malloc(N * N * sizeof(double));
       fill_matrix(A);
       fill_matrix(C);
    }
    MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE,local_A, rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
            }
        }
    }
    if (rank == 0) {
        printf("\nResult Matrix C:\n");
        print_matrix(C);
        free(A);
        free(C);
    }

    free(B);
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}


    
