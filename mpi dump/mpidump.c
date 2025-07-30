#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// --- REDUCED FOR CLEARER OUTPUT ---
#define N 4 // Matrix size (N x N)

// Fills a matrix with values
void fill_matrix(double *m) {
    for (int i = 0; i < N * N; i++) {
        m[i] = i + 1; // Use predictable values instead of random
    }
}

// Prints a matrix or a chunk of a matrix
void print_chunk(double *m, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", m[i * cols + j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    int rank, size;
    double *A = NULL, *B = NULL, *C = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "Matrix size N (%d) must be divisible by the number of processes (%d)\n", N, size);
        }
        MPI_Finalize();
        return 1;
    }

    int rows_per_proc = N / size;
    B = (double *)malloc(N * N * sizeof(double));
    double *local_A = (double *)malloc(rows_per_proc * N * sizeof(double));
    double *local_C = (double *)calloc(rows_per_proc * N, sizeof(double));

    if (rank == 0) {
        A = (double *)malloc(N * N * sizeof(double));
        C = (double *)malloc(N * N * sizeof(double));
        fill_matrix(A);
        fill_matrix(B);
        printf("--- Initial Data (on Rank 0) ---\n");
        printf("Matrix A:\n"); print_chunk(A, N, N);
        printf("\nMatrix B:\n"); print_chunk(B, N, N);
        printf("----------------------------------\n\n");
    }

    MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE,
                local_A, rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // --- VISUALIZATION 1: What each process received ---
    for (int i = 0; i < size; i++) {
        if (rank == i) {
            printf("--- Data on Rank %d after Scatter/Bcast ---\n", rank);
            printf("My local_A chunk:\n");
            print_chunk(local_A, rows_per_proc, N);
            // Optional: print B to confirm broadcast
            // printf("\nMy B matrix:\n"); print_chunk(B, N, N);
            printf("------------------------------------------\n\n");
        }
        MPI_Barrier(MPI_COMM_WORLD); // Ensures ordered printing
    }


    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
            }
        }
    }
    
    // --- VISUALIZATION 2: What each process calculated ---
    for (int i = 0; i < size; i++) {
        if (rank == i) {
            printf("--- Result on Rank %d before Gather ---\n", rank);
            printf("My local_C chunk:\n");
            print_chunk(local_C, rows_per_proc, N);
            printf("---------------------------------------\n\n");
        }
        MPI_Barrier(MPI_COMM_WORLD); // Ensures ordered printing
    }


    MPI_Gather(local_C, rows_per_proc * N, MPI_DOUBLE,
               C, rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("--- Final Result Matrix C (on Rank 0) ---\n");
        print_chunk(C, N, N);
        printf("-------------------------------------------\n");
        free(A);
        free(C);
    }

    free(B);
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}