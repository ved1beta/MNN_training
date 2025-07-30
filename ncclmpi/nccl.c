#include <stdio.h>
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdlib> // FIX 1: Include the cstdlib header for exit(), malloc(), free()

#define CUDA_CHECK(cmd) do { \
    cudaError_t e = cmd; \
    if( e != cudaSuccess ) { \
        printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define NCCL_CHECK(cmd) do { \
    ncclResult_t r = cmd; \
    if (r != ncclSuccess) { \
        printf("Failed: NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


int main(int argc, char* argv[]) {
    int rank, size;
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. Each process selects the same GPU and creates its own context
    CUDA_CHECK(cudaSetDevice(0));

    ncclComm_t comm;
    cudaStream_t stream;
    ncclUniqueId id;

    // 2. Get a unique ID for the NCCL communicator
    if (rank == 0) {
        // FIX 2: Corrected function name from ncclGetUniqueID to ncclGetUniqueId
        NCCL_CHECK(ncclGetUniqueId(&id));
    }

    // 3. Broadcast the unique ID to all processes
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // 4. Initialize the NCCL communicator for each process
    NCCL_CHECK(ncclCommInitRank(&comm, size, id, rank));

    // 5. Create a CUDA stream for asynchronous execution
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Allocate memory and prepare data on the GPU
    const int n_elements = 16;
    float *send_buffer, *recv_buffer;
    CUDA_CHECK(cudaMalloc(&send_buffer, n_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recv_buffer, n_elements * sizeof(float)));

    // Initialize data: each process fills its buffer with its rank
    float initial_value = (float)rank;
    float* temp_host_buffer = (float*)malloc(n_elements * sizeof(float));
    for (int i = 0; i < n_elements; ++i) {
        temp_host_buffer[i] = initial_value;
    }
    CUDA_CHECK(cudaMemcpy(send_buffer, temp_host_buffer, n_elements * sizeof(float), cudaMemcpyHostToDevice));
    free(temp_host_buffer);


    // Print data before All-reduce
    if (rank == 0) printf("--- Before All-reduce ---\n");
    MPI_Barrier(MPI_COMM_WORLD);
    float first_element;
    CUDA_CHECK(cudaMemcpy(&first_element, send_buffer, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Rank %d has data with value: %.1f\n", rank, first_element);


    // 6. Perform All-reduce (summation)
    NCCL_CHECK(ncclAllReduce(send_buffer, recv_buffer, n_elements, ncclFloat, ncclSum, comm, stream));
    
    // Synchronize the stream to ensure the operation is complete
    CUDA_CHECK(cudaStreamSynchronize(stream));


    // Print data after All-reduce
    if (rank == 0) printf("\n--- After All-reduce (Sum) ---\n");
    MPI_Barrier(MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(&first_element, recv_buffer, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Rank %d has data with value: %.1f\n", rank, first_element);


    // Cleanup
    CUDA_CHECK(cudaFree(send_buffer));
    CUDA_CHECK(cudaFree(recv_buffer));
    NCCL_CHECK(ncclCommDestroy(comm));
    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_Finalize();

    return 0;
}