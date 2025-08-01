#include <iostream>
#include <vector>
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>

// Macro for checking CUDA API calls for errors
#define CUDA_CHECK(call) do { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)

// Macro for checking NCCL API calls for errors
#define NCCL_CHECK(call) do { \
  ncclResult_t res = call; \
  if (res != ncclSuccess) { \
    fprintf(stderr, "NCCL Error: %s at %s:%d\n", ncclGetErrorString(res), __FILE__, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)

int main(int argc, char* argv[]) {
    int rank, world_size;

    // --- MPI Initialization ---
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (world_size != 4) {
        if (rank == 0) {
            fprintf(stderr, "This program requires exactly 4 MPI processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // --- GPU Setup ---
    int num_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus < 1) {
        if (rank == 0) {
            fprintf(stderr, "No GPUs detected on this node.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Assign a GPU to each MPI process. This simple modulo assignment works well for a single node.
    int device_id = rank % num_gpus;
    CUDA_CHECK(cudaSetDevice(device_id));
    printf("MPI Rank %d is using GPU %d\n", rank, device_id);

    // --- Data Allocation and Initialization ---
    const int data_size = 1024 * 1024; // 1M floats
    const size_t bytes = data_size * sizeof(float);

    float* d_buffer; // Pointer for device (GPU) memory
    CUDA_CHECK(cudaMalloc(&d_buffer, bytes));

    // Create host data and initialize it. Each process will have a buffer of 1.0f.
    std::vector<float> h_buffer_in(data_size, 1.0f);
    
    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_buffer, h_buffer_in.data(), bytes, cudaMemcpyHostToDevice));

    // --- NCCL Initialization ---
    ncclComm_t nccl_comm;
    ncclUniqueId nccl_id;

    // Rank 0 creates a unique ID for the NCCL communicator and broadcasts it to all other processes.
    if (rank == 0) {
        ncclGetUniqueId(&nccl_id);
    }
    MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Each process initializes its NCCL communicator.
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank));

    // --- NCCL Collective Operation ---
    // Synchronize all processes before starting the collective operation.
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Starting ncclAllReduce...\n");
    }

    // Perform an AllReduce operation.
    // This will sum the elements from the d_buffer on all 4 GPUs
    // and place the result back into the d_buffer on each GPU.
    // Since each element was 1.0f, the result on every GPU should be 4.0f.
    NCCL_CHECK(ncclAllReduce(d_buffer, d_buffer, data_size, ncclFloat, ncclSum, nccl_comm, 0));

    // --- Verification (on Rank 0) ---
    if (rank == 0) {
        printf("AllReduce complete. Verifying result on Rank 0...\n");
        std::vector<float> h_buffer_out(data_size);
        
        // Copy the result from the device back to the host
        CUDA_CHECK(cudaMemcpy(h_buffer_out.data(), d_buffer, bytes, cudaMemcpyDeviceToHost));

        // Check the result. Every element should be 4.0.
        bool success = true;
        for (int i = 0; i < data_size; ++i) {
            if (h_buffer_out[i] != 4.0f) {
                fprintf(stderr, "Verification FAILED at index %d! Expected: 4.0, Got: %f\n", i, h_buffer_out[i]);
                success = false;
                break;
            }
        }

        if (success) {
            printf("Verification PASSED! All elements are 4.0.\n");
        }
    }

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_buffer));
    NCCL_CHECK(ncclCommDestroy(nccl_comm));
    MPI_Finalize();

    return 0;
}
