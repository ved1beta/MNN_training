#include <stdio.h>
#include <nvshmem.h>
#include <nvshmemx.h>

// A simple macro for CUDA error checking
#define CUDA_CHECK(cmd) do { \
    cudaError_t e = cmd; \
    if( e != cudaSuccess ) { \
        printf("Failed: Cuda error %s:%d '%s'\n", __FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main(int argc, char** argv) {
    // Initialize NVSHMEM. It's often bootstrapped with MPI.
    nvshmem_init();

    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    if (n_pes != 2) {
        if (my_pe == 0) {
            fprintf(stderr, "This program requires exactly 2 Processing Elements (PEs).\n");
        }
        nvshmem_finalize();
        return 1;
    }

    // Allocate a single float on the symmetric heap of each PE
    float* data = (float*)nvshmem_malloc(sizeof(float));
    if (!data) {
        fprintf(stderr, "PE %d: nvshmem_malloc failed\n", my_pe);
        nvshmem_finalize();
        return 1;
    }

    // Use a CUDA stream for operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // --- Part 1: PE 0 puts data into PE 1's memory ---
    if (my_pe == 0) {
        // Initialize the source data on the host
        float source_val = 42.0f;
        printf("PE 0: Putting value %.1f into PE 1's memory.\n", source_val);

        // Copy source value to my GPU symmetric memory
        CUDA_CHECK(cudaMemcpyAsync(data, &source_val, sizeof(float), cudaMemcpyHostToDevice, stream));

        // Perform the put operation: put my 'data' into the 'data' buffer of PE 1
        nvshmemx_float_put_on_stream(data, data, 1, 1, stream);
    }

    // Synchronize all PEs on the stream to ensure the 'put' is complete
    nvshmemx_barrier_all_on_stream(stream);

    // PE 1 verifies the data it received
    if (my_pe == 1) {
        float received_val;
        CUDA_CHECK(cudaMemcpyAsync(&received_val, data, sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("PE 1: Verified received value is %.1f.\n", received_val);
    }
    
    // --- Part 2: PE 0 gets data from PE 1's memory ---
    nvshmem_barrier_all(); // Simple barrier for clean separation

    if (my_pe == 0) {
        float* result_val_gpu = (float*)nvshmem_malloc(sizeof(float));
        printf("\nPE 0: Getting value from PE 1's memory.\n");
        
        // Perform the get operation: get PE 1's 'data' and place it in my 'result_val_gpu'
        nvshmemx_float_get_on_stream(result_val_gpu, data, 1, 1, stream);
        
        float result_val_host;
        CUDA_CHECK(cudaMemcpyAsync(&result_val_host, result_val_gpu, sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("PE 0: Verified retrieved value is %.1f.\n", result_val_host);
        
        nvshmem_free(result_val_gpu);
    }

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    nvshmem_free(data);
    nvshmem_finalize();

    return 0;
}