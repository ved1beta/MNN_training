#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 
  CHECK_CUDA(x);                                                                                                       
  CHECK_CONTIGUOUS(x)

__global__ void add_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

torch::Tensor add(torch::Tensor a , torch::Tensor b){
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    
    torch::Tensor output = torch::empty(size, c.options())

    int num_threads = 256;
    int num_blocks = (a.size(0) + num_threads - 1) / num_threads;
    add_kernel<<<num_blocks, num_threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), size);

    return c;
}