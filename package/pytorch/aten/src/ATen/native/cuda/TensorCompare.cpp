#include <ATen/Functions.h>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorCompare.h>

namespace at::native {

namespace {

// Composite op implementation for simplicity. This materializes the cross product of elements and test elements,
// so it is not very memory efficient, but it is fast on CUDA.
void isin_default_kernel_gpu(
    const Tensor& elements, const Tensor& test_elements, bool invert, const Tensor& out) {
  std::vector<int64_t> bc_shape(elements.dim(), 1);
  bc_shape.push_back(-1);
  out.copy_(invert ? elements.unsqueeze(-1).ne(test_elements.view(bc_shape)).all(-1)
            : elements.unsqueeze(-1).eq(test_elements.view(bc_shape)).any(-1));
}

} // anonymous namespace

Tensor& protectclamp_self_out_cuda(const Tensor& self, const Tensor& min, const Tensor& max, Tensor& out) {
    //TORCH_CHECK(self.is_cuda(), "protectclamp only supports CUDA tensors");
    //Tensor result = at::empty_like(self);
    auto iter = TensorIteratorConfig()
        .add_output(out)
        .add_const_input(self)
        .add_const_input(min)
        .add_const_input(max)
        .build();
    protectclamp_stub(iter.device_type(), iter);
    return out;
}

Tensor protectclamp_cuda(const Tensor& self, const Tensor& min, const Tensor& max) {
  Tensor ret = at::empty_like(self);
  at::native::protectclamp_self_out_cuda(self, min, max, ret);
  return ret;
}


REGISTER_CUDA_DISPATCH(isin_default_stub, &isin_default_kernel_gpu);

} // namespace at::native
