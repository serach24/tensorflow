
#include "tensorflow/core/kernels/cast_op_impl.h"

#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// todo(chenhao) neglect cpu for now
CastFunctorType GetCpuCastFromCus(DataType dst_dtype) {
  CURRY_TYPES3(CAST_CASE, CPUDevice, cus_type);
  return nullptr;
}

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
CastFunctorType GetGpuCastFromCus(DataType dst_dtype) {
  if (dst_dtype == DT_FLOAT) {
    return [](OpKernelContext* ctx, const Tensor& inp, Tensor* out,
              bool truncate) {
      functor::CastFunctor<GPUDevice, float, cus_type> func;
      func(ctx->eigen_device<GPUDevice>(), out->flat<float>(),
           inp.flat<cus_type>(), truncate);
    };
  }
  return nullptr;
}
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
