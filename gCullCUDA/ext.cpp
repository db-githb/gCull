#include <torch/extension.h>
#include "cull_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cull_gaussians", &CullGaussiansCUDA);
}