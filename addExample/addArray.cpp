
#include "addArray.h"
#include <torch/extension.h>
// #include <Foundation/Foundation.hpp>
// #include <Metal/Metal.hpp>
// #include <QuartzCore/QuartzCore.hpp>

void addArray(torch::Tensor& input) {
    input.add_(1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &addArray);
}