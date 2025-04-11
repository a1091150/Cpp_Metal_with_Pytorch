
#include "addArray.h"
#include <torch/extension.h>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

void testMetalCpp() {
    const void* strings[] = {CFSTR("Hello"), CFSTR("Hello2"), CFSTR("Hello3")};
    CFArrayRef array = CFArrayCreate(kCFAllocatorDefault, strings, 3, &kCFTypeArrayCallBacks);
    CFShow(array);
    CFRelease(array);
}

void addArray(torch::Tensor& input) {
    input.add_(1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &addArray);
    m.def("testMetalCpp", &testMetalCpp);
}