#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <torch/torch.h>
#include <iostream>
#include <vector>

using std::vector;
using std::cout;
using std::endl;
void testTorch() {
    if (!torch::mps::is_available()) {
        cout << "MPS is not available." << endl;
        return;
    }

    torch::TensorOptions option = torch::TensorOptions().device(torch::kMPS);
    torch::Tensor tensor = torch::zeros({2, 3}, option);
    cout << tensor << endl;
}


int main() {
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    NS::Error *error = nullptr;
    testTorch();
    return 0;
}

// int main()
// {
//     std::cout << "Hello World!" << std::endl;
// }