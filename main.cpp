#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <torch/torch.h>
// #include <torch/extension.h>
#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;

void rope_2d_cpu(torch::Tensor tokens, const torch::Tensor positions, const float base, const float fwd)
{
    const int B = tokens.size(0);
    const int N = tokens.size(1);
    const int H = tokens.size(2);
    const int D = tokens.size(3) / 4;

    auto tok = tokens.accessor<float, 4>();
    auto pos = positions.accessor<int64_t, 3>();

    for (int b = 0; b < B; b++)
    {
        for (int x = 0; x < 2; x++)
        { // y and then x (2d)
            for (int n = 0; n < N; n++)
            {

                // grab the token position
                const int p = pos[b][n][x];

                for (int h = 0; h < H; h++)
                {
                    for (int d = 0; d < D; d++)
                    {
                        // grab the two values
                        float u = tok[b][n][h][d + 0 + x * 2 * D];
                        float v = tok[b][n][h][d + D + x * 2 * D];

                        // grab the cos,sin
                        const float inv_freq = fwd * p / powf(base, d / float(D));
                        float c = cosf(inv_freq);
                        float s = sinf(inv_freq);

                        // write the result
                        tok[b][n][h][d + 0 + x * 2 * D] = u * c - v * s;
                        tok[b][n][h][d + D + x * 2 * D] = v * c + u * s;
                    }
                }
            }
        }
    }
}

void testTorchCpu() {
    torch::TensorOptions option = torch::TensorOptions();

    // B, N, H, D
    torch::Tensor tokens = torch::rand({1, 4, 8, 16}, option);
    // B, N, 2
    torch::Tensor positions = torch::ones({1, 4, 2}, option.dtype(torch::kInt64));
    
    cout << "Before" << endl << tokens << endl;
    rope_2d_cpu(tokens, positions, 1, 1);
    cout << "After" << endl << tokens << endl;
}

void testTorch()
{
    if (!torch::mps::is_available())
    {
        cout << "MPS is not available." << endl;
        return;
    }

    torch::TensorOptions option = torch::TensorOptions().device(torch::kMPS);
    torch::Tensor tensor = torch::zeros({2, 3}, option);
    cout << tensor << endl;
}

int main()
{
    testTorchCpu();
    return 0;
}

// int main()
// {
//     std::cout << "Hello World!" << std::endl;
// }