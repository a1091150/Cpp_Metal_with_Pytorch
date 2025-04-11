
import torch
import torch.utils.cpp_extension

compiled_lib = torch.utils.cpp_extension.load(
    name="AddArray",
    sources=["addArray.cpp"],
    extra_cflags=['-std=c++17']
)

def testAdd(foo: torch.Tensor) -> torch.Tensor:
    compiled_lib.add(foo)
    pass

if __name__ == "__main__":
    foo = torch.ones([1, 2])
    testAdd(foo=foo)
    print(foo)
    pass