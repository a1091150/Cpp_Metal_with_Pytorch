#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

// #include <torch/extension.h>
#include <iostream>
#include <vector>
using std::vector;

int main() {

    // using nns = NS::String;
    // vector<nns *> vs = {str1, str2, str3};
    const void* strings[] = {CFSTR("Hello"), CFSTR("Hello2"), CFSTR("Hello3")};
    // NS::Array* pArray = (NS::Array*) CFArrayCreate(kCFAllocatorDefault, (const void **));
    CFArrayRef array = CFArrayCreate(kCFAllocatorDefault, strings, 3, &kCFTypeArrayCallBacks);
    CFShow(array);
    CFRelease(array);
    return 0;
}

// int main()
// {
//     std::cout << "Hello World!" << std::endl;
// }