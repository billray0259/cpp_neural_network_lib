#include "../tensor.h"
// #include <vector>
// using std::vector;

int main() {
    Tensor tensor = Tensor::FromCsv("mat0.csv"); // 3x4

    std::cout << ToLinearIndex(tensor.shape, {0, 0}) << std::endl;

    std::cout << ToLinearIndex(tensor.shape, {1, 1}) << std::endl;

    std::cout << ToLinearIndex(tensor.shape, {1, 2}) << std::endl;

    std::cout << ToLinearIndex(tensor.shape, {2, 1}) << std::endl;

    // test FromLinearIndex
    std::cout << FromLinearIndex(tensor.shape, 0)[0] << FromLinearIndex(tensor.shape, 0)[1] << std::endl;

    std::cout << FromLinearIndex(tensor.shape, 4)[0] << FromLinearIndex(tensor.shape, 4)[1] << std::endl;

    std::cout << FromLinearIndex(tensor.shape, 5)[0] << FromLinearIndex(tensor.shape, 5)[1] << std::endl;

    std::cout << FromLinearIndex(tensor.shape, 7)[0] << FromLinearIndex(tensor.shape, 7)[1] << std::endl;
    
    return 0;
}

