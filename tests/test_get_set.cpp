#include "../tensor.h"


int main() {

    // 0, 0
    // 1, 1
    // 1, 2
    // 2, 1


    Tensor tensor = Tensor::FromCsv("mat4.csv"); // 3x4

    tensor.Set({0, 0}, 1);
    std::cout << "{0, 0}\n" << tensor.ToString() << tensor.Get({0, 0}) << std::endl << std::endl;

    tensor = Tensor::FromCsv("mat4.csv");
    tensor.Set({1, 1}, 1);
    std::cout << "{1, 1}\n" << tensor.ToString() << tensor.Get({1, 1}) << std::endl << std::endl;

    tensor = Tensor::FromCsv("mat4.csv");
    tensor.Set({1, 2}, 1);
    std::cout << "{1, 2}\n" << tensor.ToString() << tensor.Get({1, 2}) << std::endl << std::endl;

    tensor = Tensor::FromCsv("mat4.csv");
    tensor.Set({2, 1}, 1);
    std::cout << "{2, 1}\n" << tensor.ToString() << tensor.Get({2, 1}) << std::endl << std::endl;

    
    return 0;
}

