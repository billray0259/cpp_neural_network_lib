#include "../tensor.h"


int main() {
    Tensor tensor = Tensor::Arange(0, 12);
    tensor.Reshape({3, 4});

    std::cout << "Tensor\n" << tensor.ToString() << std::endl << std::endl;

    vector<int> dims = {0, 1};
    std::cout << "dims={0, 1}\n" << Tensor(tensor.PermuteDims(dims)).ToString() << std::endl << std::endl;

    std::cout << "dims={1, 0}\n" << Tensor(tensor.PermuteDims(dims)).ToString() << std::endl << std::endl;

    return 0;
}

