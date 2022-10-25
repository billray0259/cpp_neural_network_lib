# include "../tensor.h"

int main() {
    Tensor tensor = Tensor::Arange(0, 12);
    tensor.Reshape({1, 3 , 4});

    Tensor product = tensor.MatMul(tensor.PermuteDims({0, 2, 1}));
    product.Reshape({3, 3});

    std::cout << product.ToString() << std::endl << std::endl;   
}


