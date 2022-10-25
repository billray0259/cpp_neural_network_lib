# include "../tensor.h"



int main() {
    Tensor data = Tensor::FromCsv("csv_mats/iris_data.csv"); // (N, 4)
    Tensor labels = Tensor::FromCsv("csv_mats/iris_labels.csv"); // (N, 1)

    Tensor weights = Tensor::Randn({4, 3}); // (4, 3)
    Tensor bias = Tensor::Randn({1, 3}); // (1, 3)

    int epochs = 10;
    int batch_size = 32;

    Tensor output = data.MatMul(weights).AddTensor(bias); // (N, 3)

    // TODO training loop
    
}