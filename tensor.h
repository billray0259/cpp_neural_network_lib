#include <vector>
#include <random>
#include <cmath>

#include "tensor_util.h"

using std::vector;
using std::string;

class Tensor {
    public:
        vector<float> data;
        bool requires_grad = false;
        vector<float> grad;
        vector<int> shape;
        int size;

        Tensor() {
            this->data = {};
            this->shape = {0};
        }

        Tensor(vector<float> data, vector<int> shape) {
            // Set the shape
            this->shape = shape;
            // Calculate the size
            this->size = 1;
            for (int i = 0; i < shape.size(); i++) {
                this->size *= shape[i];
            }

            if (this->size != data.size()) {
                throw std::invalid_argument("data and shape do not match");
            }

            // Set the data
            this->data = data;
            // Create a vector to hold the gradients
            this->grad = vector<float>(this->size);
        }


        static Tensor Arange(int start, int end, int step=1) {
            vector<float> data;
            for (int i = start; i < end; i += step) {
                data.push_back(i);
            }
            return Tensor(data, {int(data.size())});
        }


        static Tensor Randn(vector<int> shape) {
            // Create a vector to hold the data
            vector<float> data = vector<float>(ShapeToSize(shape));
            // Loop through the shape
            std::random_device rd;
            std::mt19937 gen(rd());
            for (int i = 0; i < shape[0]; i++) {
                for (int j = 0; j < shape[1]; j++) {
                    // Add a random number to the data
                    std::normal_distribution<> d(0, 1);
                    data[i] += d(gen);
                }
            }
            // Create and return the tensor
            return Tensor(data, shape);
        }


        static Tensor Full(vector<int> shape, float value) {
            // Create a vector to hold the data
            vector<float> data = vector<float>(ShapeToSize(shape));
            for (int i = 0; i < data.size(); i++) {
                data[i] = value;
            }
            // Create and return the tensor
            return Tensor(data, shape);
        }


        static Tensor FromCsv(string file_name) {
            string file_content = ReadFile(file_name);
            vector<vector<float>> matrix = ReadCsvMatrix(file_content);
            vector<float> data = vector<float>();

            for (int i = 0; i < matrix.size(); i++) {
                for (int j = 0; j < matrix[i].size(); j++) {
                    data.push_back(matrix[i][j]);
                }
            }

            return Tensor(data, {int(matrix.size()), int(matrix[0].size())});
        }

        Tensor PermuteDims(vector<int> new_order) {
            // Calculate the new shape
            vector<int> new_shape = vector<int>(new_order.size());
            for (int i = 0; i < new_order.size(); i++) {
                new_shape[i] = this->shape[new_order[i]];
            }

            // Create a vector to hold the data
            vector<float> new_data = vector<float>(this->size);

            // Loop through the data
            for (int i = 0; i < this->size; i++) {
                // Get the index
                vector<int> index = FromLinearIndex(this->shape, i);
                // Permute the index
                vector<int> new_index = vector<int>(new_order.size());
                for (int j = 0; j < new_order.size(); j++) {
                    new_index[j] = index[new_order[j]];
                }
                // Set the new data
                new_data[ToLinearIndex(new_shape, new_index)] = this->data[i];
            }

            // Create and return the tensor
            return Tensor(new_data, new_shape);
        }


        vector<vector<float>> BuildReductionMatrix(int dim) {
            // permute dims to {0, 1, ... , dim}
            vector<int> new_order = vector<int>(this->shape.size());
            for (int i = 0; i < this->shape.size()-1; i++) {
                if (i < dim) {
                    new_order[i] = i;
                } else {
                    new_order[i] = i + 1;
                }
            }
            new_order[this->shape.size()-1] = dim;

            Tensor permuted_tensor = this->PermuteDims(new_order);

            permuted_tensor.Reshape({this->size / this->shape[dim], this->shape[dim]});

            // Create the reduction matrix
            vector<vector<float>> reduction_matrix = vector<vector<float>>(this->size / this->shape[dim], vector<float>(this->shape[dim]));
            // Fill in the reduction matrix
            for (int r = 0; r < this->size / this->shape[dim]; r++) {
                for (int c = 0; c < this->shape[dim]; c++) {
                    reduction_matrix[r][c] = permuted_tensor.Get({r, c});
                }
            }
            return reduction_matrix;
        }


        static Tensor FromReductionMatrix(vector<vector<float>> reduction_matrix, int dim, vector<int> out_shape) {
            // Build a tensor such that the rows of the reduction matrix are in the dim dimension
            // verify num_cols == out_shape[dim]
            if (reduction_matrix[0].size() != out_shape[dim]) {
                throw std::invalid_argument("reduction_matrix and out_shape do not match" + std::to_string(reduction_matrix[0].size()) + " " + std::to_string(out_shape[dim]));
            }

            // Create a vector to hold the data
            vector<float> new_data = vector<float>(ShapeToSize(out_shape));
            // copy reduction matrix into new_data
            for (int r = 0; r < reduction_matrix.size(); r++) {
                for (int c = 0; c < reduction_matrix[0].size(); c++) {
                    new_data[r * reduction_matrix[0].size() + c] = reduction_matrix[r][c];
                }
            }

            Tensor new_tensor = Tensor(new_data, {int(reduction_matrix.size()), out_shape[dim]});
            // if (dim == 0) {
            //     new_tensor = new_tensor.PermuteDims({1, 0});
            // }
            // return new_tensor;

            // reshape to {out_shape[0], out_shape[1], ..., out_shape[size], out_shape[dim]}
            vector<int> new_order = vector<int>(out_shape.size());
            for (int i = 0; i < out_shape.size()-1; i++) {
                if (i < dim) {
                    new_order[i] = i;
                } else {
                    new_order[i] = i + 1;
                }
            }
            new_order[out_shape.size()-1] = dim;

            vector<int> new_shape = vector<int>(out_shape.size());
            for (int i = 0; i < out_shape.size(); i++) {
                new_shape[i] = out_shape[new_order[i]];
            }

            new_tensor.Reshape(new_shape);

            // Tensor new_tensor = Tensor(new_data, new_shape);

            // permute dims to {0, 1, ..., dim, ..., size}
            vector<int> old_order = vector<int>(out_shape.size());
            for (int i = 0; i < out_shape.size(); i++) {
                if (i < dim) {
                    old_order[i] = i;
                } else if (i > dim) {
                    old_order[i] = i - 1;
                } else {
                    old_order[i] = out_shape.size() - 1;
                }
            }

            return new_tensor.PermuteDims(old_order);
        }


        Tensor Reduce(float (*func)(vector<float>), int dim) {
            vector<vector<float>> reduction_matrix = this->BuildReductionMatrix(dim);

            vector<vector<float>> reduced_matrix = vector<vector<float>>(reduction_matrix.size(), vector<float>(1));

            // fill in the reduced matrix
            for (int r = 0; r < reduction_matrix.size(); r++) {
                reduced_matrix[r][0] = func(reduction_matrix[r]);
            }

            // Calculate the shape of the output
            vector<int> output_shape;
            for (int i = 0; i < this->shape.size(); i++) {
                if (i != dim) {
                    output_shape.push_back(this->shape[i]);
                } else {
                    output_shape.push_back(1);
                }
            }

            return Tensor::FromReductionMatrix(reduced_matrix, dim, output_shape);
        }


        Tensor Apply(vector<float> (*func)(vector<float>), int dim) {
            vector<vector<float>> reduction_matrix = this->BuildReductionMatrix(dim);

            // vector<vector<float>> applied_matrix = vector<vector<float>>(reduction_matrix.size(), vector<float>(this->shape[dim]));

            // fill in the applied matrix
            for (int r = 0; r < reduction_matrix.size(); r++) {
                vector<float> result = func(reduction_matrix[r]);
                for (int c = 0; c < this->shape[dim]; c++) {
                    reduction_matrix[r][c] = result[c];
                }
            }
            return Tensor::FromReductionMatrix(reduction_matrix, dim, this->shape);
        }



        Tensor ReduceSum(int dim) {
            return this->Reduce([](vector<float> v) {
                float sum = 0;
                for (int i = 0; i < v.size(); i++) {
                    sum += v[i];
                }
                return sum;
            }, dim);
        }


        Tensor Max(int dim) {
            return this->Reduce([](vector<float> v) {
                float max = v[0];
                for (int i = 1; i < v.size(); i++) {
                    if (v[i] > max) {
                        max = v[i];
                    }
                }
                return max;
            }, dim);
        }

        Tensor ArgMax(int dim) {
            return this->Reduce([](vector<float> v) {
                float max = v[0];
                int max_index = 0;
                for (int i = 1; i < v.size(); i++) {
                    if (v[i] > max) {
                        max = v[i];
                        max_index = i;
                    }
                }
                return float(max_index);
            }, dim);
        }

    
        Tensor Broadcast(float (*func)(float, float), Tensor other) {
            // // get the shapes but as a copy
            // vector<int> this_shape = vector<int>(this->shape);
            // vector<int> other_shape = vector<int>(other.shape);

            // for (int i = 0; i < this_shape.size(); i++) {
            //     this_shape[i] = this->shape[i];
            // }
            // for (int i = 0; i < other_shape.size(); i++) {
            //     other_shape[i] = other.shape[i];
            // }

            // // while the shapes are not the same length, prepend a 1 to the smaller shape
            // while (this_shape.size() < other_shape.size()) {
            //     this_shape.insert(this_shape.begin(), 1);
            // }

            // while (other_shape.size() < this_shape.size()) {
            //     other_shape.insert(other_shape.begin(), 1);
            // }

            vector<vector<int>> padded_shapes = PadShapes(this->shape, other.shape);
            vector<int> this_shape = padded_shapes[0];
            vector<int> other_shape = padded_shapes[1];

            bool equal = true;
            vector<int> new_shape = vector<int>(this_shape.size());
            for (int i = 0; i < this_shape.size(); i++) {
                if (this_shape[i] != other_shape[i] && this_shape[i] != 1 && other_shape[i] != 1) {
                    equal = false;
                } else {
                    new_shape[i] = this_shape[i] > other_shape[i] ? this_shape[i] : other_shape[i];
                }
            }

            if (!equal) {
                throw std::invalid_argument("Tensors must have the same shape or axis must be 1");
            }


            vector<float> new_data = vector<float>(ShapeToSize(new_shape));
            Tensor new_tensor = Tensor(new_data, new_shape);

            for (vector<int> index : AllIndicies(new_shape)) {
                vector<int> this_index = vector<int>(index.size());
                vector<int> other_index = vector<int>(index.size());
                for (int i = 0; i < index.size(); i++) {
                    if (this_shape[i] == 1) {
                        this_index[i] = 0;
                    } else {
                        this_index[i] = index[i];
                    }

                    if (other_shape[i] == 1) {
                        other_index[i] = 0;
                    } else {
                        other_index[i] = index[i];
                    }
                }

                new_tensor.Set(index, func(this->Get(this_index), other.Get(other_index)));
            }

            return new_tensor;
        }


        Tensor AddTensor(Tensor other) {
            return this->Broadcast([](float a, float b) {
                return a + b;
            }, other);
        }


        Tensor MulTensor(Tensor other) {
            return this->Broadcast([](float a, float b) {
                return a * b;
            }, other);
        }


        Tensor SubTensor(Tensor other) {
            return this->Broadcast([](float a, float b) {
                return a - b;
            }, other);
        }


        Tensor DivTensor(Tensor other) {
            return this->Broadcast([](float a, float b) {
                return a / b;
            }, other);
        }


        Tensor MatMul(Tensor other) {
            // mat mul between the last two dimensions of each tensor
            if (this->shape.size() < 2 || other.shape.size() < 2) {
                throw std::invalid_argument("Tensors must have at least 2 dimensions");
            }

            int this_dim = this->shape.size() - 1;
            int other_dim = other.shape.size() - 2;

            if (this->shape[this_dim] != other.shape[other_dim]) {
                throw std::invalid_argument("Dimensions not compatible for matmul");
            }

            vector<vector<int>> padded_shapes = PadShapes(this->shape, other.shape);
            vector<int> this_shape = padded_shapes[0];
            vector<int> other_shape = padded_shapes[1];

            vector<int> new_shape = vector<int>(this_shape.size());
            for (int i = 0; i < this_shape.size() - 2; i++) {
                new_shape[i] = this_shape[i] > other_shape[i] ? this_shape[i] : other_shape[i];
            }
            new_shape[this_shape.size() - 2] = this_shape[this_shape.size() - 2];
            new_shape[this_shape.size() - 1] = other_shape[other_shape.size() - 1];

            vector<float> new_data = vector<float>(ShapeToSize(new_shape));
            Tensor new_tensor = Tensor(new_data, new_shape);

            for (vector<int> index : AllIndicies(new_shape)) {
                vector<int> this_index = vector<int>(index.size());
                vector<int> other_index = vector<int>(index.size());
                int row = index[index.size() - 2];
                int col = index[index.size() - 1];
                for (int i = 0; i < index.size() - 2; i++) {
                    this_index[i] = index[i];
                    other_index[i] = index[i];
                }
                this_index[this_index.size() - 2] = row;
                this_index[this_index.size() - 1] = 0;
                other_index[other_index.size() - 2] = 0;
                other_index[other_index.size() - 1] = col;

                float sum = 0;
                for (int i = 0; i < this->shape[this_dim]; i++) {
                    this_index[this_index.size() - 1] = i;
                    other_index[other_index.size() - 2] = i;
                    sum += this->Get(this_index) * other.Get(other_index);
                }

                new_tensor.Set(index, sum);
            }

            return new_tensor;
        }
        

        Tensor MulScaler(float scaler) {
            vector<float> new_data = vector<float>(this->data);
            for (int i = 0; i < this->data.size(); i++) {
                new_data[i] = this->data[i] * scaler;
            }

            return Tensor(new_data, this->shape);
        }

        Tensor PowScaler(float scaler) {
            vector<float> new_data = vector<float>(this->data);
            for (int i = 0; i < this->data.size(); i++) {
                new_data[i] = pow(this->data[i], scaler);
            }

            return Tensor(new_data, this->shape);
        }

        Tensor Exp() {
            vector<float> new_data = vector<float>(this->data);
            for (int i = 0; i < this->data.size(); i++) {
                new_data[i] = exp(this->data[i]);
            }

            return Tensor(new_data, this->shape);
        }

        // Tensor ApplyMul(Tensor other, int dim) {
        //     return this->Apply([&other, dim](vector<float> v) {
        //         vector<float> result = vector<float>(v.size());
        //         for (int i = 0; i < v.size(); i++) {
        //             result[i] = v[i] * other.data[i];
        //         }
        //         return result;
        //     }, dim);
        // }

        Tensor Normalize(int dim) {
            // divide each element by the sum of the row
            return this->Apply([](vector<float> v) {
                float sum = 0;
                for (int i = 0; i < v.size(); i++) {
                    sum += v[i];
                }
                vector<float> new_row = vector<float>(v.size());
                for (int i = 0; i < v.size(); i++) {
                    new_row[i] = v[i] / sum;
                }
                return new_row;
            }, dim);
        }


        void Reshape(vector<int> shape) {
            // Assert that the new shape has the same size as the old shape
            int new_size = 1;
            for (int i = 0; i < shape.size(); i++) {
                new_size *= shape[i];
            }

            if (new_size != this->size) {
                // error message with the original shape and the new shape
                string original_shape = "";
                for (int i = 0; i < this->shape.size(); i++) {
                    original_shape += std::to_string(this->shape[i]) + " ";
                }
                string new_shape = "";
                for (int i = 0; i < shape.size(); i++) {
                    new_shape += std::to_string(shape[i]) + " ";
                }
                throw std::invalid_argument("The new shape must have the same size as the old shape.\nOriginal shape: " + original_shape + "\n" + "New shape: " + new_shape);
            }

            // Set the new shape
            this->shape = shape;
        }


        float Get(vector<int> index) {
            return this->data[ToLinearIndex(this->shape, index)];
        }


        void Set(vector<int> index, float value) {
            this->data[ToLinearIndex(this->shape, index)] = value;
        }

        // ToString
        string ToString() {
            // Create a string to hold the tensor
            string tensor_string = "";
            // add the shape
            tensor_string += "Shape: ";
            for (int i = 0; i < this->shape.size(); i++) {
                tensor_string += std::to_string(this->shape[i]);
                if (i < this->shape.size() - 1) {
                    tensor_string += ", ";
                }
            }
            tensor_string += "\nData: ";
            // Loop through the data
            for (int i = 0; i < this->data.size(); i++) {
                // Add the value to the string
                tensor_string += std::to_string(this->data[i]);
                // Add a comma if it is not the last value
                if (i != this->data.size() - 1) {
                    tensor_string += ", ";
                }
            }
            // Return the string
            return tensor_string;
        }

        static Tensor Zeros(vector<int> shape) {
            int size = 1;
            for (int i = 0; i < shape.size(); i++) {
                size *= shape[i];
            }
            return Tensor(vector<float>(size), shape);
        }
};