#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <cmath>

#include "util.h"

using std::vector;
using std::string;

int ToLinearIndex(vector<int> shape, vector<int> index) {
    // 2D linear index:
    // i = row * num_cols + col
    // i = index[0] * shape[1] + index[1]

    // 3D linear index:
    // i = (row * num_cols * num_depths) + (col * num_depths) + depth
    // i = index[0] * shape[1] * shape[2] + index[1] * shape[2] + index[2]

    // etc

    int linear_index = 0;
    int multiplier = 1;
    for (int i = shape.size()-1; i >= 0; i--) {
        linear_index += index[i] * multiplier;
        multiplier *= shape[i];
    }

    return linear_index;
}


vector<int> FromLinearIndex(vector<int> shape, int linear_index) {
    // 2D linear index:
    // i = row * num_cols + col
    // i = index[0] * shape[1] + index[1]

    // 3D linear index:
    // i = (row * num_cols * num_depths) + (col * num_depths) + depth
    // i = index[0] * shape[1] * shape[2] + index[1] * shape[2] + index[2]

    // etc

    vector<int> index(shape.size());
    int multiplier = 1;
    for (int i = shape.size()-1; i >= 0; i--) {
        multiplier *= shape[i];
    }
    for (int i = 0; i < shape.size(); i++) {
        multiplier /= shape[i];
        index[i] = linear_index / multiplier;
        linear_index -= index[i] * multiplier;
    }

    return index;
}


int ShapeToSize(vector<int> shape) {
    int size = 1;
    for (int i = 0; i < shape.size(); i++) {
        size *= shape[i];
    }
    return size;
}


vector<vector<int>> AllIndicies(vector<int> shape) {
    vector<vector<int>> indicies = vector<vector<int>>(ShapeToSize(shape), vector<int>(shape.size()));

    for (int i = 0; i < indicies.size(); i++) {
        indicies[i] = FromLinearIndex(shape, i);
    }

    return indicies;
}


vector<vector<int>> PadShapes(vector<int> shape1, vector<int> shape2) {
    // get the shapes but as a copy
    vector<int> this_shape = vector<int>(shape1);
    vector<int> other_shape = vector<int>(shape2);

    for (int i = 0; i < this_shape.size(); i++) {
        this_shape[i] = shape1[i];
    }
    for (int i = 0; i < other_shape.size(); i++) {
        other_shape[i] = shape2[i];
    }

    // while the shapes are not the same length, prepend a 1 to the smaller shape
    while (this_shape.size() < other_shape.size()) {
        this_shape.insert(this_shape.begin(), 1);
    }

    while (other_shape.size() < this_shape.size()) {
        other_shape.insert(other_shape.begin(), 1);
    }

    // return the shapes
    return {this_shape, other_shape};
}