#include "NeuralLayer.h"
#include <vector>
#include <random>
#include <cassert>
using std::vector;

using tensor1d = vector<float>;
using tensor2d = vector<tensor1d>;

// implemented constructor
NeuralLayer::NeuralLayer(int in_dim, int out_dim) {
    this->in_dim = in_dim;
    this->out_dim = out_dim;
    // stores the last input during forward pass
    this->last_x = {};

    // weight with vector shape (out_dim, in_dim)
    this->weight = tensor2d(out_dim, tensor1d(in_dim));
    // initialize weights to all have random values in range (-0.1, 0.1)
    init_weight(this->weight);
    // initialize bias of all 0s with vector shape (out_dim, 1)
    this->bias = tensor2d(out_dim, tensor1d(1));

    // gradient of the weights with respect to the loss
    this->grad_weight = {};
    // gradient of the bias with respect to the loss
    this->grad_bias = {};

}

tensor2d NeuralLayer::operator()(tensor2d x) {
    return forward(x);
}


tensor2d NeuralLayer::forward(tensor2d x) {
    // set last input
    this->last_x = x;
    // return forward pass
    return add(mat_mul(this->weight, x), this->bias);
}

// grad_in is this the derivative of this layer's output w/r to loss
tensor2d NeuralLayer::backward(tensor2d out_grad) {
    /*
    note: 
    the derivative of this layer's parameters w/r to loss = the derivative
    of the layer's output w/r to layer parameters * the derivative
    of the loss w/r to this layer's output -> dO/dP * dL/dO = dL/dP
    */
   tensor1d out_grad_1d = get_column(out_grad, 0); // converts (d, 1) 2d tensor to (d) 1d tensor
   tensor1d last_x_1d = get_column(this->last_x, 0);

   this->grad_weight = outer(out_grad_1d, last_x_1d);
   this->grad_bias = out_grad;
   return mat_mul(out_grad, this->weight);
}

tensor2d outer(tensor1d a, tensor1d b) {
    int n_row = a.size();
    int n_col = b.size();
    tensor2d outer_product = tensor2d(n_row, tensor1d(n_col));
    for(int c = 0; c < n_col; c++) {
        for(int r = 0; r < n_row; r++) {
            outer_product[r][c] = a[r] * b[c];
        }
    }

    return outer_product;
}


void init_weight(tensor2d weight) {
    // auto& allows access by reference for modification
    // replace all weight matrix elements with random small value
    for (auto& row : weight) {
        for (auto& val : row) {
            val = ((std::rand() - (RAND_MAX/2)) / static_cast<float>(RAND_MAX)) * 0.2f;
        }
    }
}


// addition operation
// out_vector: (out_size, 1), bias: (out_size, 1)
tensor2d add(const tensor2d& a, const tensor2d& b) {
    assert(a.size() == b.size());

    // get output dimensions
    int n_row = a.size();
    int n_col = a[0].size();

    // initialize output tensor
    tensor2d sum = tensor2d(n_row, tensor1d(n_col));

    for(int r = 0; r < n_row; r++) {
        // validating dimensions for row
        assert((a[r].size() == b[r].size()) && (a[r].size() == n_col));

        for(int c = 0; c < n_col; c++) {
            sum[r][c] = a[r][c] + b[r][c];
        }
    }
    return sum;
}

// matrix multiply operation for weights and input
tensor2d mat_mul(const tensor2d& a, const tensor2d& b) {
    // validate row num
    assert(a.size() == b.size());

    // get dimensions for product vector
    int n_row = a.size();
    int n_col = b[0].size();

    // initialize output vector
    tensor2d out_matrix = tensor2d(n_row, tensor1d(n_col));

    for (int r = 0; r < n_row; r++) {
        // validate col size
        assert((a[r].size() == b[r].size()) && (a[r].size() == n_col));

        for (int c = 0; c < n_col; c++) {
            out_matrix[r][c] = dot(a[r], get_column(b, c));
        }
    }

    return out_matrix;
}

// returns the dot product of 2 1d vectors
float dot(const tensor1d& a, const tensor1d& b) {
    assert(a.size() == b.size());

    int length{a.size()};
    float dot_sum{0};

    for(int i = 0; i < length; i++) {
        dot_sum += a[i] * b[i];
    }

    return dot_sum;
}

tensor1d get_column(const tensor2d& tensor, int n_col) {
    // initialize empty vector
    tensor1d column;

    // iterate through row and append item to new vector
    for (tensor1d row : tensor) {
        column.emplace_back(row[n_col]);
    } 
    
    // validate dimensions
    assert(column.size() == tensor.size());

    return column;
}



