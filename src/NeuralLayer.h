#ifndef NEURALLAYER_H 
#define NEURALLAYER_H
#include <vector>
using std::vector;

class NeuralLayer {
private:
    using tensor2d = vector<vector<float>>;
    int in_dim;
    int out_dim;
    tensor2d weight;
    tensor2d bias;
    tensor2d last_x;
    tensor2d grad_weight;
    tensor2d grad_bias;
public:

    // constructor
    NeuralLayer(int in_dim, int out_dim);

    // forward pass
    tensor2d forward(tensor2d x);

    // backward pass
    tensor2d backward(tensor2d grad_in);

    tensor2d operator()(tensor2d x);
};

#endif