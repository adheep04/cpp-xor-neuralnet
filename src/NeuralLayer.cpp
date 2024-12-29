#include "NeuralLayer.h"
#include <vector>
#include <random>
using std::vector;

// implemented constructor
NeuralLayer::NeuralLayer(int inDim, int outDim) {
    this->inDim = inDim;
    this->outDim = outDim;

    // weight with vector shape (outDim, inDim)
    this->weight = vector<vector<float>>(
        outDim, 
        vector<float>(inDim)
        );
    
    // initialize weights to all have random values in range (-0.1, 0.1)
    initWeight(this->weight);
    
    // initialize bias of all 0s with vector shape (outDim, 1)
    this->bias = vector<vector<float>>(
        outDim, 
        vector<float>(1)
        );

    // gradient of the weights with respect to the loss
    this->gradWeight = {};
    // gradient of the bias with respect to the loss
    this->gradBias = {};
    // gradient of this layer's inputs with respect to loss
    // (gets propagated backward)
    this->gradOut = {};

    // stores the last input during forward pass
    this->lastInput = {};

}


// implemented weight initialization
void initWeight(vector<vector<float>> weight) {
    // auto& allows access by reference for modification
    // replace all weight matrix elements with random small value
    for (auto& row : weight) {
        for (auto& val : row) {
            val = ((std::rand() - (RAND_MAX/2)) / static_cast<float>(RAND_MAX)) * 0.2f;
        }
    }
}

// addition operation
vector<vector<float>> add(vector<vector<float>> weight, vector<vector<float>> bias) {
    int outDim =  
}

// matrix multiply operation
vector<vector<float>> matMul(vector<vector<float>> weight, vector<vector<float>> x) {

}


