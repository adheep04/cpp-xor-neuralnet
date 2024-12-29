#ifndef NEURALLAYER_H 
#define NEURALLAYER_H
#include <vector>
using std::vector;

class NeuralLayer {
private:
    int inDim;
    int outDim;
    vector<vector<float>> weight;
    vector<vector<float>> bias;
    vector<vector<float>> lastInput;
    vector<vector<float>> gradWeight;
    vector<vector<float>> gradBias;
    vector<vector<float>> gradOut;
public:

    // constructor
    NeuralLayer(int inDim, int outDim);

    // forward pass
    vector<vector<float>> forward(vector<vector<float>> x);

    // backward pass
    vector<vector<float>> backward();

    // getters
    vector<vector<float>> getLastInput();
};

#endif