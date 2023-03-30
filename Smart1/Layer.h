#pragma once
#include "Matrix.cuh"

class Layer {
private:
public:
	bool outputNeuron;
	Matrix Weights;
	Matrix Outputs;
	Matrix* Inputs;

	Matrix* PreviousErrorSignal;
	Matrix* PreviousWeights;
	Matrix ErrorSignal;

	Layer(int sizeInput, int sizeOutput, bool outputNeuron, Matrix* Inputs, Matrix* PreviousWeights, Matrix* PreviousErrorSignal);
	void deleteLayer();
	void forward();
	void calculatErrorSignal();
	void calculateNewWeights(float learnRate);
};