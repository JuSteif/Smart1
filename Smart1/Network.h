#pragma once
#include <vector>

#include "Layer.h"

class Network {
private:
public:
	std::vector<Layer*> layers;
	Matrix Inputs;
	Matrix Target;
	float learnRate;

	Network(int sizeInputs);
	void deleteNetwork();

	int addLayer(int sizeOutputs);
	int prepareNetwork();
	int forward();
	Matrix* getNetworkOutput();
	void printNetwork();

	void Backpropogation(float learnRate, Matrix Target);
	void calcErrorSignal();
	void calcNewWeights();
};