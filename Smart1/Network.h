#pragma once
#include <vector>

#include "Layer.h"

class Network {
private:
public:
	std::vector<Layer*> layers;
	Matrix Inputs;

	Network(int sizeInputs);
	void deleteNetwork();

	int addLayer(int sizeOutputs);
	int prepareNetwork();
	int forward();
	Matrix* getNetworkOutput();
	void printNetwork();
};