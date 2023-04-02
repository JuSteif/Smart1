#include "Network.h"

Network::Network(int sizeInputs) {
	int error;
	Inputs = Matrix(1, sizeInputs, &error, 2, 1);
}

void Network::deleteNetwork() {
	Inputs.deleteMatrix();

	for (Layer* layer : layers) {
		layer->deleteLayer();
	}
}

int Network::addLayer(int sizeOutputs) {
	Matrix* mat;
	if (layers.size() < 1) {
		mat = &Inputs;
	}
	else {
		mat = &(layers[layers.size() - 1]->Outputs);
	}

	Layer* layer = new Layer(mat->getHeight(), sizeOutputs, false, mat, NULL, NULL);
	layers.push_back(layer);

	return 0;
}

int Network::prepareNetwork() {
	layers[layers.size() - 1]->outputNeuron = true;

	for (int i = layers.size() - 2; i >= 0; i--) {
		layers[i]->PreviousErrorSignal = &(layers[i + 1]->ErrorSignal);
		layers[i]->PreviousWeights = &(layers[i + 1]->Weights);
	}

	return 0;
}

int Network::forward() {
	for (Layer* layer : layers) {
		layer->forward();
	}

	return 0;
}

Matrix* Network::getNetworkOutput() {
	return &(layers[layers.size() - 1]->Outputs);
}

void Network::printNetwork() {
	for (Layer* layer : layers) {
		layer->Weights.printMatrix();
		printf("\n");
		layer->Outputs.printMatrix();
	}
}

void Network::Backpropogation(float learnRate, Matrix Target) {
	this->Target = Target;
	this->learnRate = learnRate;
	
	layers[layers.size() - 1]->PreviousErrorSignal = &Target;

	calcErrorSignal();
	calcNewWeights();
}

void Network::calcErrorSignal() {
	for (int i = layers.size() - 1; i >= 0; i--)
	{
		layers[i]->calculatErrorSignal();
	}
}

void Network::calcNewWeights() {
	for (int i = layers.size() - 1; i >= 0; i--)
	{
		layers[i]->calculateNewWeights(learnRate);
	}
}