#include "Network.h"

Network::Network(int sizeInputs) {
	int error;
	Inputs = Matrix(1, sizeInputs + 1, &error, 0);
	Inputs.setData(Inputs.getWidth() - 1, Inputs.getHeight() - 1, 1);
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
		printf("Weights: width %d  height %d\n", layer->Weights.getWidth(), layer->Weights.getHeight());
		layer->Weights.printMatrix();
		printf("\nOutputs: height %d\n", layer->Outputs.getHeight());
		layer->Outputs.printMatrix();
		printf("\nErrorSignal: height %d\n", layer->ErrorSignal.getHeight());
		layer->ErrorSignal.printMatrix();
		printf("________________________\n");
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
		/*printf("\n___________________________________\nError Signal %d:\n", i);
		layers[i]->ErrorSignal.printMatrix();*/
	}
}

void Network::calcNewWeights() {
	for (int i = layers.size() - 1; i >= 0; i--)
	{
		layers[i]->calculateNewWeights(learnRate);
		/*printf("\n___________________________________\nnew Weights %d:\n", i);
		layers[i]->Weights.printMatrix();*/
	}
}

void Network::setInput(float* data){
	memcpy(Inputs.dataHost, data, sizeof(float) * (Inputs.getHeight() - 1));
}

float* Network::getOutputArray(int* size){
	*size = this->getNetworkOutput()->getHeight() - 1;
	return this->getNetworkOutput()->dataHost;
}