﻿#include "Network.h"



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

int Network::addLayer(int sizeOutputs, int activationFunction) {
	Matrix* mat;
	if (layers.size() < 1) {
		mat = &Inputs;
	}
	else {
		mat = &(layers[layers.size() - 1]->Outputs);
	}

	Layer* layer = new Layer(mat->getHeight(), sizeOutputs, false, mat, NULL, NULL, activationFunction);
	layers.push_back(layer);

	return 0;
}

int Network::prepareNetwork() {
	layers[layers.size() - 1]->outputNeuron = true;
	int error;
	Target = Matrix(1, layers[layers.size() - 1]->Outputs.getHeight() - 1, &error, 0);

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

void Network::Backpropogation(float learnRate) {
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

	/*Inputs.copyMatrixToDevice(0, InputsHeight.getHeight() - 2);
	activateMatrices <<<Inputs.getHeight() / BLOCK_SIZE + 1, BLOCK_SIZE >>> (Inputs.dataDevice, Inputs.getWidth(), Inputs.getHeight() - 1, activationFunction);
	Inputs.copyMatrixToHost(0, InputsHeight.getHeight() - 2);
	//*/
}

float* Network::getOutputArray(int* size){
	*size = this->getNetworkOutput()->getHeight() - 1;
	return this->getNetworkOutput()->dataHost;
}

void Network::setTarget(float* targetData) {
	memcpy(Target.dataHost, targetData, sizeof(float) * (Target.getHeight()));
}

void Network::safeNetwork(char* path) {
	std::vector<int> sizesNetwork;
	
	std::string fileName(path);
	fileName.append(".smart");

	std::ofstream file;
	file.open(fileName, std::ios::binary);

	if (!file.is_open()) {
		printf("Can`t open file");
		return;
	}

	int netSize = layers.size();
	file.write((char*)&netSize, sizeof(netSize));
	int inputsSize = Inputs.getHeight();
	file.write((char*)&inputsSize, sizeof(inputsSize));
	sizesNetwork.push_back(inputsSize);
	for (int i = 0; i < netSize; i++) {
		int outputsSize = layers[i]->Outputs.getHeight();
		file.write((char*)&outputsSize, sizeof(outputsSize));
		sizesNetwork.push_back(outputsSize);
		int rem = layers[i]->activationFunction;
		file.write((char*)&rem, sizeof(rem));
	}

	for (int i = 0; i < layers.size(); i++) {
		for (int j = 0; j < layers[i]->Weights.getHeight() * layers[i]->Weights.getWidth(); j++) {
			float rem = layers[i]->Weights.dataHost[j];
			printf("%f\n", rem);
			file.write((char*)&rem, sizeof(rem));
		}
	}

	

	file.close();
}

Network::Network(char* path){
	std::ifstream file;
	file.open(path, std::ios::binary);

	if (!file.is_open()) {
		printf("Can`t open file");
		return;
	}

	int netSize;
	file.read((char*)&netSize, sizeof(netSize));
	
	int inputsSize;
	file.read((char*)&inputsSize, sizeof(inputsSize));
	printf("%d\n", inputsSize);
	int error;
	Inputs = Matrix(1, inputsSize, &error, 0);
	Inputs.setData(Inputs.getWidth() - 1, Inputs.getHeight() - 1, 1);

	for (int i = 0; i < netSize; i++) {
		int outputsSize;
		file.read((char*)&outputsSize, sizeof(outputsSize));
		printf("%d\n", outputsSize);
		int aF;
		file.read((char*)&aF, sizeof(aF));
		addLayer(outputsSize - 1, aF);
	}

	for (int i = 0; i < layers.size(); i++) {
		for (int j = 0; j < layers[i]->Weights.getHeight() * layers[i]->Weights.getWidth(); j++) {
			file.read((char*)&(layers[i]->Weights.dataHost[j]), sizeof(layers[i]->Weights.dataHost[j]));
		}
	}

	prepareNetwork();
}