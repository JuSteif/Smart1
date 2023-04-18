#pragma once
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <String>

#include "Layer.h"

/**
 * @brief Main class for neural network
*/
class Network {
private:
public:
	/**
	 * @brief Vector of all layers in one network
	 * 
	 */
	std::vector<Layer*> layers;
	/**
	 * @brief Inputs for network
	 * 
	 */
	Matrix Inputs;
	/**
	 * @brief Correct Outputs for network
	 * 
	 */
	Matrix Target;
	/**
	 * @brief Determines how fast this network will learn
	 * 
	 */
	float learnRate;

	/**
	 * @brief Construct a new Network object
	 * 
	 * @param sizeInputs size of inputs
	 */
	Network(int sizeInputs);
	/**
	 * @brief deletes complete network and all it`s layers
	 * 
	 */
	void deleteNetwork();

	/**
	 * @brief Adds new layer to network
	 *
	 * @param sizeOutputs size of outputs from this layer
	 * @param activationFunction Determines the activation function that will be used
	 * @return Success if creation was successful
	 */
	int addLayer(int sizeOutputs, int activationFunction);
	/**
	 * @brief Prepares the network, this function must be called before first forward pass
	 * 
	 * @return int 
	 */
	int prepareNetwork();
	/**
	 * @brief Performs the forward pass for complete network
	 * 
	 * @return int 
	 */
	int forward();
	/**
	 * @brief Get the Network Output
	 * 
	 * @return Matrix* 
	 */
	Matrix* getNetworkOutput();
	/**
	 * @brief Prints complete network to console
	 * 
	 */
	void printNetwork();

	/**
	 * @brief Performs backpropogation for complete network on one example
	 * 
	 * @param learnRate Determines how fast this network learns
	 */
	void Backpropogation(float learnRate);
	/**
	 * @brief Calculates all error signals for complete network
	 * 
	 */
	void calcErrorSignal();
	/**
	 * @brief Calculates new weights for complete network. Error signals have to be calculated before.
	 * 
	 */
	void calcNewWeights();

	/**
	 * @brief Set the Input for the neural net
	 * 
	 * @param data Used for Input. Should have the exact same size as the determined input
	 */
	void setInput(float* data);

	/**
	 * @brief Get the Output of this network
	 * 
	 * @param size size of output
	 * @return float* output array
	 */
	float* getOutputArray(int* size);

	/**
	 * @brief Set the Target for current network run
	 * 
	 * @param targetData Determines the data for correct output, wich is used by backpropogation
	 */
	void setTarget(float* targetData);

	void safeNetwork(char* path);
};