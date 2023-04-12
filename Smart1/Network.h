#pragma once
#include <vector>

#include "Layer.h"

/**
 * @brief main class for neural network
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
	 * @return Success if creation was successful
	 */
	int addLayer(int sizeOutputs);
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
	 * @param Target Sets the desired output for this network
	 */
	void Backpropogation(float learnRate, Matrix Target);
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
};