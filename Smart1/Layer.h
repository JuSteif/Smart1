#pragma once
#include "Matrix.cuh"

/**
 * @brief Represents one layer in an network
 * 
 */
class Layer {
private:
public:
	/**
	 * @brief Determines if it is the last layer in a network
	 * 
	 */
	bool outputNeuron;
	/**
	 * @brief Weights within the layer
	 * 
	 */
	Matrix Weights;
	/**
	 * @brief The output from the layer
	 * 
	 */
	Matrix Outputs;
	/**
	 * @brief A pointer to the outputs from the last layer, wich are the inputs for this one.
	 * 
	 */
	Matrix* Inputs;


	/**
	 * @brief A pointer to the errorsignal from the previous layer
	 * 
	 */
	Matrix* PreviousErrorSignal;
	/**
	 * @brief A pointer to the weights from the previous layer
	 * 
	 */
	Matrix* PreviousWeights;
	/**
	 * @brief The errorsignal from the current layer
	 * 
	 */
	Matrix ErrorSignal;

	/**
	 * @brief Construct a new Layer object
	 * 
	 * @param sizeInput Size of the last layers output or the network input
	 * @param sizeOutput Size of the output from the current layer
	 * @param outputNeuron If true it is the last layer of the network, if false it is a hidden layer
	 * @param Inputs Inputs for the new layer
	 * @param PreviousWeights Weights of the next layer
	 * @param PreviousErrorSignal Errorsignal from the next layer
	 */
	Layer(int sizeInput, int sizeOutput, bool outputNeuron, Matrix* Inputs, Matrix* PreviousWeights, Matrix* PreviousErrorSignal);
	/**
	 * @brief deletes the current layer
	 * 
	 */
	void deleteLayer();
	/**
	 * @brief Performs the forwardpass for the current layer
	 * 
	 */
	void forward();
	/**
	 * @brief Calculates the errorsignal of the current layer, if the errorsignal from the next layer was calculated before
	 * 
	 */
	void calculatErrorSignal();
	/**
	 * @brief If the errorsignal was calculated, this function will calculate the new weights
	 * 
	 * @param learnRate Rate wich determines how fast the layer will learn
	 */
	void calculateNewWeights(float learnRate);
};