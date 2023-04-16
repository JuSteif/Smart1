#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 640

#define SIGMOID_FUNCTION 0
#define LINEAR_FUNCTION 1
#define STEP_FUNCTION 2

/**
 * @brief multiplys two matrices and save the result in a third matrix
 * 
 * @param A Factor 1
 * @param B Factor 2
 * @param C Result
 * @param widthA Width of matrix A
 * @param widthB Width of matrix B
 * @param heightA Height of matrix A
 * @param heightB Height of matrix B
 */
__global__ void multplyMatrices(float* A, float* B, float* C, int widthA, int widthB, int heightA, int heightB);
/**
 * @brief Applies the activation function to all elements of the matrix
 * 
 * @param A Matrix
 * @param widthA Width of Matrix A
 * @param heightA Height of Matrix A
 * @param activatioonFunction Activation function that will be applied to the matrix
 */
__global__ void activateMatrices(float* A, int widthA, int heightA, uint8_t activatioonFunction = SIGMOID_FUNCTION);
/**
 * @brief Substracts two matrices and save the result in a third matrix C = A - B
 * 
 * @param A First Matrix
 * @param B Second Matrix
 * @param C Result
 * @param widthA Width of all matrices
 * @param heightA Height of all matrices
 * @return __global__ 
 */
__global__ void substractMatrices(float* A, float* B, float* C, int widthA, int heightA);
/**
 * @brief Adds two matrices and save the result in a third matrix C = A + B
 * 
 * @param A First Matrix
 * @param B Second Matrix
 * @param C Result
 * @param widthA Width of all matrices
 * @param heightA Height of all matrices
 * @return __global__ 
 */
__global__ void addMatrices(float* A, float* B, float* C, int widthA, int heightA);
/**
 * @brief Multiplies all elements of matrix errorsignal with the derivate of the current outputs. The error signal have to be calculated to do this step.
 * 
 * @param outputs Outputs from current layer
 * @param errorSignal ErrorSignal from currentlayer
 * @param sizeOutputs Size of outputs
 * @param activatioonFunction Determines the activation function
 */
__global__ void multiplyWithDerivate(float* outputs, float* errorSignal, int sizeOutputs, uint8_t activatioonFunction = SIGMOID_FUNCTION);
/**
 * @brief Calculates the new weights. The error signal have to be calculated before this step
 * 
 * @param weights Weights wich will be updated
 * @param Error Rrror signal wich must be calculated before
 * @param Inputs Inputs of the layer
 * @param sizeError Size of matrix error signal
 * @param sizeInputs size of inputs
 * @param learnRate Determines how fast the layer will learn
 */
__global__ void calculateNewWeights(float* weights, float* Error, float* Inputs, int sizeError, int sizeInputs, float learnRate);
/**
 * @brief Calculates the scalar product of error and weights 
 * 
 * @param weights Weights wich will be modified 
 * @param previousErrorSignal Error signal of next layer
 * @param errorSignal Error signal of current layer
 * @param sizeError Size of error
 * @param sizePreviousError Size of next layer error signal
 */
__global__ void multiplyAndSum(float* weights, float* previousErrorSignal, float* errorSignal, int sizeError, int sizePreviousError);

class Matrix {
private:
	/**
	 * @brief Copies part of Matrix from Host to device. The part is defiend by a rectangle wich is determined by two points
	 * 
	 * @param leftTopX X-Coordinate from first point
	 * @param leftTopY Y-Coordinate from first point
	 * @param bottomRightX X-Coordinate from second point
	 * @param bottomRightY Y-Coordinate from second point
	 * @return return cudaSuccess if coopying was successful
	 */
	int copyMatrixToDevice(int leftTopX, int leftTopY, int bottomRightX, int bottomRightY);
	/**
	 * @brief Copies part of Matrix from Device to Host. The part is defiend by a rectangle wich is determined by two points
	 * 
	 * @param leftTopX X-Coordinate from first point
	 * @param leftTopY Y-Coordinate from first point
	 * @param bottomRightX X-Coordinate from second point
	 * @param bottomRightY Y-Coordinate from second point
	 * @return return cudaSuccess if coopying was successful
	 */
	int copyMatrixToHost(int leftTopX, int leftTopY, int bottomRightX, int bottomRightY);

	int copyMatrixToDeviceVector(int top, int bottom);

	int copyMatrixToHostVector(int top, int bottom);
public:
	/**
	 * @brief height and width of current matrix
	 * 
	 */
	int width, height;
	/**
	 * @brief Data stored on normal RAM
	 * 
	 */
	float* dataHost;
	/**
	 * @brief Data stored on GPU
	 * 
	 */
	float* dataDevice;

	/**
	 * @brief Construct a new Matrix object
	 * 
	 * @param width Width of matrix
	 * @param height Height of matrix
	 * @param errorStatus pointer to error if something went wrong
	 * @param method 0 = not initalised, 1 = initalised with seed parameter, 2 = random initalised 
	 * @param seed Start value for every element
	 */
	Matrix(int width, int height, int* errorStatus, uint8_t method = 0, float seed = 0);
	/**
	 * @brief Construct a new Matrix, no elements in this matrix
	 * 
	 */
	Matrix::Matrix();

	/**
	 * @brief deletes current matrix
	 * 
	 */
	void deleteMatrix();

	/**
	 * @brief Get the Width of this matrix
	 * 
	 * @return unsigned int 
	 */
	unsigned int getWidth();

	/**
	 * @brief Get the Height of this matrix
	 * 
	 * @return unsigned int 
	 */
	unsigned int getHeight();

	/**
	 * @brief Get the Data of this object on single position
	 * 
	 * @param x X-Coordinate
	 * @param y Y-Coordinate
	 * @return float 
	 */
	float getData(unsigned int x, unsigned int y);

	/**
	 * @brief Set the Data of this object on single position
	 * 
	 * @param x X-Coordinate
	 * @param y Y-Coordinate
	 * @param value value wich will be set on this position
	 */
	void setData(unsigned int x, unsigned int y, float value);

	/**
	 * @brief Prints whole Matrix
	 * 
	 */
	void printMatrix();

	/**
	 * @brief Sultiplies two matrices
	 * 
	 * @param B Second factor
	 * @return Product
	 */
	Matrix operator * (Matrix B);
	/**
	 * @brief Substract two matrices
	 * 
	 * @param B Second factor
	 * @return Result
	 */
	Matrix operator - (Matrix B);
	/**
	 * @brief Add two matrices
	 * 
	 * @param B Second factor
	 * @return Sesult
	 */
	Matrix operator + (Matrix B);
	/**
	 * @brief Multipies every element of matrix errorSignal with the derivate of output from current layer
	 * 
	 * @param errorSignal Error signal wich have to be calculated before
	 * @param activationFunction Activation Functions: SIGMOID_FUNCTION LINEAR_FUNCTION STEP_FUNCTION
	 */
	void multiplyWithDerivateMatrix(Matrix* errorSignal, int activationFunction = SIGMOID_FUNCTION);
	/**
	 * @brief Calculates new weights for current layer
	 * 
	 * @param Inputs Inputs of current layer
	 * @param Error Error of current layer
	 * @param learnRate Determines the rate how fast the network will learn
	 */
	void calculateNewWeightsMatrix(Matrix* Inputs, Matrix* Error, float learnRate);
	/**
	 * @brief Calculates new weights for current layer
	 * 
	 * @param weights Matrix that will be updated
	 * @param previousErrorSignal Error signals from next layer
	 */
	void multiplyAndSumMatrix(Matrix* weights, Matrix* previousErrorSignal);
	/**
	 * @brief Forward pass for current layer, matrix multiplication
	 * 
	 * @param A Weights of current layer
	 * @param B Inputs for current layer
	 * @param activationFunction Activation Function: SIGMOID_FUNCTION LINEAR_FUNCTION STEP_FUNCTION
	 */
	void Forward(Matrix& A, Matrix& B, uint8_t activationFunction);
	/**
	 * @brief Substracts target from output of network
	 * 
	 * @param A First matrix
	 * @param B Second matrix
	 */
	void SubstactTargetFromOutput(Matrix& A, Matrix& B);
};