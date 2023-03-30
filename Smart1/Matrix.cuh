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

__global__ void multplyMatrices(float* A, float* B, float* C, int widthA, int widthB, int heightA, int heightB);
__global__ void activateMatrices(float* A, int widthA, int heightA, uint8_t activatioonFunction = SIGMOID_FUNCTION);
__global__ void substractMatrices(float* A, float* B, float* C, int widthA, int heightA);
__global__ void addMatrices(float* A, float* B, float* C, int widthA, int heightA);
__global__ void multiplyWithDerivate(float* outputs, float* errorSignal, int sizeOutputs, uint8_t activatioonFunction = SIGMOID_FUNCTION);
__global__ void calculateNewWeights(float* weights, float* Error, float* Inputs, int sizeError, int sizeInputs, float learnRate);

class Matrix {
private:
	int width, height;
	float* dataHost;

	int copyMatrixToDevice();
	int copyMatrixToHost();
public:
	float* dataDevice;

	Matrix(int width, int height, int* errorStatus, uint8_t method = 0, float seed = 0);
	Matrix::Matrix();

	void deleteMatrix();

	unsigned int getWidth();

	unsigned int getHeight();

	float getData(unsigned int x, unsigned int y);

	void setData(unsigned int x, unsigned int y, float value);

	void printMatrix();

	Matrix operator * (Matrix B);
	Matrix operator - (Matrix B);
	Matrix operator + (Matrix B);
	void multiplyWithDerivateMatrix(Matrix* errorSignal, int activationFunction = SIGMOID_FUNCTION);
	void calculateNewWeightsMatrix(Matrix* Inputs, Matrix* Error, float learnRate);
};