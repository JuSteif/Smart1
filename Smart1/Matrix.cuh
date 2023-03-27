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

__global__ void multplyMatricesAndActivate(float* A, float* B, float* C, int widthA, int widthB, int heightA, int heightB, uint8_t activatioonFunction = SIGMOID_FUNCTION);

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
};