#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "StandardActivationFunctions.h"

#define BLOCK_SIZE 640
#define TILE_SIZE 64

__global__ void multplyMatricesAndActivate(float* A, float* B, float* C, int widthA, int widthB, int heightA, int heightB, uint8_t activatioonFunction = SIGMOID_FUNCTION) {
	// declare shared memory for caching input data
	__shared__ float sA[TILE_SIZE][TILE_SIZE];
	__shared__ float sB[TILE_SIZE][TILE_SIZE];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * TILE_SIZE + ty;
	int col = bx * TILE_SIZE + tx;

	float out = 0;

	// loop over tiles of A and B
	for (int i = 0; i < ceil(float(widthA) / TILE_SIZE); i++) {

		// cache tile of A into shared memory
		if (row < heightA && i * TILE_SIZE + tx < widthA)
			sA[ty][tx] = A[row * widthA + i * TILE_SIZE + tx];
		else
			sA[ty][tx] = 0.0;

		// cache tile of B into shared memory
		if (col < widthB && i * TILE_SIZE + ty < heightB)
			sB[ty][tx] = B[(i * TILE_SIZE + ty) * widthB + col];
		else
			sB[ty][tx] = 0.0;

		// synchronize threads before using shared memory
		__syncthreads();

		// perform matrix multiplication using cached data in shared memory
		for (int j = 0; j < TILE_SIZE; j++)
			out += sA[ty][j] * sB[j][tx];

		// synchronize threads before loading new data into shared memory
		__syncthreads();
	}

	// apply activation function to result
	switch (activatioonFunction) {
	case SIGMOID_FUNCTION:
		out = sigmoid(out);
		break;
	case STEP_FUNCTION:
		out = step(out);
		break;
	case LINEAR_FUNCTION:
		out = linear(out);
		break;
	}

	// write result to output matrix
	if (row < heightA && col < widthB)
		C[row * widthB + col] = out;
}


class Matrix {
private:
	int width, height;
	float* dataHost;

	int copyMatrixToDevice() {
		return cudaMemcpy(dataDevice, dataHost, width * height * sizeof(float), cudaMemcpyHostToDevice);
	}

	int copyMatrixToHost() {
		return cudaMemcpy(dataHost, dataDevice, width * height * sizeof(float), cudaMemcpyDeviceToHost);
	}
public:
	float* dataDevice;

	Matrix(int width, int height, int* errorStatus, uint8_t method = 0, float seed = 0): height(height), width(width) {
		*errorStatus = cudaSuccess;
		
		dataHost = (float*)malloc(sizeof(float) * height * width);
		int error = cudaMalloc(&dataDevice, sizeof(float) * width * height);
		if (error != cudaSuccess) {
			printf("Error while allocating matrix-space");
			fflush(stdout);
			*errorStatus = error;
			return;
		}

		if (method == 1) {
			srand(time(0));
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					dataHost[i * width + j] = (float)rand()/(float)rand();
				}
			}
		}
		else if(method == 2)
		{
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					dataHost[i * width + j] = seed;
				}
			}
		}
	}
	
	void deleteMatrix() {
		free(dataHost);
		cudaFree(dataDevice);
	}

	unsigned int getWidth() {
		return width;
	}

	unsigned int getHeight() {
		return height;
	}

	float getData(unsigned int x, unsigned int y) {
		if (x > width || y > height) return 0;
		return dataHost[y * width + x];
	}

	void setData(unsigned int x, unsigned int y, float value) {
		if (x > width || y > height) return;
		dataHost[y * width + x] = value;
	}

	void printMatrix() {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				printf("%+9f ", dataHost[i * width + j]);
			}
			printf("\n");
		}
	}

	Matrix operator * (Matrix B) {
		if (this->width != B.getHeight()) {
			return Matrix(0, 0, NULL);
		}

		int error = cudaSuccess;
		Matrix result = Matrix(B.getWidth(), this->getHeight(), &error, 0);
		
		long Ticker = clock();

		this->copyMatrixToDevice();
		B.copyMatrixToDevice();

		multplyMatricesAndActivate <<<B.getWidth() * this->getHeight() / BLOCK_SIZE + 1, BLOCK_SIZE >>> (this->dataDevice, B.dataDevice, result.dataDevice, this->getWidth(), B.getWidth(), this->getHeight(), B.getHeight(), STEP_FUNCTION);

		result.copyMatrixToHost();

		Ticker = clock() - Ticker;
		std::cout << "Result: " << Ticker << std::endl;

		return result;
	}
};

int main(int argc, char** argv) {
	int error;
	Matrix A(10000, 5000, &error, 2, 0.1);
	if (error != cudaSuccess) return;
	Matrix B(1, 10000, &error, 2, 1);
	if (error != cudaSuccess) return;

	/*A.printMatrix();
	printf("\n");
	B.printMatrix();*/

	Matrix C = A * B;
	
	printf("\nResult: %d\n", C.getHeight());
	//printf("Value: %f", C.getData(0, 0));
	//C.printMatrix();

	A.deleteMatrix();
	B.deleteMatrix();
	C.deleteMatrix();

	return 0;
}