#include "Matrix.cuh"

#pragma region ACTIVATION_FUNCTIONS

__device__ float sigmoid(float net) {
	return 1.0f / (1.0f + exp(-net));
}

__device__ float sigmoidDerivative(float out) {
	return out * (1 - out);
}

__device__ float step(float net) {
	if (net > 0.5) return 1;
	else return 0;
}

__device__ float stepDerivative(float out) {
	return 1;
}

__device__ float linear(float net) {
	return net;
}

__device__ float linearDerivative(float out) {
	return 1;
}

#pragma endregion

__global__ void multplyMatricesAndActivate(float* A, float* B, float* C, int widthA, int widthB, int heightA, int heightB, uint8_t activatioonFunction) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float* pA = &A[idx / widthB];
	float* pB = &B[idx % widthB];

	if (idx <= widthA * heightB) {
		float out = 0;
		for (int i = 0; i < heightB; i++) {
			out += *pA * *pB;
			pA++;
			pB += widthB;
		}

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
		C[idx] = out;
	}
}

int Matrix::copyMatrixToDevice() {
	return cudaMemcpy(dataDevice, dataHost, width * height * sizeof(float), cudaMemcpyHostToDevice);
}

int Matrix::copyMatrixToHost() {
	return cudaMemcpy(dataHost, dataDevice, width * height * sizeof(float), cudaMemcpyDeviceToHost);
}

Matrix::Matrix() {

}

Matrix::Matrix(int width, int height, int* errorStatus, uint8_t method, float seed) : height(height), width(width) {
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
				dataHost[i * width + j] = (float)rand() / (float)rand();
			}
		}
	}
	else if (method == 2)
	{
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				dataHost[i * width + j] = seed;
			}
		}
	}
}

void Matrix::deleteMatrix() {
	free(dataHost);
	cudaFree(dataDevice);
}

unsigned int Matrix::getWidth() {
	return width;
}

unsigned int Matrix::getHeight() {
	return height;
}

float Matrix::getData(unsigned int x, unsigned int y) {
	if (x > width || y > height) return 0;
	return dataHost[y * width + x];
}

void Matrix::setData(unsigned int x, unsigned int y, float value) {
	if (x > width || y > height) return;
	dataHost[y * width + x] = value;
}

void Matrix::printMatrix() {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			printf("%+9f ", dataHost[i * width + j]);
		}
		printf("\n");
	}
}

Matrix Matrix::operator * (Matrix B) {
	if (this->width != B.getHeight()) {
		return Matrix(0, 0, NULL);
	}

	int error = cudaSuccess;
	Matrix result = Matrix(B.getWidth(), this->getHeight(), &error, 0);

	long ticker = clock();

	this->copyMatrixToDevice();
	B.copyMatrixToDevice();

	multplyMatricesAndActivate <<<B.getWidth() * this->getHeight() / BLOCK_SIZE + 1, BLOCK_SIZE >>> (this->dataDevice, B.dataDevice, result.dataDevice, this->getWidth(), B.getWidth(), this->getHeight(), B.getHeight(), STEP_FUNCTION);

	result.copyMatrixToHost();

	ticker = clock() - ticker;
	std::cout << "Ticker: " << ticker << std::endl;

	return result;
}