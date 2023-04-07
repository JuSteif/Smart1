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

__global__ void multplyMatrices(float* A, float* B, float* C, int widthA, int widthB, int heightA, int heightB) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float* pA = &A[idx / widthB];
	float* pB = &B[idx % widthB];

	if (idx < widthA * heightB) {
		float out = 0;
		for (int i = 0; i < heightB; i++) {
			out += *pA * *pB;
			pA++;
			pB += widthB;
		}

		C[idx] = out;
	}
}

__global__ void activateMatrices(float* A, int widthA, int heightA, uint8_t activatioonFunction) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx <= widthA * heightA) {
		switch (activatioonFunction) {
		case SIGMOID_FUNCTION:
			A[idx] = sigmoid(A[idx]);
			break;
		case STEP_FUNCTION:
			A[idx] = step(A[idx]);
			break;
		case LINEAR_FUNCTION:
			A[idx] = linear(A[idx]);
			break;
		}
	}
}

__global__ void substractMatrices(float* A, float* B, float* C, int widthA, int heightA) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx <= widthA * heightA) {
		C[idx] = A[idx] - B[idx];
	}
}

__global__ void addMatrices(float* A, float* B, float* C, int widthA, int heightA) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx <= widthA * heightA) {
		C[idx] = A[idx] + B[idx];
	}
}

__global__ void multiplyWithDerivate(float* outputs, float* errorSignal, int sizeOutputs, uint8_t activatioonFunction) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < sizeOutputs) {
		switch (activatioonFunction) {
		case SIGMOID_FUNCTION:
			errorSignal[idx] *= sigmoidDerivative(outputs[idx]);
			break;
		case STEP_FUNCTION:
			errorSignal[idx] *= stepDerivative(outputs[idx]);
			break;
		case LINEAR_FUNCTION:
			errorSignal[idx] *= linearDerivative(outputs[idx]);
			break;
		}
	}
}

__global__ void calculateNewWeights(float* weights, float* Error, float* Inputs, int sizeError, int sizeInputs, float learnRate) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < sizeError * sizeInputs) {
		weights[idx] += -learnRate * Error[idx / sizeInputs] * Inputs[idx % sizeInputs];
	}
}

__global__ void multiplyAndSum(float* weights, float* previousErrorSignal, float* errorSignal, int sizeError, int sizePreviousError) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < sizeError) {
		float sum = 0;
		for (int i = 0; i < sizePreviousError; i++) {
			sum += weights[sizeError * i + idx] * previousErrorSignal[i];
		}
		errorSignal[idx] = sum;
	}
}

int Matrix::copyMatrixToDevice(int leftTopX, int leftTopY, int bottomRightX, int bottomRightY) {
	//printf("Device TopX: %d TopY %d BottomX: %d BottomY %d\n", leftTopX, leftTopY, bottomRightX, bottomRightY);
	int newWidth = bottomRightX - leftTopX + 1;
	int newHeight = bottomRightY - leftTopY + 1;

	int error = cudaSuccess;
	cudaFree(dataDevice);
	error = cudaMalloc(&dataDevice, width * height * sizeof(float));
	if (error != cudaSuccess) {
		printf("error while allocating\n");
		return error;
	}
	
	for (int i = leftTopY; i <= bottomRightY; i++) {
		//printf(" i: %d %d %d %d %d\n", i, newWidth * (i - leftTopY), i * width + leftTopX, newWidth, newWidth * newHeight);
		error = cudaMemcpy(&dataDevice[newWidth * (i - leftTopY)], &dataHost[i * width + leftTopX], newWidth * sizeof(float), cudaMemcpyHostToDevice);
		
		if (error != cudaSuccess) {
			printf("error while copying\n");
			return error;
		}
	}
	return error;
}

int Matrix::copyMatrixToHost(int leftTopX, int leftTopY, int bottomRightX, int bottomRightY) {
	//printf("Host TopX: % d TopY % d BottomX : % d BottomY % d\n", leftTopX, leftTopY, bottomRightX, bottomRightY);
	int newWidth = bottomRightX - leftTopX + 1;
	int newHeight = bottomRightY - leftTopY + 1;

	int error = cudaSuccess;
	for (int i = leftTopY; i <= bottomRightY; i++) {
		//printf(" i: %d %d %d %d %d\n", i, newWidth * (i - leftTopY), i * width + leftTopX, newWidth, newWidth * newHeight);
		cudaMemcpy(&dataHost[i * width + leftTopX], &dataDevice[newWidth * (i - leftTopY)], (newWidth) * sizeof(float), cudaMemcpyDeviceToHost);
		if (error != cudaSuccess) {
			printf("error while copying\n");
			return error;
		}
	}
	return error;
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
		int error;
		return Matrix(0, 0, &error);
		printf("Error ocurred in Matrixmultiplication\n");
	}

	int error = cudaSuccess;
	Matrix result = Matrix(B.getWidth(), this->getHeight(), &error, 0);

	long ticker = clock();

	this->copyMatrixToDevice(0, 0, this->getWidth() - 1, this->getHeight());
	B.copyMatrixToDevice(0, 0, B.getWidth() - 1, B.getHeight());

	multplyMatrices <<<B.getWidth() * this->getHeight() / BLOCK_SIZE + 1, BLOCK_SIZE >>> (this->dataDevice, B.dataDevice, result.dataDevice, this->getWidth(), B.getWidth(), this->getHeight(), B.getHeight());
	activateMatrices <<<B.getWidth() * this->getHeight() / BLOCK_SIZE + 1, BLOCK_SIZE >>> (result.dataDevice, result.getWidth(), result.getHeight(), SIGMOID_FUNCTION);

	result.copyMatrixToHost(0, 0, result.getWidth() - 1, result.getHeight());

	ticker = clock() - ticker;
	std::cout << "Ticker: " << ticker << std::endl;

	return result;
}

Matrix Matrix::operator - (Matrix B) {
	if (this->width != B.getWidth() || this->height != B.getHeight()) {
		int error;
		return Matrix(0, 0, &error);
		printf("Error ocurred\n");
	}

	int error = cudaSuccess;
	Matrix result = Matrix(B.getWidth(), this->getHeight(), &error, 0);

	long ticker = clock();

	this->copyMatrixToDevice(0, 0, this->getWidth() - 1, this->getHeight());
	B.copyMatrixToDevice(0, 0, B.getWidth() - 1, B.getHeight());

	substractMatrices <<<B.getWidth() * this->getHeight() / BLOCK_SIZE + 1, BLOCK_SIZE >>> (this->dataDevice, B.dataDevice, result.dataDevice, this->getWidth(), this->getHeight());

	result.copyMatrixToHost(0, 0, result.getWidth() - 1, result.getHeight());

	return result;
}

Matrix Matrix::operator + (Matrix B) {
	if (this->width != B.getWidth() || this->height != B.getHeight()) {
		int error;
		return Matrix(0, 0, &error);
		printf("Error ocurred in Matrixaddition\n");
	}

	int error = cudaSuccess;
	Matrix result = Matrix(B.getWidth(), this->getHeight(), &error, 0);

	long ticker = clock();

	this->copyMatrixToDevice(0, 0, this->getWidth() - 1, this->getHeight());
	B.copyMatrixToDevice(0, 0, B.getWidth() - 1, B.getHeight());

	addMatrices << <B.getWidth() * this->getHeight() / BLOCK_SIZE + 1, BLOCK_SIZE >> > (this->dataDevice, B.dataDevice, result.dataDevice, this->getWidth(), this->getHeight());

	result.copyMatrixToHost(0, 0, result.getWidth() - 1, result.getHeight());

	return result;
}

void Matrix::multiplyWithDerivateMatrix(Matrix* errorSignal, int activationFunction) {
	if(this->height != errorSignal->getHeight()) {
		int error;
		printf("Error ocurred in multiply with Derivate\n");
		printf("Output %d\tError%d", this->height, errorSignal->getHeight());
	}

	int error = cudaSuccess;

	this->copyMatrixToDevice(0, 0, this->getWidth() - 1, this->getHeight());
	errorSignal->copyMatrixToDevice(0, 0, errorSignal->getWidth() - 1, errorSignal->getHeight());

	multiplyWithDerivate <<<errorSignal->getWidth() * this->getHeight() / BLOCK_SIZE + 1, BLOCK_SIZE >>> (this->dataDevice, errorSignal->dataDevice, this->getHeight(), activationFunction);

	errorSignal->copyMatrixToHost(0, 0, errorSignal->getWidth() - 1, errorSignal->getHeight());
}

void Matrix::calculateNewWeightsMatrix(Matrix* Inputs, Matrix* Error, float learnRate) {
	if (this->height != Error->getHeight() || this->width != Inputs->getHeight()) {
		int error;
		printf("Error ocurred\n");
	}

	int error = cudaSuccess;

	Error->copyMatrixToDevice(0, 0, Error->getWidth() - 1, Error->getHeight());
	Inputs->copyMatrixToDevice(0, 0, Inputs->getWidth() - 1, Inputs->getHeight());
	printf("\n");
	Error->printMatrix();
	printf("\n");
	Inputs->printMatrix();
	printf("\n");
	this->printMatrix();


	calculateNewWeights <<<this->getWidth() * this->getHeight() / BLOCK_SIZE + 1, BLOCK_SIZE >>> (this->dataDevice, Error->dataDevice, Inputs->dataDevice, this->getHeight(), this->getWidth(), learnRate);

	this->copyMatrixToHost(0, 0, this->getWidth() - 1, this->getHeight());
}

void Matrix::multiplyAndSumMatrix(Matrix* weights, Matrix* previousErrorSignal) {
	if (this->height != weights->getWidth() || weights->getHeight() != previousErrorSignal->getHeight()) {
		int error;
		printf("Error ocurred in Multiply and Sum\n");
	}

	int error = cudaSuccess;

	weights->copyMatrixToDevice(0, 0, weights->getWidth() - 1, weights->getHeight());
	previousErrorSignal->copyMatrixToDevice(0, 0, previousErrorSignal->getWidth() - 1, previousErrorSignal->getHeight());

	multiplyAndSum <<<this->getWidth() * this->getHeight() / BLOCK_SIZE + 1, BLOCK_SIZE >>> (weights->dataDevice, previousErrorSignal->dataDevice, this->dataDevice, this->getHeight(), previousErrorSignal->getHeight());

	this->copyMatrixToHost(0, 0, this->getWidth() - 1, this->getHeight() - 1);
}

void Matrix::Forward(Matrix& Inputs, Matrix& Weights, uint8_t activationFunction) {
	if (Weights.width != Inputs.getHeight()) {
		int error;
		printf("Error ocurred in Matrixmultiplication WeightsWidth %d InputsHeight %d\n", Weights.getWidth(), Inputs.getHeight());
		return;
	}
	//printf("WeightsWidth %d AHeight %d\n", Weights.getWidth(), Inputs.getHeight());

	Inputs.copyMatrixToDevice(0, 0, Inputs.getWidth() - 1, Inputs.getHeight() - 1);
	Weights.copyMatrixToDevice(0, 0, Weights.getWidth() - 1, Weights.getHeight() - 1);

	multplyMatrices <<<Weights.getHeight() * Inputs.getWidth() / BLOCK_SIZE + 1, BLOCK_SIZE >>> (Weights.dataDevice, Inputs.dataDevice, this->dataDevice, Weights.getWidth(), Inputs.getWidth(), Weights.getHeight(), Inputs.getHeight());
	activateMatrices <<<Weights.getHeight() * Inputs.getWidth() / BLOCK_SIZE + 1, BLOCK_SIZE >>> (this->dataDevice, this->getWidth(), this->getHeight() - 1, activationFunction);

	this->copyMatrixToHost(0, 0, this->getWidth() - 1, this->getHeight() - 2);
}