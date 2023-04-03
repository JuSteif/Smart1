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

	if (idx <= widthA * heightB) {
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

__global__ void multiplyAndSum(float* weights, float* errorSignal, float* previousErrorSignal, int sizeError, int sizePreviousError) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < sizeError) {
		int sum = 0;
		for (int i = 0; i < sizePreviousError; i++) {
			sum += previousErrorSignal[0];// weights[sizeError * i + idx] * previousErrorSignal[i];
		}
		errorSignal[idx] = sum;
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
		int error;
		return Matrix(0, 0, &error);
		printf("Error ocurred\n");
	}

	int error = cudaSuccess;
	Matrix result = Matrix(B.getWidth(), this->getHeight(), &error, 0);

	long ticker = clock();

	this->copyMatrixToDevice();
	B.copyMatrixToDevice();

	multplyMatrices <<<B.getWidth() * this->getHeight() / BLOCK_SIZE + 1, BLOCK_SIZE >>> (this->dataDevice, B.dataDevice, result.dataDevice, this->getWidth(), B.getWidth(), this->getHeight(), B.getHeight());
	activateMatrices <<<B.getWidth() * this->getHeight() / BLOCK_SIZE + 1, BLOCK_SIZE >>> (result.dataDevice, result.getWidth(), result.getHeight(), SIGMOID_FUNCTION);

	result.copyMatrixToHost();

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

	this->copyMatrixToDevice();
	B.copyMatrixToDevice();

	substractMatrices <<<B.getWidth() * this->getHeight() / BLOCK_SIZE + 1, BLOCK_SIZE >>> (this->dataDevice, B.dataDevice, result.dataDevice, this->getWidth(), this->getHeight());

	result.copyMatrixToHost();

	return result;
}

Matrix Matrix::operator + (Matrix B) {
	if (this->width != B.getWidth() || this->height != B.getHeight()) {
		int error;
		return Matrix(0, 0, &error);
		printf("Error ocurred\n");
	}

	int error = cudaSuccess;
	Matrix result = Matrix(B.getWidth(), this->getHeight(), &error, 0);

	long ticker = clock();

	this->copyMatrixToDevice();
	B.copyMatrixToDevice();

	addMatrices << <B.getWidth() * this->getHeight() / BLOCK_SIZE + 1, BLOCK_SIZE >> > (this->dataDevice, B.dataDevice, result.dataDevice, this->getWidth(), this->getHeight());

	result.copyMatrixToHost();

	return result;
}

void Matrix::multiplyWithDerivateMatrix(Matrix* errorSignal, int activationFunction) {
	if(this->height != errorSignal->getHeight()) {
		int error;
		printf("Error ocurred in multiply with Derivate\n");
		printf("Output %d\tError%d", this->height, errorSignal->getHeight());
	}

	int error = cudaSuccess;

	this->copyMatrixToDevice();
	errorSignal->copyMatrixToDevice();

	multiplyWithDerivate <<<errorSignal->getWidth() * this->getHeight() / BLOCK_SIZE + 1, BLOCK_SIZE >>> (this->dataDevice, errorSignal->dataDevice, this->getHeight(), activationFunction);

	errorSignal->copyMatrixToHost();
}

void Matrix::calculateNewWeightsMatrix(Matrix* Inputs, Matrix* Error, float learnRate) {
	if (this->height != Error->getHeight() || this->width != Inputs->getHeight()) {
		int error;
		printf("Error ocurred\n");
	}

	int error = cudaSuccess;

	Error->copyMatrixToDevice();
	Inputs->copyMatrixToDevice();
	printf("\n");
	Error->printMatrix();
	printf("\n");
	Inputs->printMatrix();
	printf("\n");
	this->printMatrix();


	calculateNewWeights <<<this->getWidth() * this->getHeight() / BLOCK_SIZE + 1, BLOCK_SIZE >>> (this->dataDevice, Error->dataDevice, Inputs->dataDevice, this->getHeight(), this->getWidth(), learnRate);

	this->copyMatrixToHost();
}

void Matrix::multiplyAndSumMatrix(Matrix* weights, Matrix* previousErrorSignal) {
	if (this->height != weights->getWidth() || weights->getHeight() != previousErrorSignal->getHeight()) {
		int error;
		printf("Error ocurred in Multiply and Sum\n");
	}

	int error = cudaSuccess;

	weights->dataHost[1] = 2;

	weights->copyMatrixToDevice();
	previousErrorSignal->copyMatrixToDevice();
	printf("Previous ErrorSignal\n------------------------------------------------\n");
	previousErrorSignal->printMatrix();

	printf("ErrorSignal Größe %d\n", this->height);
	printf("PreviousErrorSignal Größe %d\n", previousErrorSignal->getHeight());
	printf("Weights x %d   y %d\n", weights->getWidth(), weights->getHeight());

	multiplyAndSum <<<this->getWidth() * this->getHeight() / BLOCK_SIZE + 1, BLOCK_SIZE >>> (weights->dataDevice, this->dataDevice, previousErrorSignal->dataDevice, this->getHeight(), previousErrorSignal->getHeight());

	this->copyMatrixToHost();
	this->printMatrix();
}