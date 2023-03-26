#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>

#define SIGMOID_FUNCTION 0
#define LINEAR_FUNCTION 1
#define STEP_FUNCTION 2

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