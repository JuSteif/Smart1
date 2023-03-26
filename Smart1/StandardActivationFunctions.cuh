#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>

#define SIGMOID_FUNCTION 0
#define LINEAR_FUNCTION 1
#define STEP_FUNCTION 2

__device__ float sigmoid(float net);

__device__ float sigmoidDerivative(float out);

__device__ float step(float net);

__device__ float stepDerivative(float out);

__device__ float linear(float net);

__device__ float linearDerivative(float out);

#endif

