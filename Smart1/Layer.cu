#include "Layer.h"

Layer::Layer(int sizeInput, int sizeOutput, bool outputNeuron, Matrix* Inputs, Matrix* PreviousWeights, Matrix* PreviousErrorSignal) : outputNeuron(outputNeuron), Inputs(Inputs), PreviousWeights(PreviousWeights), PreviousErrorSignal(PreviousErrorSignal) {
	int error;
	Weights = Matrix(sizeInput, sizeOutput, &error, 2, 1);
	if (error != cudaSuccess) return;
	Outputs = Matrix(1, sizeOutput, &error, 0);
	if (error != cudaSuccess) return;
	ErrorSignal = Matrix(1, sizeOutput, &error, 0);
	if (error != cudaSuccess) return;
}

void Layer::deleteLayer() {
	Weights.deleteMatrix();
	Outputs.deleteMatrix();
}

void Layer::forward() {
	Outputs = Weights * *Inputs;
}

void Layer::calculatErrorSignal() {
	//*
	printf("\nPreviousErrorSignalFunction:\n");
	PreviousErrorSignal->printMatrix();
	/*printf("\nOutputs:\n");
	Outputs.printMatrix();
	//*/

	if (outputNeuron) {
		ErrorSignal = Outputs - *PreviousErrorSignal;
		printf("Last Layer\n");
	}
	else {
		printf("\nPreviousWeights:\n");
		PreviousWeights->printMatrix();

		ErrorSignal.multiplyAndSumMatrix(PreviousWeights, PreviousErrorSignal);
		//ErrorSignal = *PreviousWeights * *PreviousErrorSignal;
		printf("Hidden Layer\n");
	}
	//*
	printf("\nErrorSignalFunctionAfter:\n");
	ErrorSignal.printMatrix();
	//*/

	Outputs.multiplyWithDerivateMatrix(&ErrorSignal, SIGMOID_FUNCTION);
}

void Layer::calculateNewWeights(float learnRate) {
	Weights.calculateNewWeightsMatrix(Inputs, &ErrorSignal, learnRate);
}