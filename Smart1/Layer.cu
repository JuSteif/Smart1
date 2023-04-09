#include "Layer.h"

Layer::Layer(int sizeInput, int sizeOutput, bool outputNeuron, Matrix* Inputs, Matrix* PreviousWeights, Matrix* PreviousErrorSignal) : outputNeuron(outputNeuron), Inputs(Inputs), PreviousWeights(PreviousWeights), PreviousErrorSignal(PreviousErrorSignal) {
	int error;
	Weights = Matrix(sizeInput, sizeOutput, &error, 2, 1);
	if (error != cudaSuccess) return;
	Outputs = Matrix(1, sizeOutput + 1, &error, 0);
	if (error != cudaSuccess) return;
	Outputs.setData(Outputs.getWidth() - 1, Outputs.getHeight() - 1, 1);
	ErrorSignal = Matrix(1, sizeOutput, &error, 0);
	if (error != cudaSuccess) return;
}

void Layer::deleteLayer() {
	Weights.deleteMatrix();
	Outputs.deleteMatrix();
}

void Layer::forward() {
	//Outputs = Weights * *Inputs;
	Outputs.Forward(*Inputs, Weights, SIGMOID_FUNCTION);
}

void Layer::calculatErrorSignal() {
	//*
	printf("\nPreviousErrorSignalFunction:\n");
	PreviousErrorSignal->printMatrix();

	if (outputNeuron) {
		//ErrorSignal = Outputs - *PreviousErrorSignal;
		ErrorSignal.SubstactTargetFromOutput(Outputs, *PreviousErrorSignal);
		printf("Last Layer\n");
	}
	else {
		printf("\nPreviousWeights:\n");
		PreviousWeights->printMatrix();

		ErrorSignal.multiplyAndSumMatrix(PreviousWeights, PreviousErrorSignal);
		//ErrorSignal = *PreviousWeights * *PreviousErrorSignal;
		printf("Hidden Layer\n");
	}
	printf("Error Signal\n");
	ErrorSignal.printMatrix();
	printf("\nErrorSignalFunctionAfter:\n");
	ErrorSignal.printMatrix();

	Outputs.multiplyWithDerivateMatrix(&ErrorSignal, SIGMOID_FUNCTION);
}

void Layer::calculateNewWeights(float learnRate) {
	Weights.calculateNewWeightsMatrix(Inputs, &ErrorSignal, learnRate);
}